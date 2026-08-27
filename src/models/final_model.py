import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling (Radenovic et al., 2018). A learnable
    interpolation between average pooling (p=1) and max pooling (p -> inf):

        gem(x) = ( mean(x.clamp(min=eps) ** p) ) ** (1/p)

    Melanoma is diagnosed from a few salient structures (irregular borders,
    colour variation, a specific pigment network) sitting inside a lesion
    that is otherwise fairly uniform skin. Plain average pooling blends that
    small salient region in with everything else in the feature map; GeM's
    learnable `p` lets the network anneal toward emphasising the peak
    activations (the salient structure) instead, without hand-picking max
    vs. average up front.

    `p` is a learned scalar, initialised at 3.0 (a common default that starts
    partway between mean and max) via nn.Parameter, so it is optimised by the
    same optimiser as everything else.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W), channels-first.
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / self.p).flatten(1)


def pool_backbone_features(feats, num_features, gem):
    """
    Reduces a timm backbone's raw (unpooled) output to a (B, num_features)
    vector with GeM pooling, regardless of which of the three output layouts
    that backbone uses -- this matters because they are NOT interchangeable:

      - CNNs (efficientnet, convnext, resnet...): (B, C, H, W), channels-first.
        GeM applies directly.
      - Some transformer backbones (swin...): (B, H, W, C), channels-LAST.
        Applying GeM without permuting first would pool over the wrong axes
        and silently produce garbage -- there is no error, just a model that
        trains poorly for a reason that never shows up in a shape check.
      - Pure ViT-style backbones (eva02...): (B, N, C), a flat token sequence
        with no 2D spatial structure at all (patch tokens + a class token).
        GeM has nothing to pool over spatially here, so we mean-pool over the
        token dimension instead as the sane fallback.

    Detecting channels-first vs. channels-last is done by checking which
    dimension actually equals num_features, rather than assuming a layout by
    architecture family, so this keeps working for backbones not in this
    docstring.
    """
    if feats.dim() == 4:
        if feats.shape[1] == num_features:
            pass  # already channels-first: (B, C, H, W)
        elif feats.shape[-1] == num_features:
            feats = feats.permute(0, 3, 1, 2).contiguous()  # -> (B, C, H, W)
        else:
            raise ValueError(
                f"Backbone output shape {tuple(feats.shape)} doesn't match "
                f"num_features={num_features} on either the channel-first or "
                "channel-last axis; this backbone's output layout needs its "
                "own case in pool_backbone_features()."
            )
        return gem(feats)

    if feats.dim() == 3:
        # Token sequence (ViT-style): mean over the token axis, class token
        # included. No spatial grid to run GeM over.
        return feats.mean(dim=1)

    # Already pooled to (B, num_features) by the backbone itself.
    return feats


class TabularEmbedding(nn.Module):
    """
    Embeds the categorical metadata instead of one-hot + linear. Entity
    embeddings generally beat one-hot for low-cardinality categoricals
    because the network can place similar categories near each other in
    embedding space -- e.g. "palms/soles" and "oral/genital" can end up nearer
    each other than either is to "torso" if that's what the data supports --
    instead of one-hot's built-in assumption that every category is equally
    (maximally) different from every other one.

    sex_idx and site_idx are the raw label-encoded columns from
    step2_make_folds.py (sex_enc: 0/1/2, site_enc: 0-6), not one-hot vectors.
    age is the already-normalised age_norm column (age / 90).
    """

    def __init__(self, num_sex=3, num_site=7, sex_dim=8, site_dim=8,
                age_dim=16, out_dim=256, dropout=0.3):
        super().__init__()
        self.sex_embed = nn.Embedding(num_sex, sex_dim)
        self.site_embed = nn.Embedding(num_site, site_dim)
        self.age_mlp = nn.Sequential(
            nn.Linear(1, age_dim),
            nn.LayerNorm(age_dim),
            nn.GELU(),
        )
        combined_dim = sex_dim + site_dim + age_dim
        self.project = nn.Sequential(
            nn.Linear(combined_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, sex_idx, site_idx, age):
        sex_e = self.sex_embed(sex_idx)
        site_e = self.site_embed(site_idx)
        age_e = self.age_mlp(age)
        return self.project(torch.cat([sex_e, site_e, age_e], dim=1))


class GatedFusion(nn.Module):
    """
    Gated multimodal fusion, replacing plain concatenation.

    A gate in (0, 1), predicted from the metadata embedding, scales an
    additive metadata contribution before it's added onto the image features:

        fused = img_feats + gate(meta) * project(meta)

    This deliberately is NOT `img_feats * gate` (a literal 0-1 scaling gate
    multiplying the image features themselves). A multiplicative gate that
    can hit exactly 0 lets a single noisy metadata row -- a missing age, a
    rare anatomical site -- silence the image branch completely, which is the
    opposite of "tabular data cannot overpower the visual signal": it would
    let bad tabular data erase good visual data. With the additive-residual
    form here, the image features are always fully present unchanged; the
    gate only controls how much (from none, at gate=0, up to the full
    projected embedding, at gate=1) metadata gets ADDED on top. Worst case
    for noisy metadata is "contributes nothing", never "destroys the image
    signal".
    """

    def __init__(self, feat_dim, meta_dim, dropout=0.3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(meta_dim, feat_dim),
            nn.Sigmoid(),
        )
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, img_feats, meta_feats):
        gate = self.gate(meta_feats)
        contribution = self.meta_proj(meta_feats)
        return img_feats + gate * contribution


class SkinMelanomaFinalModel(nn.Module):
    """
    Multi-modal final architecture for SIIM-ISIC melanoma classification.

    Vision backbone (any timm model; efficientnet_b4/convnext/swin all work,
    see pool_backbone_features) -> GeM pooling -> projected to `proj_dim`.
    Metadata (sex, anatom_site, age) -> entity embeddings -> gated fusion onto
    the image features -> LayerNorm -> classifier.

    `use_gem=False` falls back to the backbone's own built-in average
    pooling, kept for backbones where GeM either doesn't apply cleanly or
    isn't worth the extra parameter (e.g. quick ablations).
    """

    def __init__(
        self,
        backbone_name='tf_efficientnet_b4_ns',
        pretrained=True,
        drop_rate=0.3,
        meta_dropout=0.3,
        proj_dim=256,
        use_gem=True,
        gem_p=3.0,
        num_sex=3,
        num_site=7,
        use_metadata=True,
    ):
        super().__init__()
        self.use_gem = use_gem
        self.use_metadata = use_metadata

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='' if use_gem else 'avg',
            drop_rate=drop_rate,
        )
        backbone_dim = self.backbone.num_features
        self.gem = GeM(p=gem_p) if use_gem else None

        self.image_proj = nn.Sequential(
            nn.Linear(backbone_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.SiLU(),
            nn.Dropout(drop_rate),
        )

        if use_metadata:
            self.meta_embed = TabularEmbedding(
                num_sex=num_sex, num_site=num_site, out_dim=proj_dim, dropout=meta_dropout,
            )
            self.fusion = GatedFusion(proj_dim, proj_dim, dropout=drop_rate)
        else:
            self.meta_embed = None
            self.fusion = None

        self.fusion_norm = nn.LayerNorm(proj_dim)
        self.classifier = nn.Linear(proj_dim, 1)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        """Everything that isn't the backbone -- image_proj, meta_embed,
        fusion, fusion_norm, classifier, and gem's learned p. Used to build
        the differential-LR optimiser param groups in train_final.py."""
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in backbone_ids]

    def forward_image_features(self, image):
        feats = self.backbone(image)
        if self.use_gem:
            feats = pool_backbone_features(feats, self.backbone.num_features, self.gem)
        return feats

    def forward(self, image, metadata=None):
        img_feats = self.image_proj(self.forward_image_features(image))

        if self.meta_embed is None or metadata is None or metadata.shape[1] == 0:
            return self.classifier(self.fusion_norm(img_feats))

        sex_idx = metadata[:, 0].long()
        site_idx = metadata[:, 1].long()
        age = metadata[:, 2:3].float()
        meta_feats = self.meta_embed(sex_idx, site_idx, age)

        fused = self.fusion(img_feats, meta_feats)
        fused = self.fusion_norm(fused)
        return self.classifier(fused)


def build_final_model(backbone_name='tf_efficientnet_b4_ns', **kwargs):
    """Small factory so callers configure the model from a dict/argparse
    namespace without importing the class directly."""
    return SkinMelanomaFinalModel(backbone_name=backbone_name, **kwargs)
