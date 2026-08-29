import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """Generalized Mean Pooling (Radenovic et al., 2018) with a learnable p.
    p=1 is average pooling, p->inf is max pooling; init at 3.0."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # Force fp32. Under autocast this runs in fp16, and pow(p) with p around
        # 3 overflows: any activation above about 40 cubed passes fp16's 65504
        # ceiling, becomes inf, and pooling turns that into NaN. It can survive
        # a dozen epochs and then poison a whole fold's predictions.
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            x = x.clamp(min=self.eps).pow(self.p)
            x = F.adaptive_avg_pool2d(x, 1)
            return x.pow(1.0 / self.p).flatten(1)


def pool_backbone_features(feats, num_features, gem):
    """GeM-pool a timm backbone's unpooled output, handling the three layouts
    timm returns: (B,C,H,W) CNNs, (B,H,W,C) swin, (B,N,C) ViT token sequences."""
    if feats.dim() == 4:
        if feats.shape[1] == num_features:
            pass
        elif feats.shape[-1] == num_features:
            feats = feats.permute(0, 3, 1, 2).contiguous()  # channels-last -> channels-first
        else:
            raise ValueError(
                f"Backbone output {tuple(feats.shape)} doesn't match num_features={num_features}"
            )
        return gem(feats)

    if feats.dim() == 3:
        return feats.mean(dim=1)  # token sequence: no spatial grid, mean over tokens

    return feats


class TabularEmbedding(nn.Module):
    """Entity embeddings for sex/site + a small MLP for age.

    sex_idx/site_idx are the raw label codes from step2_make_folds (sex 0-2,
    site 0-6); age is the normalised age_norm column.
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
    """fused = img_feats + gate(meta) * project(meta).

    The gate scales an additive metadata contribution rather than multiplying
    the image features, so noisy metadata can only add nothing (gate->0), never
    zero out the image branch.
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
    """Vision backbone (GeM-pooled) fused with tabular metadata via a gated
    residual, for SIIM-ISIC melanoma classification.

    use_metadata=False runs image-only; use_gem=False falls back to the
    backbone's own average pooling.
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
        """Everything except the backbone, for the differential-LR param groups."""
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
    return SkinMelanomaFinalModel(backbone_name=backbone_name, **kwargs)
