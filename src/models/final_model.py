import torch
import torch.nn as nn
import timm


class MetadataEncoder(nn.Module):
    """
    Turns the tabular metadata (age, sex, anatom_site_general_challenge -- one
    numeric column and two one-hot encoded categoricals, ~11 features total)
    into a fixed-size embedding.

    Heavier dropout than the image branch is deliberate: 11 numeric columns
    are trivial for a network to memorise, and with ~2,000 unique patients in
    the training data, an under-regularised metadata branch will happily
    learn "this age/site combination = melanoma" instead of anything about
    the lesion itself.
    """

    def __init__(self, num_meta_features, hidden_dim=64, out_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_meta_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
        )

    def forward(self, x):
        return self.net(x)


class SkinMelanomaFinalModel(nn.Module):
    """
    Multi-modal final architecture for SIIM-ISIC melanoma classification.

    An ImageNet-pretrained vision backbone (any timm model -- convnext_tiny
    and efficientnet_b3 are the two this project targets) is fused with a
    small tabular metadata branch.

    Image features and metadata features are each projected into the same
    `proj_dim` space before fusion. That buys two things:
      1. the fusion layer is a plain concat + linear instead of a huge
         (backbone_dim + meta_dim) -> hidden layer that changes shape every
         time the backbone changes;
      2. a residual connection becomes possible. `use_residual=True` adds the
         image projection back onto the fused vector before the classifier,
         so the model always has a direct shortcut to the (harder to
         overfit) image signal even if the metadata branch's contribution is
         noisy early in training.
    """

    def __init__(
        self,
        backbone_name='convnext_tiny',
        num_meta_features=11,
        pretrained=True,
        drop_rate=0.3,
        meta_dropout=0.3,
        proj_dim=256,
        use_residual=True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.num_meta_features = num_meta_features

        # 1. Image backbone. num_classes=0 makes timm return the pooled
        # feature vector instead of class logits, so the same line works for
        # convnext_tiny, efficientnet_b3, resnet.. any timm model.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
        )
        backbone_dim = self.backbone.num_features

        self.image_proj = nn.Sequential(
            nn.Linear(backbone_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.SiLU(),
            nn.Dropout(drop_rate),
        )

        # 2. Metadata branch. Skipped entirely at forward time if
        # num_meta_features is 0, so this class also works as an image-only
        # ablation without a second model definition.
        if num_meta_features > 0:
            self.meta_encoder = MetadataEncoder(
                num_meta_features, hidden_dim=64, out_dim=proj_dim, dropout=meta_dropout,
            )
        else:
            self.meta_encoder = None

        # 3. Fusion head + residual connection back to the image projection.
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.SiLU(),
            nn.Dropout(drop_rate),
        )

        # LayerNorm on the fused vector, after the residual add, right before
        # the classifier. BatchNorm above is fine during training (batches of
        # 32+), but it leans on running statistics at inference, which get
        # unreliable at the small/odd batch sizes a deployed model actually
        # sees (a batch of 1 for a single uploaded photo, say). LayerNorm
        # normalises per-sample instead, so the classifier's input is stable
        # no matter what batch size it is called with.
        self.fusion_norm = nn.LayerNorm(proj_dim)

        self.classifier = nn.Linear(proj_dim, 1)

    def forward(self, image, metadata=None):
        img_feats = self.image_proj(self.backbone(image))

        if self.meta_encoder is None or metadata is None or metadata.shape[1] == 0:
            return self.classifier(self.fusion_norm(img_feats))

        meta_feats = self.meta_encoder(metadata)
        fused = self.fusion(torch.cat([img_feats, meta_feats], dim=1))

        if self.use_residual:
            fused = fused + img_feats

        fused = self.fusion_norm(fused)
        return self.classifier(fused)


def build_final_model(backbone_name='convnext_tiny', num_meta_features=11, **kwargs):
    """Small factory so callers configure the model from a dict/argparse
    namespace without importing the class directly."""
    return SkinMelanomaFinalModel(
        backbone_name=backbone_name, num_meta_features=num_meta_features, **kwargs
    )
