import timm
import torch.nn as nn


class CustomCNN(nn.Module):
    """
    A small from-scratch CNN, useful as a sanity baseline.

    If a pretrained resnet34 cannot beat this, something is wrong with the
    training setup rather than with the architecture.
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Dropout before the classifier. With only ~5,000 melanoma photos this
        # model will otherwise memorise them.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_model(model_name='resnet34', pretrained=True, dropout=0.3):
    """
    num_classes=1 gives a single logit. Pair it with BCEWithLogitsLoss and do
    NOT put a sigmoid in the model, or the loss applies one a second time.
    """
    if model_name == 'custom_cnn':
        return CustomCNN(dropout=dropout)
    return timm.create_model(model_name, pretrained=pretrained,
                             num_classes=1, drop_rate=dropout)
