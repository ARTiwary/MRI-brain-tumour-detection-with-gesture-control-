# model.py
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────
# MODEL 1 — Custom CNN
# Test acc : 92.00% | Val acc : 96.70%
# Weights  : tumor-detection-model.pth
# ─────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1 — 128×128 → 64×64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2 — 64×64 → 32×32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 3 — 32×32 → 16×16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 4 — 16×16 → 8×8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    @property
    def last_conv_layer(self):
        """Last Conv2d in features — target layer for Grad-CAM."""
        for m in reversed(list(self.features.children())):
            if isinstance(m, nn.Conv2d):
                return m
        raise ValueError("No Conv2d found in CNN.features")


# ─────────────────────────────────────────
# MODEL 2 — ResNet18
# Test acc : 94.81% | Val acc : 97.32%
# Weights  : tumor-detection-resnet18.pth
# ─────────────────────────────────────────
class ResNetModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetModel, self).__init__()

        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features          # 512

        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)

    @property
    def last_conv_layer(self):
        """layer4[-1].conv2 — correct Grad-CAM target for ResNet18."""
        return self.model.layer4[-1].conv2