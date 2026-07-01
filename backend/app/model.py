# model.py
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────
# MODEL 1 — Custom CNN
# Test acc : 89.94% | Val acc : 93.84%  (class-weighted, focal loss)
# Weights  : tumor-detection-cnn-128.pth
# Input    : 128×128
# Per-class sensitivity (test): Glioma 80.75% | Meningioma 80.75% |
#            Pituitary 99.00% | No Tumor 99.25%
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
        """Last Conv2d in features — target layer for Grad-CAM.
        Matches the notebook's Section 14B Grad-CAM target (last conv before
        flattening into the classifier head)."""
        for m in reversed(list(self.features.children())):
            if isinstance(m, nn.Conv2d):
                return m
        raise ValueError("No Conv2d found in CNN.features")


# ─────────────────────────────────────────
# MODEL 2 — ResNet18 (transfer learning)
# Test acc : 95.12% | Val acc : 99.02%  (class-weighted, focal loss)
# Weights  : tumor-detection-resnet18-224.pth
# Input    : 224×224, ImageNet normalization
# Per-class sensitivity (test): Glioma 83.75% | Meningioma 97.47% |
#            Pituitary 99.26% | No Tumor 100.00%
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
        """layer4[-1] (full residual block, post-addition) — matches the
        Grad-CAM target actually used and validated in the training notebook
        (Section 22B: `resnet_target_layer = resnet_model.layer4[-1]`).
        NOTE: this is the whole block, not `.conv2` — using `.conv2` instead
        hooks the pre-residual-addition activations and produces a different,
        non-equivalent heatmap. Keep this consistent with the notebook."""
        return self.model.layer4[-1]


# ─────────────────────────────────────────
# MODEL 3 — EfficientNet-B0 (transfer learning)
# Test acc : 95.75% | Val acc : 99.11%  (class-weighted, focal loss)
# Weights  : tumor-detection-efficientnet-b0-224.pth
# Input    : 224×224, ImageNet normalization
# Per-class sensitivity (test): Glioma 85.75% | Meningioma 98.28% |
#            Pituitary 99.03% | No Tumor 100.00%
# Best overall accuracy AND fewest parameters of the three models —
# preferred choice for deployment if inference speed/size matters.
# ─────────────────────────────────────────
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=4):
        super(EfficientNetModel, self).__init__()

        backbone = models.efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features    # 1280

        backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)

    @property
    def last_conv_layer(self):
        """features[-1] — final conv block (head conv) before pooling.
        Matches the Grad-CAM target used in the training notebook
        (Section 22F: `eff_target_layer = eff_model.features[-1]`)."""
        return self.model.features[-1]


# ─────────────────────────────────────────
# Registry — single source of truth for backend/frontend model selection.
# Mirrors the structure already referenced in main.py's AVAILABLE_MODELS.
# ─────────────────────────────────────────
AVAILABLE_MODELS = {
    "cnn": {
        "label": "Custom CNN (128×128)",
        "class": CNN,
        "kwargs": {"num_classes": 4},
        "weights_file": "tumor-detection-cnn-128.pth",
        "input_size": (3, 128, 128),
    },
    "resnet18": {
        "label": "ResNet18 (224×224, transfer learning)",
        "class": ResNetModel,
        "kwargs": {"num_classes": 4},
        "weights_file": "tumor-detection-resnet18-224.pth",
        "input_size": (3, 224, 224),
    },
    "efficientnet_b0": {
        "label": "EfficientNet-B0 (224×224, transfer learning)",
        "class": EfficientNetModel,
        "kwargs": {"num_classes": 4},
        "weights_file": "tumor-detection-efficientnet-b0-224.pth",
        "input_size": (3, 224, 224),
    },
}