import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # This matches your notebook: 3 Conv blocks
        self.cnn_model = nn.Sequential(
            # Block 1: 3 -> 16
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),

            # Block 2: 16 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            # Block 3: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )

        # Fully Connected Layers
        # 128 / 2 / 2 / 2 = 16. So feature map is 16x16
        self.fc_model = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)  # 4 classes: Glioma, Meningioma, Pituitary, No Tumor
        )

    def forward(self, x):
        x = self.cnn_model(x)
        # Flatten the output for the linear layers
        x = x.view(x.size(0), -1) 
        x = self.fc_model(x)
        return x