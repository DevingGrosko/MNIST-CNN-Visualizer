import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn

class Forward(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Input: [batch, 1, 28, 28]
            # Learn 8 different 3x3 filters
            # Padding=1 keeps spatial size at 28x28
            nn.Conv2d(
                in_channels=1,  # grayscale input
                out_channels=8,  # 8 learned feature detectors
                kernel_size=3,  # each looks at a 3x3 patch
                stride=1,
                padding=1  # preserves height & width
            ),
            # Remove negative responses
            nn.ReLU(),

            # Downsample spatial dimensions
            # 28x28 → 14x14
            nn.MaxPool2d(kernel_size=2),

            # Input now: [batch, 8, 14, 14]
            # Learn 16 higher-level feature detectors
            nn.Conv2d(
                in_channels=8,  # must match previous out_channels
                out_channels=16,  # increase feature richness
                kernel_size=3,
                stride=1,
                padding=1  # keeps 14x14
            ),
            nn.ReLU(),
            # Downsample again
            # 14x14 → 7x7
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(

            # Flatten [16, 7, 7] → [784]
            nn.Flatten(),

            # Combine extracted features
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            # Final class scores (logits)
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # x: [batch, 1, 28, 28]

        x = self.conv_block(x)
        # x: [batch, 16, 7, 7]

        x = self.classifier(x)
        # x: [batch, 10]

        return x
