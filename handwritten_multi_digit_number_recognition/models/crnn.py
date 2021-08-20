import torch
import torch.nn as nn


class CRNN(nn.Module):
    """CRNN model proposed in Shi, Bai & Yao (2015).

    Reference:
    https://arxiv.org/pdf/1507.05717.pdf
    """

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(512, hidden_size=256, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(256, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 1, H_, W_). Input image.

        Returns:
            (B, num_classes, S). Logits.
        """
        x = self.cnn(x)  # (B, C, H, W)
        B, C, H, W = x.size()
        S = H * W
        x = x.view(B, C, S)  # (B, C, S)
        x = x.permute(2, 0, 1)  # (S, B, C)
        x, _ = self.lstm(x)  # (S, B, 2 * hidden_size)
        x = x.view(S, B, 2, -1).sum(dim=2)  # (S, B, hidden_size)
        x = self.fc(x)  # (S, B, num_classes)
        return x.permute(1, 2, 0)  # (B, num_classes, S)
