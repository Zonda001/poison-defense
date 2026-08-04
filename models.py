"""
Protected Model — основна модель, яку ми захищаємо від отруєння.

Це невеликий ResNet для CIFAR-розмірів.
Її задача — звичайна класифікація.
Захист реалізований у train.py через зважування loss за trust_weights від Detector'а.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Стандартний ResNet block."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ProtectedModel(nn.Module):
    """
    Малий ResNet (~ResNet-14) для CIFAR.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def weighted_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """
    Cross-entropy з вагою на кожен зразок.
    Це ключове місце "імунного захисту":
        weights близько 1 → нормальне навчання
        weights близько 0 → зразок майже не впливає
    """
    per_sample_loss = F.cross_entropy(logits, targets, reduction="none")
    # Нормалізуємо за сумою ваг, щоб масштаб loss'у був стабільним
    weight_sum = weights.sum().clamp(min=1e-6)
    return (per_sample_loss * weights).sum() / weight_sum


if __name__ == "__main__":
    model = ProtectedModel(num_classes=10)
    x = torch.rand(4, 3, 32, 32)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Тест weighted loss
    targets = torch.randint(0, 10, (4,))
    weights = torch.tensor([1.0, 0.1, 1.0, 0.0])  # 2-й трохи підозрілий, 4-й — точно отрута
    loss = weighted_cross_entropy(out, targets, weights)
    print(f"Weighted loss: {loss.item():.4f}")
    