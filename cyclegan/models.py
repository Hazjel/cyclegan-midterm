import torch
from torch import nn

from .config import ArchitectureConfig


def _validate_images(images: torch.Tensor) -> None:
    if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 3:
        raise ValueError("Expected a nonempty batch [N, 3, H, W]")
    if images.dtype != torch.float32:
        raise TypeError("Expected torch.float32 images normalized to [-1, 1]")
    if any(size < 32 or size % 4 for size in images.shape[2:]):
        raise ValueError("H and W must be >= 32 and divisible by 4; use 256 for this dataset")
    if not torch.isfinite(images).all().item():
        raise ValueError("Images must contain only finite values")
    if images.min().item() < -1 or images.max().item() > 1:
        raise ValueError("Images must be normalized to [-1, 1]")


def _initialize_weights(module: nn.Module, std: float) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, norm_eps: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=True),
            nn.InstanceNorm2d(channels, eps=norm_eps, affine=False, track_running_stats=False),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=True),
            nn.InstanceNorm2d(channels, eps=norm_eps, affine=False, track_running_stats=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.layers(features)


class Generator(nn.Module):
    def __init__(self, config: ArchitectureConfig = ArchitectureConfig()):
        super().__init__()
        self.config = config
        width = config.generator_filters
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(config.channels, width, 7, bias=True),
            nn.InstanceNorm2d(width, eps=config.norm_eps, affine=False, track_running_stats=False),
            nn.ReLU(),
        ]
        for _ in range(2):
            layers.extend([
                nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(width * 2, eps=config.norm_eps, affine=False, track_running_stats=False),
                nn.ReLU(),
            ])
            width *= 2
        layers.extend(ResidualBlock(width, config.norm_eps) for _ in range(config.residual_blocks))
        for _ in range(2):
            layers.extend([
                nn.ConvTranspose2d(width, width // 2, 3, stride=2, padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(width // 2, eps=config.norm_eps, affine=False, track_running_stats=False),
                nn.ReLU(),
            ])
            width //= 2
        layers.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(width, config.channels, 7, bias=True),
            nn.Tanh(),
        ])
        self.layers = nn.Sequential(*layers)
        self.apply(lambda module: _initialize_weights(module, config.init_std))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _validate_images(images)
        return self.layers(images)


class PatchDiscriminator(nn.Module):
    def __init__(self, config: ArchitectureConfig = ArchitectureConfig()):
        super().__init__()
        self.config = config
        width = config.discriminator_filters
        layers = [nn.Conv2d(config.channels, width, 4, stride=2, padding=1, bias=True),
                  nn.LeakyReLU(0.2)]
        for stride in (2, 2, 1):
            layers.extend([
                nn.Conv2d(width, width * 2, 4, stride=stride, padding=1, bias=True),
                nn.InstanceNorm2d(width * 2, eps=config.norm_eps, affine=False, track_running_stats=False),
                nn.LeakyReLU(0.2),
            ])
            width *= 2
        layers.append(nn.Conv2d(width, 1, 4, stride=1, padding=1, bias=True))
        self.layers = nn.Sequential(*layers)
        self.apply(lambda module: _initialize_weights(module, config.init_std))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _validate_images(images)
        return self.layers(images)
