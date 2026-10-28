from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ArchitectureConfig:
    architecture_version: str = "resnet_patchgan_v1"
    channels: int = 3
    generator_filters: int = 64
    discriminator_filters: int = 64
    residual_blocks: int = 9
    norm_eps: float = 1e-5
    init_std: float = 0.02

    def __post_init__(self):
        if self.architecture_version != "resnet_patchgan_v1":
            raise ValueError("Unsupported architecture_version; use resnet_patchgan_v1")
        for name in ("channels", "generator_filters", "discriminator_filters", "residual_blocks"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.channels != 3:
            raise ValueError("The integration contract requires 3 RGB channels")
        for name in ("norm_eps", "init_std"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
