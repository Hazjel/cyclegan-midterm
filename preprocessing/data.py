"""RGB image loading for the project's unpaired CycleGAN input contract."""

import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, root, split):
        if split not in ("trainA", "trainB", "testA", "testB"):
            raise ValueError("split must be trainA, trainB, testA or testB")
        root = Path(root).resolve()
        self.paths = []
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["version"] != 1:
                raise ValueError("Unsupported manifest version")
            for row in manifest["samples"]:
                path = (root / row["path"]).resolve()
                if row["split"] not in ("trainA", "trainB", "testA", "testB") or path.parent != root / row["split"]:
                    raise ValueError("Manifest path escapes its dataset split")
                if row["split"] == split:
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    self.paths.append(path)
        else:
            # The supplied cleaned dataset is organized into split folders but
            # has no manifest. Discover files in-place without modifying it.
            split_dir = root / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Required dataset split folder not found: {split_dir}")
            self.paths = sorted(p.resolve() for p in split_dir.iterdir()
                                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        if not self.paths:
            raise ValueError(f"Empty domain: {split}")
        self.training = split.startswith("train")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index], formats=["JPEG", "PNG", "BMP", "WEBP"]) as source:
            image = source.convert("RGB")
        size = 286 if self.training else 256
        image = image.resize((size, size), Image.Resampling.BICUBIC)
        if self.training:
            left, top = torch.randint(0, 31, (2,)).tolist()
            image = image.crop((left, top, left + 256, top + 256))
            if torch.rand(()).item() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        return tensor.reshape(256, 256, 3).permute(2, 0, 1).to(torch.float32).div_(255).mul_(2).sub_(1).contiguous()


class UnpairedDataset(Dataset):
    def __init__(self, root):
        self.A = ImageDataset(root, "trainA")
        self.B = ImageDataset(root, "trainB")

    def __len__(self):
        return max(len(self.A), len(self.B))

    def __getitem__(self, index):
        # Visit the larger domain once; resample the shorter without a fixed repeated prefix.
        a = index if len(self.A) >= len(self.B) else torch.randint(len(self.A), ()).item()
        b = index if len(self.B) > len(self.A) else torch.randint(len(self.B), ()).item()
        return {"A": self.A[a], "B": self.B[b]}


def create_dataloaders(root, batch_size=1, num_workers=0, seed=7):
    datasets = {"train": UnpairedDataset(root),
                "testA": ImageDataset(root, "testA"), "testB": ImageDataset(root, "testB")}
    return {name: DataLoader(dataset, batch_size=batch_size, shuffle=name == "train",
                            num_workers=num_workers, drop_last=False,
                            generator=torch.Generator().manual_seed(seed))
            for name, dataset in datasets.items()}
