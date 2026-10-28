"""Check prepared files, complete loader epochs and real batches against all four models."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

from PIL import Image, __version__ as pillow_version
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cyclegan import ArchitectureConfig, Generator, PatchDiscriminator, create_dataloaders


def check_data(root, output):
    if not __debug__:
        raise RuntimeError("Run this assert-based check without Python -O")
    root = Path(root)
    preparation = json.loads((root / "preparation_report.json").read_text(encoding="utf-8"))
    manifest_bytes = (root / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == preparation["manifest_sha256"]
    manifest = json.loads(manifest_bytes)
    modes, sizes, hashes = Counter(), Counter(), set()
    for sample in manifest["samples"]:
        path = root / sample["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sample["sha256"], path
        with Image.open(path) as image:
            image.load()
            assert image.mode == "RGB" and image.size == (256, 256), path
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            assert pixel_hash == sample["pixel_sha256"] and pixel_hash not in hashes, path
            hashes.add(pixel_hash)
            modes[image.mode] += 1
            sizes[str(image.size)] += 1
    torch.set_num_threads(4)
    torch.manual_seed(7)
    loaders = create_dataloaders(root, batch_size=4, seed=7)
    epochs = {}
    for name, loader in loaders.items():
        count, minimum, maximum = 0, 1., -1.
        for batch in loader:
            tensors = batch.values() if isinstance(batch, dict) else (batch,)
            if isinstance(batch, dict):
                assert set(batch) == {"A", "B"} and batch["A"].shape == batch["B"].shape
            for tensor in tensors:
                assert tensor.shape[1:] == (3, 256, 256) and tensor.dtype == torch.float32
                assert tensor.device.type == "cpu" and torch.isfinite(tensor).all()
                minimum, maximum = min(minimum, tensor.min().item()), max(maximum, tensor.max().item())
                assert minimum >= -1 and maximum <= 1
            count += (batch["A"] if isinstance(batch, dict) else batch).shape[0]
        assert count == len(loader.dataset)
        epochs[name] = {"samples": count, "batches": len(loader), "range": [minimum, maximum]}
    torch.manual_seed(7)
    batch = next(iter(create_dataloaders(root, batch_size=1)["train"]))
    outputs = {}
    config = ArchitectureConfig()
    for name, model_type, domain in (("G_AB", Generator, "A"), ("G_BA", Generator, "B"),
                                     ("D_A", PatchDiscriminator, "A"), ("D_B", PatchDiscriminator, "B")):
        model = model_type(config).eval()
        with torch.no_grad():
            prediction = model(batch[domain])
        expected = (1, 3, 256, 256) if name.startswith("G") else (1, 1, 30, 30)
        assert tuple(prediction.shape) == expected and torch.isfinite(prediction).all()
        outputs[name] = {"input": list(batch[domain].shape), "output": list(prediction.shape),
                         "finite": True, "range": [prediction.min().item(), prediction.max().item()]}
        del model
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "device": "cpu", "seed": 7,
              "torch": torch.__version__, "python": platform.python_version(), "pillow": pillow_version,
              "manifest_sha256": preparation["manifest_sha256"],
              "counts": preparation["counts"], "modes": dict(modes), "sizes": dict(sizes),
              "unique_pixel_hashes": len(hashes), "epochs": epochs, "models": outputs,
              "training_performed": False}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/horse2zebra"))
    parser.add_argument("--output", type=Path, default=Path("docs/bukti/pemeriksaan_data.json"))
    args = parser.parse_args()
    check_data(args.root, args.output)


if __name__ == "__main__":
    main()
