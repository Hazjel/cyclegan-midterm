"""Optional preparation from two source ZIPs; not needed to use data/horse2zebra.

Requires the original preprocessing ZIP and Kaggle ZIP. An already prepared
dataset folder is not a replacement for either archive input.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform
import tempfile
import zipfile

from PIL import Image, __version__ as pillow_version


SPLITS = ("trainA", "trainB", "testA", "testB")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_URL = "https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset"
KAGGLE_V1_SHA256 = "0655d8feb8eb83911ac08c1d230a0464794418c107412a28adc47e1eb69e90a0"


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_members(archive, prefix=""):
    members = archive.namelist()
    if len(members) != len(set(members)):
        raise ValueError("Duplicate ZIP members are ambiguous")
    result = {}
    for member in members:
        if not member.startswith(prefix) or PurePosixPath(member).suffix.lower() not in EXTENSIONS:
            continue
        relative = member.removeprefix(prefix)
        parts = relative.split("/")
        if (len(parts) != 2 or parts[0] not in SPLITS or
                any(c in parts[1] for c in '<>:"\\|?*') or
                parts[1].rstrip(" .") != parts[1]):
            raise ValueError(f"Unsafe or unknown image path: {member}")
        result[relative] = member
    return result


def read_csv(archive, name):
    return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig"))))


def prepare_data(preproc, source, output):
    preproc, source, output = Path(preproc), Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory")
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "source_archive_sha256": sha256_file(source),
        "preproc_archive_sha256": sha256_file(preproc),
        "python": platform.python_version(), "pillow": pillow_version,
        "blur_threshold": 100,
        "duplicate_policy": "identical decoded RGB pixels; keep test before train, then lexical path",
        "metadata_corrections": [], "excluded": [], "converted_rgb": [],
    }
    report["source_matches_pinned_kaggle_v1"] = report["source_archive_sha256"] == KAGGLE_V1_SHA256
    with zipfile.ZipFile(preproc) as local, zipfile.ZipFile(source) as kaggle:
        originals = image_members(local, "data/")
        canonical = image_members(kaggle)
        cleaned = image_members(local, "data_cleaned/")
        review = image_members(local, "blur_review/")
        if not canonical or originals.keys() != canonical.keys():
            raise ValueError("Kaggle source image paths differ from the supplied originals")
        for archive in (local, kaggle):
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC check failed")
        by_name = defaultdict(list)
        for path in canonical:
            by_name[PurePosixPath(path).name].append(path)
        metadata = read_csv(local, "data/metadata.csv")
        report["metadata_identical_to_kaggle"] = local.read("data/metadata.csv") == kaggle.read("metadata.csv")
        resolved = {}
        for row in metadata:
            claimed = row["image_path"].replace("\\", "/")
            candidates = [claimed] if claimed in canonical else by_name[PurePosixPath(claimed).name]
            if len(candidates) != 1 or candidates[0] in resolved:
                raise ValueError(f"Ambiguous or duplicate metadata: {claimed}")
            actual = candidates[0]
            if row.get("image_id") != PurePosixPath(actual).stem:
                raise ValueError(f"Metadata image_id conflicts with source path: {claimed}")
            split = actual.split("/")[0]
            domain = "A (Horse)" if split.endswith("A") else "B (Zebra)"
            fixed = {**row, "image_path": actual, "domain": domain, "split": split[:-1]}
            if fixed != row:
                report["metadata_corrections"].append({"before": row, "after": fixed})
            resolved[actual] = fixed
        if resolved.keys() != canonical.keys():
            raise ValueError("Metadata must cover each source image exactly once")
        scores = {}
        for row in read_csv(local, "blur_report.csv"):
            path = row["path"].replace("\\", "/").removeprefix("data/")
            score = float(row["blur_score"])
            if path in scores or path not in canonical or not math.isfinite(score) or score < 0:
                raise ValueError(f"Invalid blur score/path: {path}")
            scores[path] = score
        if scores.keys() != canonical.keys():
            raise ValueError("Blur report must cover each source image exactly once")
        if set(cleaned) != {p for p, s in scores.items() if s >= 100} or set(review) != {p for p, s in scores.items() if s < 100}:
            raise ValueError("Blur partition differs from threshold 100")

        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="prepare-", dir=output.parent) as temporary:
            staging = Path(temporary) / "dataset"
            staging.mkdir()
            samples, seen = [], {}
            for path in sorted(canonical, key=lambda p: (not p.startswith("test"), p)):
                content = kaggle.read(canonical[path])
                partition = cleaned if path in cleaned else review
                if local.read(originals[path]) != content or local.read(partition[path]) != content:
                    raise ValueError(f"Kaggle source bytes differ: {path}")
                split = path.split("/")[0]
                prefix_domain = {"n02381460": "A", "n02391049": "B"}.get(PurePosixPath(path).stem.split("_")[0])
                if prefix_domain is not None and prefix_domain != split[-1]:
                    raise ValueError(f"Filename domain conflicts with source folder: {path}")
                if path in review:
                    report["excluded"].append({"source_path": path, "reason": "blur", "blur_score": scores[path]})
                    continue
                with Image.open(io.BytesIO(content), formats=["JPEG", "PNG", "BMP", "WEBP"]) as original:
                    original.load()
                    if original.size != (256, 256):
                        raise ValueError(f"Expected source image size 256x256: {path}")
                    if original.mode != "RGB":
                        report["converted_rgb"].append({"source_path": path, "original_mode": original.mode})
                    image = original.convert("RGB")
                pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
                if pixel_hash in seen:
                    kept = seen[pixel_hash]
                    if kept["split"][-1] != split[-1]:
                        raise ValueError(f"Identical pixels have conflicting domain: {path}")
                    report["excluded"].append({"source_path": path, "reason": "duplicate", "duplicate_of": kept["source_path"], "pixel_sha256": pixel_hash})
                    continue
                relative = str(PurePosixPath(path).with_suffix(".png"))
                target = staging / relative
                if target.exists():
                    raise ValueError(f"Output filename collision: {relative}")
                target.parent.mkdir(exist_ok=True)
                image.save(target, "PNG")
                sample = {"path": relative, "source_path": path, "split": split,
                          "domain": split[-1], "source_sha256": hashlib.sha256(content).hexdigest(),
                          "pixel_sha256": pixel_hash, "sha256": sha256_file(target), "blur_score": scores[path]}
                samples.append(sample)
                seen[pixel_hash] = sample
            samples.sort(key=lambda row: row["path"])
            report["counts"] = {s: sum(row["split"] == s for row in samples) for s in SPLITS}
            if not all(report["counts"].values()):
                raise ValueError("Each train/test domain must be nonempty after preparation")
            report["source_verified_images"] = len(canonical)
            report["prepared_rgb_images"] = len(samples)
            manifest = {"version": 1, "source_url": SOURCE_URL, "samples": samples}
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            report["manifest_sha256"] = sha256_file(staging / "manifest.json")
            with (staging / "metadata_corrected.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
                writer.writeheader()
                writer.writerows(resolved[p] for p in sorted(resolved))
            (staging / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preproc", type=Path, default=Path("preproc.zip"))
    parser.add_argument("--source", type=Path, default=Path("data/source/horse2zebra-kaggle.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/horse2zebra"))
    args = parser.parse_args()
    if sha256_file(args.source) != KAGGLE_V1_SHA256:
        parser.error("Source ZIP SHA-256 differs from the verified Kaggle version 1 archive")
    report = prepare_data(args.preproc, args.source, args.output)
    print(json.dumps({k: report[k] for k in ("counts", "source_verified_images", "prepared_rgb_images", "manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
