"""Optional read-only audit of the original preprocessing ZIP, not a dataset folder.

Never executes notebook cells. Not required for loading or checking the prepared
data/horse2zebra dataset; the original archive must be available to run this audit.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

from PIL import Image


def audit_archive(path: Path) -> dict:
    path = Path(path)
    with path.open("rb") as source:
        archive_hash = hashlib.file_digest(source, "sha256").hexdigest()
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        names = set(members)
        image_names = sorted(n for n in names if PurePosixPath(n).suffix.lower() in
                             {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        groups = {root: [n for n in image_names if n.startswith(root + "/")]
                  for root in ("data", "data_cleaned", "blur_review")}
        counts = {root: dict(Counter(n.split("/")[1] for n in files)) for root, files in groups.items()}
        modes, sizes, non_rgb, errors, different, hash_paths = Counter(), Counter(), [], [], [], defaultdict(list)
        identical = 0
        for name in groups["data_cleaned"]:
            content = archive.read(name)
            hash_paths[hashlib.sha256(content).hexdigest()].append(name)
            try:
                with Image.open(io.BytesIO(content)) as image:
                    image.load()
                    modes[image.mode] += 1
                    sizes[f"{image.width}x{image.height}"] += 1
                    if image.mode != "RGB":
                        non_rgb.append({"path": name, "mode": image.mode})
            except (OSError, ValueError) as error:
                errors.append({"path": name, "error": str(error)})
            original = "data/" + name.removeprefix("data_cleaned/")
            if original in names and archive.read(original) == content:
                identical += 1
            else:
                different.append(name)

        def read_csv(name):
            return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig"))))

        metadata = read_csv("data/metadata.csv")
        originals_by_basename = defaultdict(list)
        for name in groups["data"]:
            originals_by_basename[PurePosixPath(name).name].append(name)
        missing = []
        for row in metadata:
            relative = row["image_path"].replace("\\", "/")
            if "data/" + relative not in names:
                missing.append({**row, "original_candidates": originals_by_basename[PurePosixPath(relative).name]})

        blur_rows = read_csv("blur_report.csv")
        expected_cleaned, expected_review, unreadable = set(), set(), []
        for row in blur_rows:
            relative = row["path"].replace("\\", "/").removeprefix("data/")
            if not row["blur_score"]:
                unreadable.append(row["path"])
            elif float(row["blur_score"]) < 100:
                expected_review.add("blur_review/" + relative)
            else:
                expected_cleaned.add("data_cleaned/" + relative)
        partition_mismatches = sorted(expected_cleaned.symmetric_difference(groups["data_cleaned"]) |
                                      expected_review.symmetric_difference(groups["blur_review"]))
        notebook = json.loads(archive.read("preproc.ipynb"))
        cells = [{"index": i, "cell_type": cell["cell_type"],
                  "execution_count": cell.get("execution_count"),
                  "source": "".join(cell.get("source", []))}
                 for i, cell in enumerate(notebook["cells"])]
        duplicate_images = [paths for paths in hash_paths.values() if len(paths) > 1]
        cross_split_duplicates = [paths for paths in duplicate_images
                                  if any("/train" in p for p in paths) and any("/test" in p for p in paths)]
        return {
            "checked_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
            "archive": path.name,
            "archive_sha256": archive_hash,
            "archive_entries": len(members),
            "zip_crc_error": archive.testzip(),
            "duplicate_zip_members": [n for n, count in Counter(members).items() if count > 1],
            "original_counts": counts["data"],
            "cleaned_counts": counts["data_cleaned"],
            "review_counts": counts["blur_review"],
            "cleaned_sizes": dict(sizes),
            "cleaned_modes": dict(modes),
            "cleaned_non_rgb": non_rgb,
            "cleaned_decode_errors": errors,
            "cleaned_byte_identical_to_original": identical,
            "cleaned_different_or_missing_original": different,
            "cleaned_duplicate_hash_groups": duplicate_images,
            "cleaned_train_test_duplicate_hash_groups": cross_split_duplicates,
            "metadata_rows": len(metadata),
            "metadata_missing_paths": missing,
            "metadata_missing_by_split": dict(Counter(row.get("split") for row in missing)),
            "blur_report_rows": len(blur_rows),
            "blur_threshold_from_notebook_review": 100,
            "blur_unreadable_rows": unreadable,
            "blur_partition_mismatches": partition_mismatches,
            "notebook": {"executed_by_audit": False, "metadata": notebook.get("metadata"), "cells": cells},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?", default=Path("preproc.zip"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.archive.resolve():
        parser.error("Output must not overwrite the input archive")
    report = audit_archive(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audit written to {args.output}; cleaned={sum(report['cleaned_counts'].values())}, "
          f"non-RGB={len(report['cleaned_non_rgb'])}, metadata missing={len(report['metadata_missing_paths'])}")
    print("Audit records findings; exit code 0 does not mean the dataset has no anomalies.")


if __name__ == "__main__":
    main()
