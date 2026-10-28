import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from scripts.audit_archive import audit_archive


class ArchiveAuditTests(unittest.TestCase):
    def test_reads_archive_without_modifying_data_and_flags_wrong_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.zip"
            buffer = io.BytesIO()
            Image.new("L", (256, 256), 80).save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("data/trainB/zebra.png", image_bytes)
                archive.writestr("data_cleaned/trainB/zebra.png", image_bytes)
                archive.writestr("data/metadata.csv", "image_path,domain,split\ntrainA/zebra.png,A (Horse),train\n")
                archive.writestr("blur_report.csv", "path,folder,filename,blur_score,status\ndata/trainB/zebra.png,trainB,zebra.png,101,analyzed\n")
                archive.writestr("preproc.ipynb", json.dumps({"cells": [
                    {"cell_type": "code", "source": ["raise RuntimeError('MUST NOT EXECUTE')"], "execution_count": None}
                ], "metadata": {}}))
            before = path.read_bytes()
            report = audit_archive(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(report["cleaned_counts"]["trainB"], 1)
            self.assertEqual(report["cleaned_sizes"], {"256x256": 1})
            self.assertEqual(report["cleaned_modes"], {"L": 1})
            self.assertEqual(report["cleaned_byte_identical_to_original"], 1)
            self.assertEqual(len(report["metadata_missing_paths"]), 1)
            self.assertEqual(report["metadata_missing_paths"][0]["original_candidates"], ["data/trainB/zebra.png"])
            self.assertEqual(report["notebook"]["cells"][0]["source"], "raise RuntimeError('MUST NOT EXECUTE')")
            self.assertEqual(report["cleaned_decode_errors"], [])
            self.assertEqual(report["blur_partition_mismatches"], [])

    def test_corrupt_cleaned_image_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("data_cleaned/trainA/bad.jpg", b"not an image")
                archive.writestr("data/metadata.csv", "image_path\n")
                archive.writestr("blur_report.csv", "path,blur_score\n")
                archive.writestr("preproc.ipynb", '{"cells": [], "metadata": {}}')
            report = audit_archive(path)
            self.assertEqual(len(report["cleaned_decode_errors"]), 1)
            self.assertEqual(report["cleaned_decode_errors"][0]["path"], "data_cleaned/trainA/bad.jpg")

    def test_identical_bytes_across_train_test_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.zip"
            buffer = io.BytesIO()
            Image.new("RGB", (256, 256)).save(buffer, format="PNG")
            with zipfile.ZipFile(path, "w") as archive:
                for name in ("data_cleaned/trainA/one.png", "data_cleaned/testA/two.png"):
                    archive.writestr(name, buffer.getvalue())
                archive.writestr("data/metadata.csv", "image_path\n")
                archive.writestr("blur_report.csv", "path,blur_score\n")
                archive.writestr("preproc.ipynb", '{"cells": [], "metadata": {}}')
            report = audit_archive(path)
            self.assertEqual(report["cleaned_train_test_duplicate_hash_groups"], [
                ["data_cleaned/testA/two.png", "data_cleaned/trainA/one.png"]])


if __name__ == "__main__":
    unittest.main()
