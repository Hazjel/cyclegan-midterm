import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from PIL import Image
import torch

from cyclegan import create_dataloaders as public_create_dataloaders
from preprocessing.prepare_data import prepare_data
from preprocessing.data import ImageDataset, UnpairedDataset, create_dataloaders


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.preproc = self.root / "preproc.zip"
        self.source = self.root / "source.zip"
        self.output = self.root / "ready"
        self.images = {
            "trainA/horse.png": Image.new("RGB", (256, 256), (255, 0, 127)),
            "trainB/zebra.png": Image.new("L", (256, 256), 80),
            "trainB/other.png": Image.new("RGB", (256, 256), (0, 255, 0)),
            "testA/testhorse.png": Image.new("RGB", (256, 256), (20, 30, 40)),
            "testB/testzebra.png": Image.new("RGB", (256, 256), (50, 60, 70)),
            "trainA/leak.png": Image.new("RGB", (256, 256), (20, 30, 40)),
            "trainA/blur.png": Image.new("RGB", (256, 256), (10, 10, 10)),
        }
        self.write_archives()

    def write_archives(self, wrong_metadata=True):
        metadata = io.StringIO()
        writer = csv.writer(metadata)
        writer.writerow(["image_id", "domain", "split", "image_path"])
        blur = io.StringIO()
        scores = csv.writer(blur)
        scores.writerow(["path", "blur_score"])
        with zipfile.ZipFile(self.preproc, "w") as local, zipfile.ZipFile(self.source, "w") as source:
            for name, image in self.images.items():
                buffer = io.BytesIO()
                image.save(buffer, "PNG")
                content = buffer.getvalue()
                source.writestr(name, content)
                local.writestr("data/" + name, content)
                is_blur = "blur" in name
                local.writestr(("blur_review/" if is_blur else "data_cleaned/") + name, content)
                scores.writerow(["data/" + name, 99 if is_blur else 100])
                split, filename = name.split("/", 1)
                claimed = "trainA/zebra.png" if wrong_metadata and name == "trainB/zebra.png" else name
                writer.writerow([Path(filename).stem, "A (Horse)" if claimed.split("/")[0].endswith("A") else "B (Zebra)", split[:-1], claimed])
            local.writestr("data/metadata.csv", metadata.getvalue())
            source.writestr("metadata.csv", metadata.getvalue())
            local.writestr("blur_report.csv", blur.getvalue())

    def prepare(self):
        return prepare_data(self.preproc, self.source, self.output)

    def test_preparation_reconciles_rgb_blur_and_leak_without_changing_sources(self):
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.preproc, self.source)]
        report = self.prepare()
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(report["source_verified_images"], 7)
        self.assertEqual(len(report["metadata_corrections"]), 1)
        self.assertEqual(report["counts"], {"trainA": 1, "trainB": 2, "testA": 1, "testB": 1})
        self.assertEqual(len(manifest["samples"]), 5)
        self.assertTrue(any(r["source_path"] == "trainA/leak.png" and r["reason"] == "duplicate" for r in report["excluded"]))
        self.assertTrue(any(r["reason"] == "blur" for r in report["excluded"]))
        for sample in manifest["samples"]:
            with Image.open(self.output / sample["path"]) as image:
                self.assertEqual((image.mode, image.size), ("RGB", (256, 256)))
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.preproc, self.source)])
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_source_mismatch_fails_without_publishing_partial_output(self):
        with zipfile.ZipFile(self.source, "a") as archive:
            archive.writestr("trainA/unexpected.png", b"wrong source")
        with self.assertRaisesRegex(ValueError, "source|Source|Kaggle"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_source_byte_mismatch_fails(self):
        with zipfile.ZipFile(self.source) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
        contents["trainA/horse.png"] = b"different image bytes"
        with zipfile.ZipFile(self.source, "w") as archive:
            for name, content in contents.items():
                archive.writestr(name, content)
        with self.assertRaisesRegex(ValueError, "source bytes differ"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_cli_rejects_unpinned_source(self):
        script = Path(__file__).resolve().parents[1] / "preprocessing" / "prepare_data.py"
        result = subprocess.run([sys.executable, str(script), "--preproc", str(self.preproc),
                                 "--source", str(self.source), "--output", str(self.output)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Source ZIP SHA-256 differs", result.stderr)
        self.assertFalse(self.output.exists())

    def test_ambiguous_metadata_fails(self):
        self.images["testB/zebra.png"] = Image.new("RGB", (256, 256), (10, 20, 30))
        self.write_archives()
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_domain_conflict_fails(self):
        self.images["trainB/conflict.png"] = self.images["trainA/horse.png"]
        self.write_archives(False)
        with self.assertRaisesRegex(ValueError, "domain"):
            self.prepare()

    def test_normalization_keeps_rgb_channel_order_and_test_is_deterministic(self):
        self.prepare()
        train = ImageDataset(self.output, "trainA")
        pixel = train[0][:, 0, 0]
        torch.testing.assert_close(pixel, torch.tensor([1., -1., 127 / 127.5 - 1]))
        zebra = ImageDataset(self.output, "trainB")
        gray = next(zebra[i] for i, p in enumerate(zebra.paths) if p.stem == "zebra")
        self.assertTrue(torch.equal(gray[0], gray[1]) and torch.equal(gray[1], gray[2]))
        test = ImageDataset(self.output, "testA")
        self.assertTrue(torch.equal(test[0], test[0]))
        self.assertEqual(test[0].shape, (3, 256, 256))
        self.assertEqual(test[0].dtype, torch.float32)
        self.assertTrue(torch.isfinite(test[0]).all())
        with self.assertRaises(ValueError):
            ImageDataset(self.output, "validationA")

    def test_augmentation_varies_on_train_but_not_on_test(self):
        gradient = Image.frombytes("RGB", (256, 256), bytes(range(256)) * (256 * 3))
        self.images["trainA/horse.png"] = gradient
        self.write_archives(False)
        self.prepare()
        dataset = ImageDataset(self.output, "trainA")
        torch.manual_seed(17)
        samples = [dataset[0] for _ in range(6)]
        self.assertTrue(any(not torch.equal(samples[0], x) for x in samples[1:]))
        test = ImageDataset(self.output, "testA")
        self.assertTrue(torch.equal(test[0], test[0]))

    def test_archive_traversal_is_rejected(self):
        self.images["trainA/../escape.png"] = Image.new("RGB", (256, 256))
        self.write_archives(False)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_loaders_keep_eval_domains_separate_and_batch_last_samples(self):
        self.assertIs(public_create_dataloaders, create_dataloaders)
        self.prepare()
        dataset = UnpairedDataset(self.output)
        self.assertEqual(len(dataset), 2)
        loaders = create_dataloaders(self.output, batch_size=3)
        batch = next(iter(loaders["train"]))
        self.assertEqual(set(batch), {"A", "B"})
        for tensor in batch.values():
            self.assertEqual(tensor.shape, (2, 3, 256, 256))
            self.assertGreaterEqual(tensor.min().item(), -1)
            self.assertLessEqual(tensor.max().item(), 1)
        self.assertFalse(torch.equal(batch["B"][0], batch["B"][1]))
        self.assertEqual(sum(x.shape[0] for x in loaders["testA"]), 1)
        self.assertEqual(sum(x.shape[0] for x in loaders["testB"]), 1)
        worker_batch = next(iter(create_dataloaders(self.output, num_workers=1)["train"]))
        self.assertEqual(worker_batch["A"].shape, (1, 3, 256, 256))

    def test_manifest_traversal_empty_split_and_corrupt_image_fail(self):
        self.prepare()
        path = self.output / "manifest.json"
        original = json.loads(path.read_text())
        tampered = json.loads(path.read_text())
        tampered["samples"][0]["path"] = "../outside.png"
        path.write_text(json.dumps(tampered))
        with self.assertRaises(ValueError):
            ImageDataset(self.output, tampered["samples"][0]["split"])
        path.write_text(json.dumps({**original, "samples": []}))
        with self.assertRaises(ValueError):
            ImageDataset(self.output, "trainA")
        path.write_text(json.dumps(original))
        dataset = ImageDataset(self.output, "trainA")
        dataset.paths[0].write_bytes(b"not an image")
        with self.assertRaises(OSError):
            dataset[0]


if __name__ == "__main__":
    unittest.main(verbosity=2)
