import io
import unittest
from dataclasses import asdict

import torch

from cyclegan import ArchitectureConfig, Generator, PatchDiscriminator


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(7)

    def test_default_models_256_forward_backward(self):
        config = ArchitectureConfig()
        for name, model in (
            ("G_AB", Generator(config)),
            ("G_BA", Generator(config)),
            ("D_A", PatchDiscriminator(config)),
            ("D_B", PatchDiscriminator(config)),
        ):
            with self.subTest(model=name):
                x = torch.empty(1, 3, 256, 256).uniform_(-1, 1).requires_grad_()
                y = model(x)
                expected = x.shape if name.startswith("G") else (1, 1, 30, 30)
                self.assertEqual(tuple(y.shape), tuple(expected))
                self.assertEqual(y.dtype, torch.float32)
                self.assertTrue(torch.isfinite(y).all().item())
                if name.startswith("G"):
                    self.assertGreaterEqual(y.min().item(), -1)
                    self.assertLessEqual(y.max().item(), 1)
                # Scalar probe only; this is not a CycleGAN training objective.
                y.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all().item())
                self.assertGreater(x.grad.abs().sum().item(), 0)
                for parameter_name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, parameter_name)
                    self.assertTrue(torch.isfinite(p.grad).all().item(), parameter_name)
                self.assertTrue(any(p.grad.abs().sum().item() > 0 for p in model.parameters()))
                print(f"PASS {name}: {tuple(y.shape)}, range=[{y.min().item():.6f}, "
                      f"{y.max().item():.6f}], finite outputs/input+parameter gradients; "
                      f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
                model.zero_grad(set_to_none=True)

    def test_instances_have_independent_parameters_and_storage(self):
        config = ArchitectureConfig(generator_filters=8, discriminator_filters=8, residual_blocks=1)
        models = [Generator(config), Generator(config), PatchDiscriminator(config), PatchDiscriminator(config)]
        ids, storage = set(), set()
        for model in models:
            for p in model.parameters():
                self.assertNotIn(id(p), ids)
                self.assertNotIn(p.untyped_storage().data_ptr(), storage)
                ids.add(id(p))
                storage.add(p.untyped_storage().data_ptr())
        for first, second in ((models[0], models[1]), (models[2], models[3])):
            other = next(second.parameters())
            before = other.detach().clone()
            with torch.no_grad():
                next(first.parameters()).add_(1)
            self.assertTrue(torch.equal(other, before))

    def test_rectangular_batch_and_eval(self):
        config = ArchitectureConfig(generator_filters=8, discriminator_filters=8, residual_blocks=1)
        x = torch.zeros(2, 3, 32, 40)
        for model, shape in ((Generator(config), (2, 3, 32, 40)),
                             (PatchDiscriminator(config), (2, 1, 2, 3))):
            model.eval()
            with torch.no_grad():
                output = model(x)
            self.assertEqual(tuple(output.shape), shape)
            self.assertTrue(torch.isfinite(output).all().item())

    def test_input_contract_rejects_invalid_images(self):
        config = ArchitectureConfig(generator_filters=8, discriminator_filters=8, residual_blocks=1)
        invalid = [torch.zeros(3, 32, 32), torch.zeros(1, 1, 32, 32),
                   torch.zeros(0, 3, 32, 32), torch.zeros(1, 3, 16, 32),
                   torch.zeros(1, 3, 33, 32), torch.zeros(1, 3, 32, 34),
                   torch.zeros(1, 3, 32, 32, dtype=torch.uint8),
                   torch.zeros(1, 3, 32, 32, dtype=torch.float64),
                   torch.full((1, 3, 32, 32), float("nan")),
                   torch.full((1, 3, 32, 32), float("inf")),
                   torch.full((1, 3, 32, 32), 1.01),
                   torch.full((1, 3, 32, 32), -1.01)]
        for model in (Generator(config), PatchDiscriminator(config)):
            for x in invalid:
                with self.subTest(model=type(model).__name__, shape=x.shape, dtype=x.dtype):
                    with self.assertRaises((TypeError, ValueError)):
                        model(x)

    def test_config_round_trip_and_rejects_invalid_values(self):
        config = ArchitectureConfig()
        self.assertEqual(ArchitectureConfig(**asdict(config)), config)
        for kwargs in ({"channels": 1}, {"generator_filters": 0}, {"discriminator_filters": -1},
                       {"residual_blocks": 0}, {"residual_blocks": 1.5},
                       {"generator_filters": True}, {"init_std": 0},
                       {"init_std": float("nan")}, {"norm_eps": float("inf")},
                       {"norm_eps": -1}, {"architecture_version": "unknown"}):
            with self.subTest(config=kwargs), self.assertRaises((TypeError, ValueError)):
                ArchitectureConfig(**kwargs)

    def test_state_dict_round_trip_preserves_outputs(self):
        config = ArchitectureConfig(generator_filters=8, discriminator_filters=8, residual_blocks=1)
        x = torch.empty(1, 3, 32, 32).uniform_(-1, 1)
        for model_type in (Generator, PatchDiscriminator):
            model = model_type(config).eval()
            buffer = io.BytesIO()
            torch.save({"architecture": asdict(config), "model": model.state_dict()}, buffer)
            buffer.seek(0)
            payload = torch.load(buffer, map_location="cpu", weights_only=True)
            restored = model_type(ArchitectureConfig(**payload["architecture"])).eval()
            restored.load_state_dict(payload["model"], strict=True)
            with torch.no_grad():
                torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)

    def test_weight_initialization_and_raw_patch_scores(self):
        config = ArchitectureConfig(generator_filters=8, discriminator_filters=8, residual_blocks=1)
        for model in (Generator(config), PatchDiscriminator(config)):
            weights = torch.cat([m.weight.detach().flatten() for m in model.modules()
                                 if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))])
            self.assertLess(abs(weights.mean().item()), 0.001)
            self.assertAlmostEqual(weights.std().item(), config.init_std, delta=0.001)
            for module in model.modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
        discriminator = PatchDiscriminator(config)
        last_conv = [m for m in discriminator.modules() if isinstance(m, torch.nn.Conv2d)][-1]
        with torch.no_grad():
            last_conv.weight.zero_()
            last_conv.bias.fill_(2)
            output = discriminator(torch.zeros(1, 3, 32, 32))
        self.assertTrue(torch.equal(output, torch.full_like(output, 2)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
