"""Train and evaluate the repository's unpaired horse↔zebra CycleGAN."""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

# Support explicit invocation from any working directory while retaining the
# required local-package import (no installed cyclegan dependency).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from cyclegan import ArchitectureConfig, Generator, PatchDiscriminator, create_dataloaders


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=REPO_ROOT.parent / "data" / "horse2zebra")
    p.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-decay-start", type=int, default=None,
                   help="first epoch in the linear decay interval")
    p.add_argument("--lr-decay-end", type=int, default=None,
                   help="epoch at which linear-decay LR reaches zero")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or torch device string")
    p.add_argument("--resume", type=Path, help="checkpoint file to resume")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lambda-cycle", type=float, default=10.0)
    p.add_argument("--lambda-identity", type=float, default=5.0)
    p.add_argument("--identity", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--limit-train-batches", type=int, default=None,
                   help="explicitly opt-in debugging cap; not a full epoch")
    p.add_argument("--log-every-batches", type=int, default=100,
                   help="print training progress every N batches; 0 disables batch progress")
    p.add_argument("--skip-test", action="store_true",
                   help="skip testA/testB evaluation for hyperparameter-tuning runs")
    p.add_argument("--validation-preview-every", type=int, default=5,
                   help="save fixed train-validation translations every N epochs (0 disables previews)")
    return p


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta tetapi PyTorch tidak mendeteksi CUDA. Gunakan --device cpu atau periksa instalasi CUDA.")
    return device


def make_experiment_dir(root):
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / f"cyclegan-{stamp}-{os.getpid()}"
    path.mkdir(exist_ok=False)
    return path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_revision():
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def split_train(loaders, fraction, seed, workers):
    """Deterministic per-domain holdout from training samples only.

    The project's unpaired dataset resamples the shorter domain internally;
    validation uses stable indices of that same dataset wrapper. Consequently
    it is a reproducible monitoring subset, not a paired validation set.
    """
    if not 0 < fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    class Pairs(Dataset):
        def __init__(self, a, b, local_seed): self.a, self.b, self.local_seed = a, b, local_seed
        def __len__(self): return max(len(self.a), len(self.b))
        def pair_indices(self, index):
            gen = torch.Generator().manual_seed(self.local_seed + index)
            ai = index if len(self.a) >= len(self.b) else int(torch.randint(len(self.a), (), generator=gen))
            bi = index if len(self.b) > len(self.a) else int(torch.randint(len(self.b), (), generator=gen))
            return ai, bi
        def __getitem__(self, index):
            ai, bi = self.pair_indices(index)
            return {"A": self.a[ai], "B": self.b[bi]}
        def sample_paths(self, index):
            ai, bi = self.pair_indices(index)
            return self.a.paths[ai], self.b.paths[bi]

    base = loaders["train"].dataset
    split = []
    for domain in (base.A, base.B):
        nval = max(1, round(len(domain) * fraction))
        order = torch.randperm(len(domain), generator=torch.Generator().manual_seed(seed + len(domain))).tolist()
        train_domain = Subset(domain, order[nval:])
        # Use a deterministic resize for the fixed validation holdout instead
        # of train's random crop/flip on every validation epoch.
        val_domain = copy.copy(domain)
        val_domain.paths = [domain.paths[i] for i in order[:nval]]
        val_domain.training = False
        split.append((train_domain, val_domain))
    train_a, val_a = split[0]; train_b, val_b = split[1]
    return (
        DataLoader(Pairs(train_a, train_b, seed), batch_size=loaders["train"].batch_size,
                   shuffle=True, num_workers=workers, drop_last=False,
                   generator=torch.Generator().manual_seed(seed)),
        DataLoader(Pairs(val_a, val_b, seed + 1), batch_size=loaders["train"].batch_size,
                   shuffle=False, num_workers=workers, drop_last=False,
                   generator=torch.Generator().manual_seed(seed + 1)),
    )


def freeze(module, value):
    for parameter in module.parameters():
        parameter.requires_grad_(not value)


def train_batch(batch, models, opts, mse, l1, device, lc, li, use_identity):
    gab, gba, da, db = models
    real_a, real_b = batch["A"].to(device), batch["B"].to(device)

    # Generator update: preserve all paths through both generators and frozen Ds.
    freeze(da, True); freeze(db, True)
    opts["G"].zero_grad(set_to_none=True)
    fake_b, fake_a = gab(real_a), gba(real_b)
    rec_a, rec_b = gba(fake_b), gab(fake_a)
    pred_fake_b, pred_fake_a = db(fake_b), da(fake_a)
    adv_ab = mse(pred_fake_b, torch.ones_like(pred_fake_b))
    adv_ba = mse(pred_fake_a, torch.ones_like(pred_fake_a))
    adv_g = adv_ab + adv_ba
    cycle = l1(rec_a, real_a) + l1(rec_b, real_b)
    identity = real_a.new_zeros(())
    if use_identity:
        identity = l1(gba(real_a), real_a) + l1(gab(real_b), real_b)
    total_g = adv_g + lc * cycle + li * identity
    total_g.backward()
    opts["G"].step()
    freeze(da, False); freeze(db, False)

    # Discriminator update uses detached fakes and fresh predictions.
    opts["D"].zero_grad(set_to_none=True)
    pred_real_a, pred_fake_a = da(real_a), da(fake_a.detach())
    pred_real_b, pred_fake_b = db(real_b), db(fake_b.detach())
    loss_da = 0.5 * (mse(pred_real_a, torch.ones_like(pred_real_a)) +
                     mse(pred_fake_a, torch.zeros_like(pred_fake_a)))
    loss_db = 0.5 * (mse(pred_real_b, torch.ones_like(pred_real_b)) +
                     mse(pred_fake_b, torch.zeros_like(pred_fake_b)))
    total_d = loss_da + loss_db
    total_d.backward()
    opts["D"].step()
    return {"G_total": total_g, "G_adversarial": adv_g, "cycle": cycle,
            "identity": identity, "D_A": loss_da, "D_B": loss_db}


@torch.no_grad()
def validate(loader, models, mse, l1, device, lc, li, use_identity):
    gab, gba, da, db = models
    totals = {k: 0.0 for k in ("G_total", "G_adversarial", "cycle", "identity", "D_A", "D_B")}
    count = 0
    for batch in loader:
        a, b = batch["A"].to(device), batch["B"].to(device)
        fb, fa = gab(a), gba(b)
        ra, rb = gba(fb), gab(fa)
        pa, pb = da(fa), db(fb)
        adv = mse(pb, torch.ones_like(pb)) + mse(pa, torch.ones_like(pa))
        cyc = l1(ra, a) + l1(rb, b)
        ident = l1(gba(a), a) + l1(gab(b), b) if use_identity else a.new_zeros(())
        da_r, da_f = da(a), da(fa)
        db_r, db_f = db(b), db(fb)
        dla = .5 * (mse(da_r, torch.ones_like(da_r)) + mse(da_f, torch.zeros_like(da_f)))
        dlb = .5 * (mse(db_r, torch.ones_like(db_r)) + mse(db_f, torch.zeros_like(db_f)))
        vals = (adv + lc * cyc + li * ident, adv, cyc, ident, dla, dlb)
        bs = a.shape[0]
        for key, val in zip(totals, vals): totals[key] += val.item() * bs
        count += bs
    return {k: v / max(count, 1) for k, v in totals.items()}


@torch.no_grad()
def test_and_samples(loaders, models, device, out):
    gab, gba, _, _ = models
    gab.eval(); gba.eval()
    result = {}
    for name, model in (("testA", gab), ("testB", gba)):
        sample_real, sample_fake, sums, n = [], [], [], 0
        for x in loaders[name]:
            x = x.to(device)
            y = model(x)
            reconstructed = (gba if name == "testA" else gab)(y)
            sums.append((reconstructed.sub(x).abs().mean().item(), x.shape[0]))
            remaining = 4 - sum(t.shape[0] for t in sample_real)
            if remaining > 0:
                sample_real.append(x[:remaining].cpu())
                sample_fake.append(y[:remaining].cpu())
            n += x.shape[0]
        result[name] = {"cycle_reconstruction_mae": sum(v*c for v,c in sums)/n, "images": n}
        if sample_real:
            # Also save reconstructions using the opposite generator.
            real, fake = torch.cat(sample_real), torch.cat(sample_fake)
            reverse = gba if name == "testA" else gab
            rec = reverse(fake.to(device)).cpu()
            save_grid(torch.cat([real, fake, rec], dim=0), out / f"{name}_real_translation_reconstruction.png")
    return result


def save_grid(tensor, path):
    from PIL import Image, ImageDraw
    tensor = tensor.clamp(-1, 1).add(1).div(2).mul(255).byte()
    n, _, h, w = tensor.shape
    canvas = Image.new("RGB", (w * n, h + 22), "white")
    draw = ImageDraw.Draw(canvas)
    for i, image in enumerate(tensor):
        rgb = image.permute(1, 2, 0).contiguous().numpy().tobytes()
        canvas.paste(Image.frombytes("RGB", (w, h), rgb), (w*i, 22))
        draw.text((w*i+4, 4), ("real", "translated", "reconstructed")[i//max(1,n//3)] if n >= 3 else str(i), fill="black")
    canvas.save(path)


@torch.no_grad()
def save_validation_previews(samples, models, device, out_dir, epoch):
    """Save fixed train-holdout examples; each generator call uses batch 1."""
    gab, gba, _, _ = models
    a_real, a_fake, a_rec = [], [], []
    b_real, b_fake, b_rec = [], [], []
    for sample in samples:
        if sample["A"].shape != (3, 256, 256) or sample["B"].shape != (3, 256, 256):
            raise ValueError("Fixed validation previews require 3x256x256 project images")
        a = sample["A"].unsqueeze(0).to(device)
        b = sample["B"].unsqueeze(0).to(device)
        fake_b = gab(a)
        fake_a = gba(b)
        a_real.append(a.cpu()); a_fake.append(fake_b.cpu()); a_rec.append(gba(fake_b).cpu())
        b_real.append(b.cpu()); b_fake.append(fake_a.cpu()); b_rec.append(gab(fake_a).cpu())
    out_dir.mkdir(parents=True, exist_ok=True)
    save_grid(torch.cat(a_real + a_fake + a_rec), out_dir / f"epoch-{epoch:04d}_A-to-B.png")
    save_grid(torch.cat(b_real + b_fake + b_rec), out_dir / f"epoch-{epoch:04d}_B-to-A.png")


def rng_state():
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    # torch.load(map_location=cuda) also relocates saved RNG tensors. The
    # default CPU generator specifically requires a CPU ByteTensor.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda"]])


def scheduler_config(args):
    if args.lr_decay_start is None:
        return None
    return {"type": "linear_epoch_decay", "base_lr": args.lr,
            "start_epoch": args.lr_decay_start, "end_epoch": args.lr_decay_end}


def lr_for_epoch(args, epoch):
    if args.lr_decay_start is None or epoch < args.lr_decay_start:
        return args.lr
    span = args.lr_decay_end - args.lr_decay_start
    return args.lr * max(0.0, (args.lr_decay_end - epoch) / span)


def save_checkpoint(path, config, models, opts, epoch, step, best, args, rng, manifest_hash, revision):
    torch.save({"architecture": asdict(config),
                "model_state_dicts": dict(zip(("G_AB", "G_BA", "D_A", "D_B"), [m.state_dict() for m in models])),
                "optimizer_state_dicts": {k: v.state_dict() for k,v in opts.items()},
                "epoch": epoch, "global_step": step, "best_validation": best,
                "training_config": {"loss": {"type":"LSGAN_MSE", "lambda_cycle":args.lambda_cycle,
                    "lambda_identity":args.lambda_identity, "identity":args.identity},
                    "optimizer":{"type":"Adam", "lr":args.lr,"betas":[args.beta1,args.beta2]},
                    "scheduler":scheduler_config(args), "batch_size":args.batch_size,
                    "validation_fraction":args.val_fraction, "num_workers":args.num_workers,
                    "epochs_target":args.epochs},
                "scheduler_state_dict": None if args.lr_decay_start is None else {
                    "last_epoch": epoch, "last_lr": opts["G"].param_groups[0]["lr"]},
                "seed": args.seed, "rng_state":rng,
                "metadata":{"torch":torch.__version__, "python":sys.version,
                    "code_revision":revision, "manifest_sha256":manifest_hash,
                    "dataset_root":str(args.dataset_root.resolve())}}, path)


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or
            args.checkpoint_every < 1 or args.log_every_batches < 0 or
            args.validation_preview_every < 0):
        raise SystemExit("epochs, batch-size and checkpoint-every must be positive; worker/log/preview counts must be nonnegative")
    if args.lr <= 0 or args.lambda_cycle < 0 or args.lambda_identity < 0:
        raise SystemExit("learning rate must be positive and loss weights nonnegative")
    if (args.lr_decay_start is None) != (args.lr_decay_end is None):
        raise SystemExit("provide both --lr-decay-start and --lr-decay-end")
    if args.lr_decay_start is not None:
        if not 1 <= args.lr_decay_start < args.lr_decay_end <= args.epochs:
            raise SystemExit("linear decay requires 1 <= decay-start < decay-end <= epochs")
        if args.lr_decay_end != args.epochs:
            raise SystemExit("for reproducible resume, --lr-decay-end must equal --epochs")
    root = args.dataset_root.resolve()
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"Dataset manifest not found at {root / 'manifest.json'}. Check --dataset-root.")
    manifest_hash = sha256(root / "manifest.json")
    seed_everything(args.seed)
    device = choose_device(args.device)
    loaders = create_dataloaders(root, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)
    train_loader, val_loader = split_train(loaders, args.val_fraction, args.seed, args.num_workers)
    fixed_val_samples = [val_loader.dataset[i] for i in range(min(4, len(val_loader.dataset)))]
    config = ArchitectureConfig()
    models = [Generator(config), Generator(config), PatchDiscriminator(config), PatchDiscriminator(config)]
    models = [m.to(device) for m in models]
    opts = {"G":torch.optim.Adam(list(models[0].parameters())+list(models[1].parameters()), lr=args.lr, betas=(args.beta1,args.beta2)),
            "D":torch.optim.Adam(list(models[2].parameters())+list(models[3].parameters()), lr=args.lr, betas=(args.beta1,args.beta2))}
    start_epoch, step, best = 1, 0, math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["architecture"] != asdict(config):
            raise ValueError("Checkpoint architecture differs from current ArchitectureConfig")
        saved = checkpoint["training_config"]
        requested_scheduler = scheduler_config(args)
        saved_scheduler = saved.get("scheduler")
        # Permit introducing the decay schedule exactly at the checkpoint
        # boundary (the epoch-50 checkpoint has no scheduler state yet).
        starting_schedule_at_boundary = (
            saved_scheduler is None and requested_scheduler is not None and
            checkpoint["epoch"] == args.lr_decay_start - 1
        )
        expected = {"loss": {"type":"LSGAN_MSE", "lambda_cycle":args.lambda_cycle,
                    "lambda_identity":args.lambda_identity, "identity":args.identity},
                    "optimizer":{"type":"Adam", "lr":args.lr,"betas":[args.beta1,args.beta2]},
                    "scheduler":requested_scheduler, "batch_size":args.batch_size,
                    "validation_fraction":args.val_fraction, "num_workers":args.num_workers}
        mismatches = [key for key, value in expected.items()
                      if saved.get(key) != value and not (key == "scheduler" and starting_schedule_at_boundary)]
        if mismatches:
            raise ValueError("Resume training settings differ from checkpoint; repeat its lr, batch size, loss weights, seed, validation fraction and workers.")
        if checkpoint["seed"] != args.seed or checkpoint["metadata"]["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume seed or dataset manifest differs from checkpoint")
        for model, key in zip(models, ("G_AB","G_BA","D_A","D_B")):
            model.load_state_dict(checkpoint["model_state_dicts"][key], strict=True)
        for k, opt in opts.items(): opt.load_state_dict(checkpoint["optimizer_state_dicts"][k])
        start_epoch, step = checkpoint["epoch"] + 1, checkpoint["global_step"]
        best = checkpoint["best_validation"]
        restore_rng(checkpoint["rng_state"])
    if start_epoch > args.epochs:
        raise SystemExit(f"Checkpoint is already at epoch {start_epoch - 1}; set --epochs above that epoch to resume.")
    exp = make_experiment_dir(args.output_root.resolve())
    # A resumed run gets a fresh output directory. Preserve the previous best
    # weights there so final test evaluation remains valid if no later epoch
    # improves the validation score.
    if args.resume:
        prior_best = args.resume.resolve().parent / "best.pt"
        if prior_best.is_file():
            shutil.copy2(prior_best, exp / "best.pt")
        elif checkpoint.get("best_validation") == checkpoint.get("validation_G_total"):
            shutil.copy2(args.resume.resolve(), exp / "best.pt")
    preview_count = min(4, len(val_loader.dataset))
    preview_manifest = []
    for i in range(preview_count):
        a_path, b_path = val_loader.dataset.sample_paths(i)
        preview_manifest.append({"preview_index": i,
                                 "A": str(Path(a_path).relative_to(root)),
                                 "B": str(Path(b_path).relative_to(root))})
    (exp / "validation_samples.json").write_text(
        json.dumps({"source": "train holdout", "batch_size": 1,
                    "image_size": [256, 256], "samples": preview_manifest}, indent=2),
        encoding="utf-8")
    (exp / "config.json").write_text(json.dumps({**vars(args), "dataset_root":str(root), "device":str(device),
        "architecture":asdict(config), "manifest_sha256":manifest_hash, "code_revision":code_revision()},
        default=str, indent=2), encoding="utf-8")
    csv_path = exp / "metrics.csv"
    columns = ["epoch","global_step","G_total","G_adversarial","cycle","identity","D_A","D_B",
               "val_G_total","val_G_adversarial","val_cycle","val_identity","val_D_A","val_D_B","lr","duration_seconds"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        mse, l1 = nn.MSELoss(), nn.L1Loss()
        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, args.epochs + 1):
            last_epoch = epoch
            train_loader.generator.manual_seed(args.seed + epoch)
            current_lr = lr_for_epoch(args, epoch)
            for optimizer in opts.values():
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
            started = time.perf_counter()
            for m in models: m.train()
            sums = {k:0.0 for k in ("G_total","G_adversarial","cycle","identity","D_A","D_B")}; count=0
            for i,batch in enumerate(train_loader):
                if args.limit_train_batches is not None and i >= args.limit_train_batches: break
                losses = train_batch(batch, models, opts, mse, l1, device, args.lambda_cycle, args.lambda_identity, args.identity)
                bs=batch["A"].shape[0]; count+=bs; step+=1
                for k,v in losses.items(): sums[k]+=v.item()*bs
                if args.log_every_batches and ((i + 1) % args.log_every_batches == 0 or i + 1 == len(train_loader)):
                    print(f"  epoch {epoch}: batch {i+1}/{len(train_loader)} G={losses['G_total'].item():.4f} D_A={losses['D_A'].item():.4f} D_B={losses['D_B'].item():.4f}", flush=True)
            train_metrics = {k:v/max(count,1) for k,v in sums.items()}
            for m in models: m.eval()
            val = validate(val_loader, models, mse, l1, device, args.lambda_cycle, args.lambda_identity, args.identity)
            if (args.validation_preview_every and
                    (epoch == start_epoch or epoch % args.validation_preview_every == 0 or epoch == args.epochs)):
                save_validation_previews(fixed_val_samples, models, device,
                                         exp / "validation_samples", epoch)
            duration=time.perf_counter()-started
            row={"epoch":epoch,"global_step":step,**train_metrics,**{"val_"+k:v for k,v in val.items()},"lr":opts["G"].param_groups[0]["lr"],"duration_seconds":duration}
            writer.writerow(row); f.flush()
            print("epoch {}/{} step={} G={:.4f} cycle={:.4f} D_A={:.4f} D_B={:.4f} val_G={:.4f} {:.1f}s".format(epoch,args.epochs,step,train_metrics["G_total"],train_metrics["cycle"],train_metrics["D_A"],train_metrics["D_B"],val["G_total"],duration),flush=True)
            if val["G_total"] < best:
                best=val["G_total"]
                save_checkpoint(exp/"best.pt",config,models,opts,epoch,step,best,args,rng_state(),manifest_hash,code_revision())
            if epoch % args.checkpoint_every == 0:
                save_checkpoint(exp/f"epoch-{epoch:04d}.pt",config,models,opts,epoch,step,best,args,rng_state(),manifest_hash,code_revision())
    (exp/"training_complete.json").write_text(json.dumps({"last_epoch":last_epoch,"global_step":step,"best_validation_G_total":best},indent=2),encoding="utf-8")
    if not args.skip_test:
        if not (exp/"best.pt").is_file():
            raise RuntimeError("No best checkpoint is available; resume checkpoint folder must contain best.pt")
        best_ckpt=torch.load(exp/"best.pt",map_location=device,weights_only=False)
        for model,key in zip(models,("G_AB","G_BA","D_A","D_B")): model.load_state_dict(best_ckpt["model_state_dicts"][key],strict=True)
        test_metrics=test_and_samples(loaders,models,device,exp)
        (exp/"test_metrics.json").write_text(json.dumps(test_metrics,indent=2),encoding="utf-8")
    else:
        print("Test evaluation skipped (--skip-test); use only validation metrics to compare tuning runs.", flush=True)
    print(f"Run files: {exp}")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit("CUDA out of memory. Retry with --batch-size 1, close GPU-heavy applications, or use --device cpu. Dataset was not modified.") from exc
