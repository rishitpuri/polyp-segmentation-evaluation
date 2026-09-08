"""Train one (model, seed) configuration on Kvasir-SEG.

Device-agnostic: picks CUDA on the pod, MPS locally, CPU as fallback.
Writes a per-epoch log and the best-val-Dice checkpoint.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PolypDataset, eval_transform, kvasir_records, seed_worker, train_transform
from .losses import DiceBCELoss
from .metrics import region_metrics
from .models import MODELS, build_model, count_params

ROOT = Path(__file__).resolve().parents[2]


def pick_device(explicit: str | None = None):
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int, deterministic: bool = False):
    """Seed the RNGs, and optionally pin non-deterministic GPU kernels.

    Seeding alone does NOT make a CUDA run reproducible: cuDNN benchmarking
    selects kernels based on runtime timings, and several backward kernels
    accumulate with atomics in non-deterministic order. Two runs of an
    identical configuration therefore differ, by an amount that we measure
    and that turns out to be comparable to the gaps between architectures.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: a few ops have no deterministic kernel; we prefer a
        # warning over aborting the run.
        torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    scores = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].numpy()
        logits = model(images)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        for p, g in zip(probs, masks):
            scores.append(region_metrics(p[0], g[0])["dice"])
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=352)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="Force a device (cuda/mps/cpu)")
    ap.add_argument("--tag", default="", help="Suffix for the run name, e.g. a CV fold id")
    ap.add_argument("--deterministic", action="store_true",
                    help="Pin cuDNN kernels for bitwise-reproducible runs")
    ap.add_argument("--data-root", type=Path, default=ROOT / "Kvasir-SEG")
    ap.add_argument("--splits", type=Path, default=ROOT / "polypseg" / "outputs" / "splits.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "polypseg" / "runs")
    args = ap.parse_args()

    set_seed(args.seed, args.deterministic)
    device = pick_device(args.device)
    run_name = f"{args.model}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PolypDataset(
        kvasir_records(args.data_root, args.splits, "train"), train_transform(args.size)
    )
    val_ds = PolypDataset(
        kvasir_records(args.data_root, args.splits, "val"), eval_transform(args.size)
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", drop_last=True, worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model).to(device)
    total, trainable = count_params(model)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"

    print(f"[{run_name}] device={device} params={total/1e6:.2f}M "
          f"train={len(train_ds)} val={len(val_ds)} amp={use_amp}", flush=True)

    log_path = out_dir / "log.csv"
    log_path.write_text("epoch,train_loss,val_dice,lr,seconds\n")
    best_dice, best_epoch = -1.0, -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        val_dice = validate(model, val_loader, device)
        elapsed = time.time() - t0
        train_loss = float(np.mean(losses))
        lr_now = optimizer.param_groups[0]["lr"]
        with log_path.open("a") as f:
            f.write(f"{epoch},{train_loss:.5f},{val_dice:.5f},{lr_now:.6e},{elapsed:.1f}\n")
        print(f"[{run_name}] epoch {epoch:3d}/{args.epochs} loss={train_loss:.4f} "
              f"val_dice={val_dice:.4f} ({elapsed:.0f}s)", flush=True)

        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_dice": val_dice, "args": vars(args) | {"data_root": str(args.data_root),
                                                                    "splits": str(args.splits),
                                                                    "out_dir": str(args.out_dir)}},
                       out_dir / "best.pth")
        elif epoch - best_epoch >= args.patience:
            print(f"[{run_name}] early stop at epoch {epoch} "
                  f"(best {best_dice:.4f} @ {best_epoch})", flush=True)
            break

    (out_dir / "summary.json").write_text(json.dumps({
        "run": run_name, "model": args.model, "seed": args.seed, "tag": args.tag,
        "splits": str(args.splits),
        "best_val_dice": best_dice, "best_epoch": best_epoch,
        "params_total": total, "params_trainable": trainable,
        "device": str(device),
    }, indent=2))
    print(f"[{run_name}] done. best val Dice {best_dice:.4f} @ epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
