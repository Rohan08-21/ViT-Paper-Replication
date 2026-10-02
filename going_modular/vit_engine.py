"""
Training and testing functions for a PyTorch ViT.

Backwards compatible with the plain call:

    results = vit_engine.train(model, train_dl, test_dl, optimizer, loss_fn,
                               epochs, device)

New behaviour is opt-in by keyword:

    results = vit_engine.train(..., use_amp=True, warmup_epochs=5,
                               grad_clip=1.0, out_dir="runs/vit_small_aug")
"""
import csv
import math
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm


def _make_scheduler(optimizer, total_steps, warmup_steps):
    """Linear warmup then cosine decay, stepped once per optimizer update."""
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(model, dataloader, loss_fn, optimizer, device,
               scheduler=None, grad_clip=None, accum_steps=1):
    """Trains for a single epoch. Returns (train_loss, train_acc), sample-weighted."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        (loss / accum_steps).backward()

        if (batch + 1) % accum_steps == 0:
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * y.size(0)
        correct += (y_pred.argmax(dim=1) == y).sum().item()
        seen += y.size(0)

    # Flush a partial accumulation window so the last batches still contribute.
    if len(dataloader) % accum_steps != 0:
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    return total_loss / seen, correct / seen


def test_step(model, dataloader, loss_fn, device):
    """Evaluates for a single epoch. Returns (test_loss, test_acc), sample-weighted."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            logits = model(X)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += y.size(0)

    return total_loss / seen, correct / seen


def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device,
          warmup_epochs=0, grad_clip=None, accum_steps=1,
          out_dir=None, resume=True):
    """Trains and tests a model in full float32, logging and checkpointing as it goes.

    Args:
        warmup_epochs: linear LR warmup, then cosine decay. 0 disables the schedule.
        grad_clip: max gradient norm, or None to skip clipping.
        accum_steps: gradient accumulation. Effective batch = batch_size * accum_steps.
        out_dir: directory for last.pt / best.pt / log.csv. None disables all of it.
        resume: pick up from out_dir/last.pt if it exists.

    Returns:
        dict with train_loss, train_acc, test_loss, test_acc lists.
    """
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    scheduler = None
    if warmup_epochs > 0:
        steps_per_epoch = max(1, len(train_dataloader) // accum_steps)
        scheduler = _make_scheduler(optimizer,
                                    total_steps=epochs * steps_per_epoch,
                                    warmup_steps=warmup_epochs * steps_per_epoch)

    start_epoch, best_acc = 0, 0.0
    ckpt_path = log_path = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path, log_path = out_dir / "last.pt", out_dir / "log.csv"

        if resume and ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            if scheduler is not None and ck.get("scheduler"):
                scheduler.load_state_dict(ck["scheduler"])
            start_epoch, best_acc = ck["epoch"] + 1, ck["best_acc"]
            results = ck.get("results", results)
            print(f"[INFO] Resumed from epoch {start_epoch}")

        if not log_path.exists():
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr", "secs"])

    for epoch in tqdm(range(start_epoch, epochs)):
        t0 = time.time()
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer,
                                           device, scheduler, grad_clip, accum_steps)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        secs = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f} | "
            f"lr: {lr_now:.2e} | {secs:.0f}s"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if out_dir is not None:
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch + 1, train_loss, train_acc, test_loss, test_acc, lr_now, round(secs, 1)])

            is_best = test_acc >= best_acc
            best_acc = max(best_acc, test_acc)
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "epoch": epoch, "best_acc": best_acc,
                        "results": results}, ckpt_path)
            if is_best:
                torch.save(model.state_dict(), out_dir / "best.pt")

    return results


def overfit_batch(model, dataloader, loss_fn, optimizer, device, steps=200, log_every=20):
    """Sanity check: drive the loss to ~0 on one fixed batch.

    A correctly wired model has more than enough capacity to memorise a single
    batch. If the loss does NOT fall here, the problem is in the model or the
    training loop, not the dataset size.
    """
    X, y = next(iter(dataloader))
    X, y = X.to(device), y.to(device)
    model.train()

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == steps - 1:
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            print(f"step {step:4d} | loss: {loss.item():.4f} | acc: {acc:.4f}")

    return loss.item()


def inspect_model(model, input_size=(2, 3, 224, 224), device="cpu"):
    """Prints parameter count, embedding stds, and checks the forward pass shape."""
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.1f}M")

    for name in ("class_embedding", "positional_embedding"):
        if hasattr(model, name):
            t = getattr(model, name)
            print(f"{name:22s} shape {tuple(t.shape)}  std {t.std().item():.4f}")

    x = torch.randn(*input_size, device=device)
    model.eval()
    with torch.inference_mode():
        out = model(x)
    print(f"forward: {tuple(x.shape)} -> {tuple(out.shape)}")
    return out.shape
