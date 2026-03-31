"""
train_mnist.py
==============
MNIST digit-classification demo for both D²NN and Fourier D²NN.

Usage
-----
Train D²NN (default)::

    python examples/train_mnist.py

Train Fourier D²NN::

    python examples/train_mnist.py --model fourier

Common options::

    python examples/train_mnist.py --model d2nn \
        --num_layers 5 --epochs 20 --batch_size 128 --lr 1e-3

All results (loss curves, phase masks, output intensity samples) are
saved to ``results/<model>_mnist/``.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from d2nn.models import D2NN, FourierD2NN
from d2nn.utils import plot_training_curves, plot_layer_phases, plot_output_intensity


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train D²NN or Fourier D²NN on MNIST"
    )
    parser.add_argument(
        "--model",
        choices=["d2nn", "fourier"],
        default="d2nn",
        help="Which model to train (default: d2nn)",
    )
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--field_size", type=int, nargs=2, default=[28, 28])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--data_dir", default="data", help="Directory for MNIST download"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save results (default: results/<model>_mnist)",
    )
    parser.add_argument(
        "--complex_modulation",
        action="store_true",
        help="Enable complex (amplitude + phase) modulation",
    )
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Using device: {device}")

    output_dir = args.output_dir or os.path.join("results", f"{args.model}_mnist")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    tf = transforms.Compose(
        [
            transforms.Resize(tuple(args.field_size)),
            transforms.ToTensor(),
        ]
    )
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tf)
    val_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    field_size = tuple(args.field_size)
    if args.model == "d2nn":
        model = D2NN(
            num_layers=args.num_layers,
            field_size=field_size,
            num_classes=10,
            wavelength=532e-9,
            z=0.1,
            dx=8e-6,
            complex_modulation=args.complex_modulation,
        )
    else:
        model = FourierD2NN(
            num_layers=args.num_layers,
            field_size=field_size,
            num_classes=10,
            complex_modulation=args.complex_modulation,
        )
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  |  Parameters: {num_params:,}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}"
        )

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(output_dir, "training_curves.png"),
    )
    plot_layer_phases(
        model,
        save_path=os.path.join(output_dir, "phase_masks.png"),
    )

    # Show one output-plane sample
    model.eval()
    images, labels = next(iter(val_loader))
    images = images[:1].to(device)
    with torch.no_grad():
        field = model.encode(images)
        for layer in model.layers:
            field = layer(field)
    plot_output_intensity(
        field[0],
        model._detector_masks,
        predicted=model(images).argmax(1).item(),
        true_label=labels[0].item(),
        save_path=os.path.join(output_dir, "output_intensity_sample.png"),
    )

    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
