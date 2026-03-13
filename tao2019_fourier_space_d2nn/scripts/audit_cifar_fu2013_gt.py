from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tao2019_fd2nn.analysis.gt_audit import compute_mask_metrics, summarize_mask_metrics

_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def _iter_pairs(split_root: Path) -> list[tuple[Path, Path]]:
    image_dir = split_root / "images"
    mask_dir = split_root / "masks"
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in _EXTS])
    pairs: list[tuple[Path, Path]] = []
    for image_path in images:
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            for ext in _EXTS:
                candidate = mask_dir / f"{image_path.stem}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break
        if not mask_path.exists():
            raise FileNotFoundError(f"missing mask for {image_path.name}")
        pairs.append((image_path, mask_path))
    return pairs


def _overlay_tile(image: np.ndarray, mask: np.ndarray, *, tile_size: int = 100) -> Image.Image:
    im = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="L").resize((tile_size, tile_size))
    mk = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L").resize((tile_size, tile_size))
    rgb = np.repeat(np.asarray(im, dtype=np.uint8)[..., None], 3, axis=2)
    alpha = np.asarray(mk, dtype=np.float32) / 255.0
    rgb[..., 0] = np.clip(rgb[..., 0] * (1.0 - 0.35 * alpha) + 255.0 * 0.35 * alpha, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(rgb[..., 1] * (1.0 - 0.65 * alpha), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(rgb[..., 2] * (1.0 - 0.65 * alpha), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _save_overlay_grid(pairs: list[tuple[Path, Path]], out_path: Path, *, max_items: int) -> None:
    sample_pairs = pairs[:max_items]
    if not sample_pairs:
        return
    tile_size = 100
    cols = 4
    rows = int(np.ceil(len(sample_pairs) / cols))
    grid = Image.new("RGB", (cols * tile_size, rows * tile_size), color=(0, 0, 0))
    for idx, (image_path, mask_path) in enumerate(sample_pairs):
        tile = _overlay_tile(_load_gray(image_path), _load_gray(mask_path), tile_size=tile_size)
        x = (idx % cols) * tile_size
        y = (idx // cols) * tile_size
        grid.paste(tile, (x, y))
    grid.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CIFAR Fu2013 pseudo-GT morphology.")
    parser.add_argument("--root", default="data/cifar10_cosaliency_fu2013_g5", help="Dataset root")
    parser.add_argument("--out-dir", default="reports/gt_audit", help="Audit output directory")
    parser.add_argument("--grid-items", type=int, default=16, help="Number of overlay tiles per split")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {"root": str(root)}
    for split in ("train", "val"):
        split_root = root / split
        pairs = _iter_pairs(split_root)
        metrics = [compute_mask_metrics(_load_gray(mask_path)) for _, mask_path in pairs]
        summary[split] = summarize_mask_metrics(metrics)
        _save_overlay_grid(pairs, out_dir / f"{split}_overlay_grid.png", max_items=int(args.grid_items))

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
