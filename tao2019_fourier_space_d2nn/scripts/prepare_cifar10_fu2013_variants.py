from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from tao2019_fd2nn.data.gt_variants import apply_gt_variant

_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_VARIANTS = ("raw", "binary", "sharpened")


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def _iter_images(image_dir: Path) -> list[Path]:
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in _EXTS])


def _resolve_mask(mask_dir: Path, image_path: Path) -> Path:
    mask_path = mask_dir / image_path.name
    if mask_path.exists():
        return mask_path
    for ext in _EXTS:
        candidate = mask_dir / f"{image_path.stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing mask for {image_path.name}")


def _save_mask(mask: np.ndarray, out_path: Path) -> None:
    Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L").save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic GT variants for CIFAR Fu2013 masks.")
    parser.add_argument("--base-root", default="data/cifar10_cosaliency_fu2013_g5", help="Base dataset root")
    parser.add_argument(
        "--out-root",
        default="data/cifar10_cosaliency_fu2013_variants",
        help="Output root containing one subdirectory per variant",
    )
    args = parser.parse_args()

    base_root = Path(args.base_root)
    out_root = Path(args.out_root)
    manifest: dict[str, object] = {"base_root": str(base_root), "variants": {}}

    for variant in _VARIANTS:
        variant_root = out_root / variant
        variant_manifest: dict[str, object] = {}
        for split in ("train", "val"):
            src_image_dir = base_root / split / "images"
            src_mask_dir = base_root / split / "masks"
            dst_image_dir = variant_root / split / "images"
            dst_mask_dir = variant_root / split / "masks"
            dst_image_dir.mkdir(parents=True, exist_ok=True)
            dst_mask_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            for image_path in _iter_images(src_image_dir):
                mask_path = _resolve_mask(src_mask_dir, image_path)
                shutil.copy2(image_path, dst_image_dir / image_path.name)
                variant_mask = apply_gt_variant(_load_gray(mask_path), variant)
                _save_mask(variant_mask, dst_mask_dir / f"{image_path.stem}.png")
                count += 1
            variant_manifest[split] = {"count": count}
        manifest["variants"][variant] = variant_manifest

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
