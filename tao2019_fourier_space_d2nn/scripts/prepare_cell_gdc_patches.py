"""Prepare cell pathology patches for tao2019 saliency configs.

Builds paired PNG folders expected by the training pipeline:
  <out_root>/train/images, <out_root>/train/masks
  <out_root>/val/images,   <out_root>/val/masks
  <out_root>/test_type1/images, <out_root>/test_type1/masks
  <out_root>/test_type2/images, <out_root>/test_type2/masks
  <out_root>/test_type3/images, <out_root>/test_type3/masks

When no mask is provided for a cell type, saliency masks are auto-generated
from the image patch using the configured saliency GT builder.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.filters import gaussian

_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _normalize01(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32)
    y = y - y.min()
    denom = max(float(y.max()), 1e-8)
    return y / denom


def _to_rgb01(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], repeats=3, axis=2)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _to_gray01(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    return _normalize01(arr)


def _ft_saliency(image: np.ndarray, *, smooth_sigma: float = 1.0) -> np.ndarray:
    rgb = _to_rgb01(image)
    lab = rgb2lab(rgb)
    mean_lab = lab.reshape(-1, 3).mean(axis=0)
    sal = np.sqrt(np.sum((lab - mean_lab) ** 2, axis=-1))
    if smooth_sigma > 0:
        sal = gaussian(sal, sigma=float(smooth_sigma), preserve_range=True)
    return _normalize01(sal)


def _spectral_residual_saliency(
    image: np.ndarray,
    *,
    log_smooth_sigma: float = 3.0,
    map_smooth_sigma: float = 2.0,
) -> np.ndarray:
    gray = _to_gray01(image)
    F = np.fft.fft2(gray)
    log_amp = np.log(np.abs(F) + 1e-8)
    phase = np.angle(F)
    avg_log_amp = gaussian(log_amp, sigma=float(log_smooth_sigma), preserve_range=True)
    residual = log_amp - avg_log_amp
    inv = np.fft.ifft2(np.exp(residual + 1j * phase))
    sal = np.abs(inv) ** 2
    if map_smooth_sigma > 0:
        sal = gaussian(sal, sigma=float(map_smooth_sigma), preserve_range=True)
    return _normalize01(sal)


class SaliencyGtBuilder:
    """Standalone saliency map builder for preprocessing."""

    def __init__(self, *, source: str, params: dict[str, float]):
        self.source = str(source)
        self.params = params

    def build(self, *, image: np.ndarray) -> np.ndarray:
        src = self.source
        if src == "ft":
            return _ft_saliency(image, smooth_sigma=float(self.params.get("smooth_sigma", 1.0)))
        if src == "spectral_residual":
            return _spectral_residual_saliency(
                image,
                log_smooth_sigma=float(self.params.get("log_smooth_sigma", 3.0)),
                map_smooth_sigma=float(self.params.get("map_smooth_sigma", 2.0)),
            )
        if src == "intensity":
            return _to_gray01(image)
        raise ValueError(f"unsupported mask source: {src}")


@dataclass(frozen=True)
class PatchRef:
    cell_type: str
    image_path: Path
    mask_path: Path | None
    x: int
    y: int
    patch_size: int


def _iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"input path not found: {path}")
    files = [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in _EXTS]
    if not files:
        raise RuntimeError(f"no image files found under: {path}")
    return files


def _build_mask_map(mask_input: Path | None) -> dict[str, Path]:
    if mask_input is None:
        return {}
    if mask_input.is_file():
        return {mask_input.name: mask_input}
    if not mask_input.exists():
        raise FileNotFoundError(f"mask path not found: {mask_input}")
    files = [p for p in sorted(mask_input.rglob("*")) if p.is_file() and p.suffix.lower() in _EXTS]
    return {p.name: p for p in files}


def _candidate_refs(
    *,
    cell_type: str,
    image_inputs: Iterable[Path],
    mask_map: dict[str, Path],
    patch_size: int,
    stride: int,
) -> list[PatchRef]:
    refs: list[PatchRef] = []
    for image_path in image_inputs:
        with Image.open(image_path) as im:
            width, height = im.size
        if width < patch_size or height < patch_size:
            continue
        mask_path = mask_map.get(image_path.name)
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                refs.append(
                    PatchRef(
                        cell_type=cell_type,
                        image_path=image_path,
                        mask_path=mask_path,
                        x=x,
                        y=y,
                        patch_size=patch_size,
                    )
                )
    if not refs:
        raise RuntimeError(f"no candidate patches for {cell_type}")
    return refs


def _rgb_or_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3]
    return arr.squeeze()


def _to_gray_uint8(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)
    if x.ndim == 3 and x.shape[2] >= 3:
        x = 0.2989 * x[:, :, 0] + 0.5870 * x[:, :, 1] + 0.1140 * x[:, :, 2]
    x = np.clip(x, 0.0, 255.0)
    return x.astype(np.uint8)


def _save_patch_split(
    refs: list[PatchRef],
    *,
    split_name: str,
    out_root: Path,
    gt_builder: SaliencyGtBuilder,
) -> list[dict[str, object]]:
    image_dir = out_root / split_name / "images"
    mask_dir = out_root / split_name / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    refs_by_img: dict[Path, list[PatchRef]] = defaultdict(list)
    refs_by_mask: dict[Path, Image.Image] = {}
    for ref in refs:
        refs_by_img[ref.image_path].append(ref)

    records: list[dict[str, object]] = []
    idx = 0
    for img_path, img_refs in refs_by_img.items():
        with Image.open(img_path) as img_pil:
            img_pil = img_pil.convert("RGB")
            for ref in img_refs:
                box = (ref.x, ref.y, ref.x + ref.patch_size, ref.y + ref.patch_size)
                patch_img = np.asarray(img_pil.crop(box), dtype=np.uint8)
                patch_gray = _to_gray_uint8(patch_img)

                if ref.mask_path is not None:
                    if ref.mask_path not in refs_by_mask:
                        refs_by_mask[ref.mask_path] = Image.open(ref.mask_path).convert("L")
                    mask_crop = np.asarray(refs_by_mask[ref.mask_path].crop(box), dtype=np.uint8)
                    mask01 = np.clip(mask_crop.astype(np.float32) / 255.0, 0.0, 1.0)
                else:
                    sal = gt_builder.build(image=_rgb_or_gray(patch_img))
                    mask01 = np.clip(sal.astype(np.float32), 0.0, 1.0)

                mask_u8 = (mask01 * 255.0 + 0.5).astype(np.uint8)
                file_name = f"{ref.cell_type}_{split_name}_{idx:06d}.png"
                Image.fromarray(patch_gray, mode="L").save(image_dir / file_name)
                Image.fromarray(mask_u8, mode="L").save(mask_dir / file_name)
                records.append(
                    {
                        "file": file_name,
                        "split": split_name,
                        "cell_type": ref.cell_type,
                        "source_image": str(ref.image_path),
                        "source_mask": str(ref.mask_path) if ref.mask_path is not None else None,
                        "x": ref.x,
                        "y": ref.y,
                        "patch_size": ref.patch_size,
                    }
                )
                idx += 1

    for im in refs_by_mask.values():
        im.close()
    return records


def _sample_refs(refs: list[PatchRef], k: int, rng: random.Random) -> list[PatchRef]:
    if k <= 0:
        return []
    if len(refs) < k:
        raise RuntimeError(f"requested {k} patches, but only {len(refs)} candidates are available")
    ids = rng.sample(range(len(refs)), k)
    return [refs[i] for i in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cell GDC patches for saliency training")
    parser.add_argument("--type1-image", required=True, help="Path to Cell Type 1 image file or directory")
    parser.add_argument("--type1-mask", default=None, help="Optional mask file/dir aligned to type1 images")
    parser.add_argument("--type2-image", default=None, help="Optional Cell Type 2 image file or directory")
    parser.add_argument("--type2-mask", default=None, help="Optional mask file/dir aligned to type2 images")
    parser.add_argument("--type3-image", default=None, help="Optional Cell Type 3 image file or directory")
    parser.add_argument("--type3-mask", default=None, help="Optional mask file/dir aligned to type3 images")
    parser.add_argument("--out-root", default="data/cell_gdc", help="Output root directory")
    parser.add_argument("--patch-size", type=int, default=200, help="Patch size in pixels")
    parser.add_argument("--stride", type=int, default=200, help="Patch sampling stride")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split sampling")
    parser.add_argument("--train-count", type=int, default=2750, help="Number of type1 train patches")
    parser.add_argument("--val-count", type=int, default=250, help="Number of type1 validation patches")
    parser.add_argument("--test-type1-count", type=int, default=500, help="Number of type1 test patches")
    parser.add_argument("--test-type2-count", type=int, default=250, help="Number of type2 test patches")
    parser.add_argument("--test-type3-count", type=int, default=250, help="Number of type3 test patches")
    parser.add_argument(
        "--mask-source",
        choices=["dataset", "ft", "spectral_residual", "intensity"],
        default="dataset",
        help="Fallback mask generation mode when mask file is missing",
    )
    parser.add_argument("--ft-sigma", type=float, default=1.0, help="Smooth sigma for FT saliency fallback")
    parser.add_argument("--sr-log-sigma", type=float, default=3.0, help="Log-spectrum smoothing sigma for SR fallback")
    parser.add_argument("--sr-map-sigma", type=float, default=2.0, help="Map smoothing sigma for SR fallback")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    rng = random.Random(int(args.seed))
    out_root.mkdir(parents=True, exist_ok=True)

    gt_source = str(args.mask_source)
    if gt_source == "dataset":
        gt_source = "ft"
    gt_builder = SaliencyGtBuilder(
        source=gt_source,
        params={
            "smooth_sigma": float(args.ft_sigma),
            "log_smooth_sigma": float(args.sr_log_sigma),
            "map_smooth_sigma": float(args.sr_map_sigma),
        },
    )

    type1_images = _iter_images(Path(args.type1_image))
    type1_masks = _build_mask_map(Path(args.type1_mask)) if args.type1_mask is not None else {}
    refs_type1 = _candidate_refs(
        cell_type="type1",
        image_inputs=type1_images,
        mask_map=type1_masks,
        patch_size=int(args.patch_size),
        stride=int(args.stride),
    )

    k_type1_total = int(args.train_count) + int(args.val_count) + int(args.test_type1_count)
    picked_t1 = _sample_refs(refs_type1, k_type1_total, rng)
    rng.shuffle(picked_t1)
    train_refs = picked_t1[: int(args.train_count)]
    val_refs = picked_t1[int(args.train_count) : int(args.train_count) + int(args.val_count)]
    test1_refs = picked_t1[int(args.train_count) + int(args.val_count) :]

    test2_refs: list[PatchRef] = []
    if args.type2_image is not None and int(args.test_type2_count) > 0:
        type2_images = _iter_images(Path(args.type2_image))
        type2_masks = _build_mask_map(Path(args.type2_mask)) if args.type2_mask is not None else {}
        refs_type2 = _candidate_refs(
            cell_type="type2",
            image_inputs=type2_images,
            mask_map=type2_masks,
            patch_size=int(args.patch_size),
            stride=int(args.stride),
        )
        test2_refs = _sample_refs(refs_type2, int(args.test_type2_count), rng)

    test3_refs: list[PatchRef] = []
    if args.type3_image is not None and int(args.test_type3_count) > 0:
        type3_images = _iter_images(Path(args.type3_image))
        type3_masks = _build_mask_map(Path(args.type3_mask)) if args.type3_mask is not None else {}
        refs_type3 = _candidate_refs(
            cell_type="type3",
            image_inputs=type3_images,
            mask_map=type3_masks,
            patch_size=int(args.patch_size),
            stride=int(args.stride),
        )
        test3_refs = _sample_refs(refs_type3, int(args.test_type3_count), rng)

    all_records: list[dict[str, object]] = []
    all_records.extend(_save_patch_split(train_refs, split_name="train", out_root=out_root, gt_builder=gt_builder))
    all_records.extend(_save_patch_split(val_refs, split_name="val", out_root=out_root, gt_builder=gt_builder))
    all_records.extend(_save_patch_split(test1_refs, split_name="test_type1", out_root=out_root, gt_builder=gt_builder))
    if test2_refs:
        all_records.extend(_save_patch_split(test2_refs, split_name="test_type2", out_root=out_root, gt_builder=gt_builder))
    if test3_refs:
        all_records.extend(_save_patch_split(test3_refs, split_name="test_type3", out_root=out_root, gt_builder=gt_builder))

    summary = {
        "out_root": str(out_root),
        "patch_size": int(args.patch_size),
        "stride": int(args.stride),
        "seed": int(args.seed),
        "requested_counts": {
            "train": int(args.train_count),
            "val": int(args.val_count),
            "test_type1": int(args.test_type1_count),
            "test_type2": int(args.test_type2_count),
            "test_type3": int(args.test_type3_count),
        },
        "generated_counts": {
            "train": len(train_refs),
            "val": len(val_refs),
            "test_type1": len(test1_refs),
            "test_type2": len(test2_refs),
            "test_type3": len(test3_refs),
            "total": len(all_records),
        },
        "mask_source_when_missing": str(gt_builder.source),
    }
    (out_root / "metadata.json").write_text(json.dumps({"summary": summary, "records": all_records}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
