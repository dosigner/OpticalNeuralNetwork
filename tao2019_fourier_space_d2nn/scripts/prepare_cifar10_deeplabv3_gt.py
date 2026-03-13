"""Generate pseudo-GT saliency masks for CIFAR-10 cat/horse using DeepLabV3.

Uses torchvision DeepLabV3-ResNet101 (COCO/VOC pretrained) to produce
pixel-level semantic segmentation masks. CIFAR-10 32x32 images are
upscaled to 256x256 for inference, then masks are downscaled back to 32x32.

Output layout (same as SaliencyPairsDataset):
  data/cifar10_deeplabv3_cat/train/images/*.png   (grayscale)
  data/cifar10_deeplabv3_cat/train/masks/*.png     (binary 0/255)
  data/cifar10_deeplabv3_cat/val/images/*.png
  data/cifar10_deeplabv3_cat/val/masks/*.png

Train = CIFAR-10 train "cat" (5000 images, filtered to fg-detected only)
Val   = CIFAR-10 test  "horse" (1000 images)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models, transforms

VOC_CAT = 8
VOC_HORSE = 13


def _build_segmodel(device: torch.device) -> torch.nn.Module:
    model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
    model.eval()
    model.to(device)
    return model


def _segment_batch(
    model: torch.nn.Module,
    images_rgb: torch.Tensor,
    target_class: int,
    device: torch.device,
    infer_size: int = 256,
) -> np.ndarray:
    B = images_rgb.shape[0]
    imgs = images_rgb.float() / 255.0
    imgs = F.interpolate(imgs, size=(infer_size, infer_size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    imgs = (imgs.to(device) - mean) / std
    with torch.no_grad():
        out = model(imgs)["out"]
        pred = out.argmax(dim=1)
    mask = (pred == target_class).float()
    mask = F.interpolate(mask.unsqueeze(1), size=(32, 32), mode="bilinear", align_corners=False)
    mask = (mask.squeeze(1) > 0.5).cpu().numpy().astype(np.uint8) * 255
    return mask


def _process_class(
    *,
    model: torch.nn.Module,
    cifar_dataset: datasets.CIFAR10,
    cifar_class_idx: int,
    voc_class_id: int,
    class_name: str,
    split_name: str,
    out_root: Path,
    device: torch.device,
    batch_size: int = 64,
    filter_empty: bool = False,
) -> dict:
    all_indices = [i for i, (_, label) in enumerate(cifar_dataset) if label == cifar_class_idx]
    image_dir = out_root / split_name / "images"
    mask_dir = out_root / split_name / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    total = len(all_indices)
    fg_detected = 0
    saved = 0
    skipped = 0

    for start in range(0, total, batch_size):
        batch_idx = all_indices[start : start + batch_size]
        imgs_list = []
        for i in batch_idx:
            img_pil, _ = cifar_dataset[i]
            arr = np.array(img_pil)
            imgs_list.append(torch.from_numpy(arr).permute(2, 0, 1))
        imgs_batch = torch.stack(imgs_list)
        masks = _segment_batch(model, imgs_batch, voc_class_id, device)

        for j, idx in enumerate(batch_idx):
            img_pil, _ = cifar_dataset[idx]
            img_gray = np.array(img_pil.convert("L"))
            mask_j = masks[j]
            has_fg = mask_j.max() > 0
            if has_fg:
                fg_detected += 1
            if filter_empty and not has_fg:
                skipped += 1
                continue

            fname = f"{split_name}_{saved:06d}_c{cifar_class_idx}_{class_name}_i{idx:05d}.png"
            Image.fromarray(img_gray, mode="L").save(image_dir / fname)
            Image.fromarray(mask_j, mode="L").save(mask_dir / fname)
            saved += 1

        print(
            f"  [{split_name}/{class_name}] {min(start + batch_size, total)}/{total} "
            f"fg_detected={fg_detected} saved={saved} skipped={skipped}",
            flush=True,
        )

    return {
        "class": class_name,
        "cifar_class_idx": cifar_class_idx,
        "voc_class_id": voc_class_id,
        "total": total,
        "fg_detected": fg_detected,
        "saved": saved,
        "skipped": skipped,
        "fg_rate": round(fg_detected / max(total, 1), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CIFAR-10 pseudo-GT with DeepLabV3")
    parser.add_argument("--cifar-root", default="data/cifar10", help="CIFAR-10 root")
    parser.add_argument("--out-root", default="data/cifar10_deeplabv3_cat", help="Output root")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--filter-empty-train", action="store_true", default=True,
                        help="Remove training images with no foreground detected")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[info] loading DeepLabV3-ResNet101...", flush=True)
    model = _build_segmodel(device)

    print("[info] loading CIFAR-10...", flush=True)
    train_ds = datasets.CIFAR10(root=args.cifar_root, train=True, download=True)
    test_ds = datasets.CIFAR10(root=args.cifar_root, train=False, download=True)

    print("[info] processing train/cat...", flush=True)
    train_stats = _process_class(
        model=model,
        cifar_dataset=train_ds,
        cifar_class_idx=3,
        voc_class_id=VOC_CAT,
        class_name="cat",
        split_name="train",
        out_root=out_root,
        device=device,
        batch_size=args.batch_size,
        filter_empty=args.filter_empty_train,
    )

    print("[info] processing val/horse...", flush=True)
    val_stats = _process_class(
        model=model,
        cifar_dataset=test_ds,
        cifar_class_idx=7,
        voc_class_id=VOC_HORSE,
        class_name="horse",
        split_name="val",
        out_root=out_root,
        device=device,
        batch_size=args.batch_size,
        filter_empty=False,
    )

    summary = {
        "method": "deeplabv3_resnet101_pseudo_gt",
        "train": train_stats,
        "val": val_stats,
    }
    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
