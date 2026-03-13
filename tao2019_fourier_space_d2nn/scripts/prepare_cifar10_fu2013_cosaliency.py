"""Generate CIFAR-10 pseudo-GT saliency maps with a Fu 2013-style co-saliency pipeline.

This script intentionally implements a single, dedicated path for the paper-like CIFAR
saliency supervision we want in this repo:

- train split: CIFAR-10 train "cat"
- val split: CIFAR-10 test "horse"
- grouped co-saliency with fixed group size
- grayscale images saved for optical training
- continuous 0..255 saliency maps saved as pseudo-GT

The implementation follows the cluster-based co-saliency structure described by
H. Fu, X. Cao, and Z. Tu (TIP 2013), while keeping the engineering surface small
and reproducible for CIFAR-10.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from skimage.color import rgb2lab
from skimage.filters import gabor
from torchvision import datasets


_CIFAR_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass(frozen=True)
class GroupResult:
    image_indices: list[int]
    saliency_maps: list[np.ndarray]


def _normalize01(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32)
    y = y - y.min()
    den = max(float(y.max()), 1e-8)
    return y / den


def _standard_gaussian_norm(x: np.ndarray) -> np.ndarray:
    vals = x.astype(np.float32)
    if vals.size == 0:
        return vals
    mu = float(vals.mean())
    sigma = float(vals.std())
    if sigma < 1e-8:
        return np.full_like(vals, 0.5, dtype=np.float32)
    z = (vals - mu) / sigma
    return norm.cdf(z).astype(np.float32)


def _rgb01(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)


def _gray01(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    return np.clip(gray / 255.0, 0.0, 1.0)


def _texture_stack(gray01: np.ndarray, *, n_orientations: int, frequency: float, bandwidth: float) -> np.ndarray:
    feats = []
    for theta in np.linspace(0.0, np.pi, n_orientations, endpoint=False):
        real, imag = gabor(gray01, frequency=frequency, theta=float(theta), bandwidth=bandwidth)
        feats.append(np.sqrt(real**2 + imag**2))
    return np.stack(feats, axis=-1).astype(np.float32)


def _feature_map(rgb: np.ndarray, *, n_orientations: int, frequency: float, bandwidth: float) -> np.ndarray:
    rgb01 = _rgb01(rgb)
    lab = rgb2lab(rgb01).astype(np.float32)
    gray = _gray01(rgb)
    tex = _texture_stack(gray, n_orientations=n_orientations, frequency=frequency, bandwidth=bandwidth)
    return np.concatenate([lab, tex], axis=-1)


def _zscore(feat: np.ndarray) -> np.ndarray:
    mu = feat.mean(axis=0, keepdims=True)
    sigma = feat.std(axis=0, keepdims=True)
    return (feat - mu) / np.maximum(sigma, 1e-6)


def _cluster_positions(h: int, w: int) -> np.ndarray:
    ys, xs = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
    return np.stack([ys, xs], axis=-1).reshape(-1, 2).astype(np.float32)


def _kmeans_labels(features: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = features.shape[0]
    k = max(1, min(int(k), int(n)))
    centers, labels = kmeans2(features.astype(np.float64), k, minit="points", iter=30, seed=int(seed))
    return centers.astype(np.float32), labels.astype(np.int32)


def _cluster_variances(features: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> np.ndarray:
    dist2 = ((features - centers[labels]) ** 2).sum(axis=1)
    k = centers.shape[0]
    variances = np.zeros(k, dtype=np.float32)
    for idx in range(k):
        mask = labels == idx
        if not np.any(mask):
            variances[idx] = 1.0
            continue
        variances[idx] = max(float(dist2[mask].mean()), 1e-4)
    return variances


def _soft_assign(features: np.ndarray, centers: np.ndarray, variances: np.ndarray, cluster_scores: np.ndarray) -> np.ndarray:
    dist2 = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    weights = np.exp(-dist2 / (2.0 * variances[None, :]))
    sal = (weights * cluster_scores[None, :]).sum(axis=1)
    den = np.maximum(weights.sum(axis=1), 1e-8)
    return (sal / den).astype(np.float32)


def _single_image_saliency(
    rgb: np.ndarray,
    *,
    k_single: int,
    gabor_orientations: int,
    gabor_frequency: float,
    gabor_bandwidth: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    feat = _feature_map(
        rgb,
        n_orientations=gabor_orientations,
        frequency=gabor_frequency,
        bandwidth=gabor_bandwidth,
    ).reshape(-1, 3 + gabor_orientations)
    feat = _zscore(feat)
    pos = _cluster_positions(h, w)
    center = np.array([0.5, 0.5], dtype=np.float32)

    centers, labels = _kmeans_labels(feat, k=k_single, seed=seed)
    variances = _cluster_variances(feat, centers, labels)

    cluster_sizes = np.bincount(labels, minlength=centers.shape[0]).astype(np.float32)
    cluster_probs = cluster_sizes / max(float(cluster_sizes.sum()), 1.0)
    centroids = np.zeros((centers.shape[0], 2), dtype=np.float32)
    for idx in range(centers.shape[0]):
        mask = labels == idx
        centroids[idx] = pos[mask].mean(axis=0) if np.any(mask) else center

    contrast = np.zeros(centers.shape[0], dtype=np.float32)
    for idx in range(centers.shape[0]):
        d = np.linalg.norm(centers[idx][None, :] - centers, axis=1)
        contrast[idx] = float((cluster_probs * d).sum())

    spatial = np.exp(-9.0 * np.sum((centroids - center[None, :]) ** 2, axis=1)).astype(np.float32)
    cue_contrast = _standard_gaussian_norm(contrast)
    cue_spatial = _standard_gaussian_norm(spatial)
    cluster_scores = cue_contrast * cue_spatial
    sal = _soft_assign(feat, centers, variances, cluster_scores).reshape(h, w)
    return _normalize01(sal), feat


def _corresponding_cue(labels: np.ndarray, image_ids: np.ndarray, k: int, m: int) -> np.ndarray:
    cue = np.zeros(k, dtype=np.float32)
    for idx in range(k):
        mask = labels == idx
        if not np.any(mask):
            cue[idx] = 0.5
            continue
        per_image = np.zeros(m, dtype=np.float32)
        total = float(mask.sum())
        for image_id in range(m):
            per_image[image_id] = float(np.logical_and(mask, image_ids == image_id).sum()) / total
        cue[idx] = 1.0 / (float(per_image.var()) + 1.0)
    return cue


def _group_cosaliency(
    rgbs: list[np.ndarray],
    *,
    k_single: int,
    k_group_base: int,
    k_group_cap: int,
    gabor_orientations: int,
    gabor_frequency: float,
    gabor_bandwidth: float,
    seed: int,
) -> GroupResult:
    m = len(rgbs)
    single_maps: list[np.ndarray] = []
    single_feats: list[np.ndarray] = []
    image_ids = []
    for idx, rgb in enumerate(rgbs):
        s_map, s_feat = _single_image_saliency(
            rgb,
            k_single=k_single,
            gabor_orientations=gabor_orientations,
            gabor_frequency=gabor_frequency,
            gabor_bandwidth=gabor_bandwidth,
            seed=seed + idx,
        )
        single_maps.append(s_map)
        single_feats.append(s_feat)
        image_ids.append(np.full(s_feat.shape[0], idx, dtype=np.int32))

    all_feat = np.concatenate(single_feats, axis=0)
    image_ids_arr = np.concatenate(image_ids, axis=0)
    k_group = min(int(k_group_base * m), int(k_group_cap))
    centers, labels = _kmeans_labels(all_feat, k=k_group, seed=seed + 7919)
    variances = _cluster_variances(all_feat, centers, labels)

    cluster_sizes = np.bincount(labels, minlength=centers.shape[0]).astype(np.float32)
    cluster_probs = cluster_sizes / max(float(cluster_sizes.sum()), 1.0)
    contrast = np.zeros(centers.shape[0], dtype=np.float32)
    for idx in range(centers.shape[0]):
        d = np.linalg.norm(centers[idx][None, :] - centers, axis=1)
        contrast[idx] = float((cluster_probs * d).sum())

    h, w = rgbs[0].shape[:2]
    pos = _cluster_positions(h, w)
    center = np.array([0.5, 0.5], dtype=np.float32)
    all_pos = np.concatenate([pos for _ in range(m)], axis=0)
    centroids = np.zeros((centers.shape[0], 2), dtype=np.float32)
    for idx in range(centers.shape[0]):
        mask = labels == idx
        centroids[idx] = all_pos[mask].mean(axis=0) if np.any(mask) else center
    spatial = np.exp(-9.0 * np.sum((centroids - center[None, :]) ** 2, axis=1)).astype(np.float32)

    corresponding = _corresponding_cue(labels, image_ids_arr, centers.shape[0], m)

    flat_single = np.concatenate([s.reshape(-1) for s in single_maps], axis=0)
    single_cluster = np.zeros(centers.shape[0], dtype=np.float32)
    for idx in range(centers.shape[0]):
        mask = labels == idx
        single_cluster[idx] = float(flat_single[mask].mean()) if np.any(mask) else 0.5

    cue_single = _standard_gaussian_norm(single_cluster)
    cue_contrast = _standard_gaussian_norm(contrast)
    cue_spatial = _standard_gaussian_norm(spatial)
    cue_corresponding = _standard_gaussian_norm(corresponding)
    cluster_scores = cue_single * cue_contrast * cue_spatial * cue_corresponding

    saliency_maps = []
    offset = 0
    for feat in single_feats:
        pix_scores = _soft_assign(feat, centers, variances, cluster_scores).reshape(h, w)
        saliency_maps.append(_normalize01(pix_scores))
        offset += feat.shape[0]
    return GroupResult(image_indices=list(range(m)), saliency_maps=saliency_maps)


def _save_gray_u8(arr01: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr01 * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _split_indices(ds: datasets.CIFAR10, class_idx: int) -> list[int]:
    return [i for i, (_, label) in enumerate(ds) if int(label) == int(class_idx)]


def _chunks(items: list[int], size: int) -> list[list[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _build_group_order(indices: list[int], *, group_size: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(indices))
    return _chunks(shuffled, group_size)


def _process_split(
    *,
    ds: datasets.CIFAR10,
    class_name: str,
    split_name: str,
    out_root: Path,
    group_size: int,
    seed: int,
    max_groups: int | None,
    k_single: int,
    k_group_base: int,
    k_group_cap: int,
    gabor_orientations: int,
    gabor_frequency: float,
    gabor_bandwidth: float,
) -> dict[str, object]:
    class_idx = _CIFAR_LABELS.index(class_name)
    indices = _split_indices(ds, class_idx)
    groups = _build_group_order(indices, group_size=group_size, seed=seed)
    if max_groups is not None:
        groups = groups[: int(max_groups)]

    image_dir = out_root / split_name / "images"
    mask_dir = out_root / split_name / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    mean_mask = 0.0
    mean_active = 0.0
    for group_idx, group in enumerate(groups):
        rgbs: list[np.ndarray] = []
        grays: list[np.ndarray] = []
        for dataset_idx in group:
            img_pil, _ = ds[int(dataset_idx)]
            rgb = np.asarray(img_pil, dtype=np.uint8)
            rgbs.append(rgb)
            grays.append(np.asarray(img_pil.convert("L"), dtype=np.uint8))

        result = _group_cosaliency(
            rgbs,
            k_single=k_single,
            k_group_base=k_group_base,
            k_group_cap=k_group_cap,
            gabor_orientations=gabor_orientations,
            gabor_frequency=gabor_frequency,
            gabor_bandwidth=gabor_bandwidth,
            seed=seed + group_idx * 101,
        )

        for local_idx, dataset_idx in enumerate(group):
            name = f"{split_name}_{saved:06d}_c{class_idx}_{class_name}_g{group_idx:04d}_i{dataset_idx:05d}.png"
            img01 = grays[local_idx].astype(np.float32) / 255.0
            sal01 = result.saliency_maps[local_idx]
            _save_gray_u8(img01, image_dir / name)
            _save_gray_u8(sal01, mask_dir / name)
            mean_mask += float(sal01.mean())
            mean_active += float((sal01 >= 0.5).mean())
            saved += 1

        if (group_idx + 1) % 20 == 0 or group_idx == 0 or group_idx + 1 == len(groups):
            print(
                f"[{split_name}/{class_name}] groups={group_idx + 1}/{len(groups)} images={saved}",
                flush=True,
            )

    denom = max(saved, 1)
    return {
        "class": class_name,
        "class_idx": class_idx,
        "group_size": int(group_size),
        "num_groups": len(groups),
        "num_images": int(saved),
        "mean_mask_intensity": round(mean_mask / denom, 6),
        "mean_active_ratio_at_0_5": round(mean_active / denom, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CIFAR-10 paper-like co-saliency GT (Fu 2013 style)")
    parser.add_argument("--cifar-root", default="data/cifar10", help="CIFAR-10 root")
    parser.add_argument("--out-root", default="data/cifar10_cosaliency_fu2013_g5", help="Output root")
    parser.add_argument("--group-size", type=int, default=5, help="Images per co-saliency group")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group shuffling")
    parser.add_argument("--max-groups", type=int, default=None, help="Optional limit for quick dry-runs")
    parser.add_argument("--k-single", type=int, default=6, help="Single-image cluster count (paper uses 6)")
    parser.add_argument("--k-group-base", type=int, default=3, help="K2 base multiplier in min(base*M, cap)")
    parser.add_argument("--k-group-cap", type=int, default=20, help="Upper cap for group clusters")
    parser.add_argument("--gabor-orientations", type=int, default=8, help="Number of Gabor orientations")
    parser.add_argument("--gabor-frequency", type=float, default=0.25, help="Gabor frequency")
    parser.add_argument("--gabor-bandwidth", type=float, default=1.0, help="Gabor bandwidth")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.CIFAR10(root=args.cifar_root, train=True, download=True)
    test_ds = datasets.CIFAR10(root=args.cifar_root, train=False, download=True)

    train_stats = _process_split(
        ds=train_ds,
        class_name="cat",
        split_name="train",
        out_root=out_root,
        group_size=int(args.group_size),
        seed=int(args.seed),
        max_groups=args.max_groups,
        k_single=int(args.k_single),
        k_group_base=int(args.k_group_base),
        k_group_cap=int(args.k_group_cap),
        gabor_orientations=int(args.gabor_orientations),
        gabor_frequency=float(args.gabor_frequency),
        gabor_bandwidth=float(args.gabor_bandwidth),
    )
    val_stats = _process_split(
        ds=test_ds,
        class_name="horse",
        split_name="val",
        out_root=out_root,
        group_size=int(args.group_size),
        seed=int(args.seed) + 1,
        max_groups=args.max_groups,
        k_single=int(args.k_single),
        k_group_base=int(args.k_group_base),
        k_group_cap=int(args.k_group_cap),
        gabor_orientations=int(args.gabor_orientations),
        gabor_frequency=float(args.gabor_frequency),
        gabor_bandwidth=float(args.gabor_bandwidth),
    )

    manifest = {
        "method": "fu2013_cluster_based_cosaliency_approx",
        "citation": "H. Fu, X. Cao, Z. Tu, Cluster-based co-saliency detection, IEEE TIP 2013",
        "notes": [
            "Dedicated paper-like CIFAR saliency GT path.",
            "Single-image cues: cluster contrast + spatial cue.",
            "Multi-image cues: single-saliency cluster mean + inter-image contrast + spatial + corresponding cue.",
            "Continuous saliency maps saved as grayscale 0..255 pseudo-GT.",
        ],
        "params": {
            "group_size": int(args.group_size),
            "seed": int(args.seed),
            "k_single": int(args.k_single),
            "k_group_formula": f"min({int(args.k_group_base)}*M, {int(args.k_group_cap)})",
            "gabor_orientations": int(args.gabor_orientations),
            "gabor_frequency": float(args.gabor_frequency),
            "gabor_bandwidth": float(args.gabor_bandwidth),
        },
        "train": train_stats,
        "val": val_stats,
    }
    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
