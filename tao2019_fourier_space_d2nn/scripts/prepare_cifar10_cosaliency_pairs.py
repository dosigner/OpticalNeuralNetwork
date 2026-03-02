"""Prepare CIFAR-10 saliency pairs in class-wise folder format.

Output layout:
  <out_root>/train/images/<class>/*.png
  <out_root>/train/targets/<class>/*.png
  <out_root>/val/images/<class>/*.png
  <out_root>/val/targets/<class>/*.png

This implements the RBD (Robust Background Detection) method from
Zhu et al. (CVPR 2014) "Saliency Optimization from Robust Background Detection":
- SLIC superpixel segmentation
- Geodesic distance on superpixel adjacency graph
- Boundary connectivity as background measure
- Background-weighted contrast as foreground cue
- Least-squares saliency optimization combining background, foreground, and smoothness terms
Each image is processed independently (no group/co-saliency concept).
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse.csgraph import shortest_path
from skimage.color import rgb2lab
from skimage.segmentation import slic


_LABELS = [
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


def _resolve_class_indices(classes_arg: str) -> list[int]:
    raw = str(classes_arg).strip()
    if not raw or raw.lower() == "all":
        return list(range(len(_LABELS)))

    selected: list[int] = []
    for token in [p.strip().lower() for p in raw.split(",") if p.strip()]:
        if token.isdigit():
            idx = int(token)
            if idx < 0 or idx >= len(_LABELS):
                raise ValueError(f"class index out of range: {idx}")
            selected.append(idx)
            continue
        if token not in _LABELS:
            raise ValueError(f"unknown class name: {token}")
        selected.append(_LABELS.index(token))

    # keep order, remove duplicates
    uniq: list[int] = []
    seen = set()
    for idx in selected:
        if idx not in seen:
            seen.add(idx)
            uniq.append(idx)
    if not uniq:
        raise ValueError("no valid classes selected")
    return uniq


@dataclass(frozen=True)
class CifarSplit:
    name: str
    images: np.ndarray  # [N, 32, 32, 3], uint8
    labels: np.ndarray  # [N], int64


def _load_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        obj = pickle.load(f, encoding="bytes")
    data = np.asarray(obj[b"data"], dtype=np.uint8)
    labels = np.asarray(obj[b"labels"], dtype=np.int64)
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


def _load_cifar10_splits(cifar_root: Path) -> tuple[CifarSplit, CifarSplit]:
    train_imgs = []
    train_lbs = []
    for i in range(1, 6):
        imgs, lbs = _load_cifar_batch(cifar_root / f"data_batch_{i}")
        train_imgs.append(imgs)
        train_lbs.append(lbs)
    tr_x = np.concatenate(train_imgs, axis=0)
    tr_y = np.concatenate(train_lbs, axis=0)

    te_x, te_y = _load_cifar_batch(cifar_root / "test_batch")
    return CifarSplit("train", tr_x, tr_y), CifarSplit("val", te_x, te_y)


def _normalize01(v: np.ndarray) -> np.ndarray:
    x = v.astype(np.float32)
    x = x - x.min()
    den = max(float(x.max()), 1e-8)
    return x / den


def _rgb_to_gray_u8(rgb_group: np.ndarray) -> np.ndarray:
    rgb = rgb_group.astype(np.float32)
    gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    return np.clip(gray + 0.5, 0.0, 255.0).astype(np.uint8)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb01 = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    return rgb2lab(rgb01).astype(np.float32)


# ---------------------------------------------------------------------------
# Zhu et al. (CVPR 2014) RBD algorithm
# ---------------------------------------------------------------------------


def _build_adjacency_graph(
    sp_labels: np.ndarray,
    mean_lab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build weighted adjacency matrix and identify boundary superpixels.

    Returns (W, is_boundary) where:
      W[i,j] = d_app(i,j) for adjacent superpixels (incl. boundary links), 0 otherwise.
      is_boundary[i] = True if superpixel i touches the image border.
    """
    h, w = sp_labels.shape
    n_sp = int(sp_labels.max()) + 1

    # Collect adjacent pairs via 4-connectivity (vectorized)
    adj_pairs: set[tuple[int, int]] = set()

    # Horizontal neighbours
    diff_h = sp_labels[:, :-1] != sp_labels[:, 1:]
    rr, cc = np.where(diff_h)
    for r, c in zip(rr, cc):
        a, b = int(sp_labels[r, c]), int(sp_labels[r, c + 1])
        adj_pairs.add((min(a, b), max(a, b)))

    # Vertical neighbours
    diff_v = sp_labels[:-1, :] != sp_labels[1:, :]
    rr, cc = np.where(diff_v)
    for r, c in zip(rr, cc):
        a, b = int(sp_labels[r, c]), int(sp_labels[r + 1, c])
        adj_pairs.add((min(a, b), max(a, b)))

    # Boundary superpixels: touching any image border
    is_boundary = np.zeros(n_sp, dtype=bool)
    is_boundary[sp_labels[0, :]] = True
    is_boundary[sp_labels[-1, :]] = True
    is_boundary[sp_labels[:, 0]] = True
    is_boundary[sp_labels[:, -1]] = True

    # Add edges between ALL boundary superpixels (Section 3.2, Figure 2)
    bnd_ids = np.where(is_boundary)[0]
    for i in range(len(bnd_ids)):
        for j in range(i + 1, len(bnd_ids)):
            adj_pairs.add((int(bnd_ids[i]), int(bnd_ids[j])))

    # Build weight matrix: edge weight = CIE-Lab Euclidean distance
    W = np.zeros((n_sp, n_sp), dtype=np.float64)
    for a, b in adj_pairs:
        d = float(np.linalg.norm(mean_lab[a] - mean_lab[b]))
        W[a, b] = d
        W[b, a] = d

    return W, is_boundary


def _compute_rbd_saliency(
    rgb_img: np.ndarray,
    *,
    n_segments: int = 20,
    compactness: float = 10.0,
    sigma_clr: float = 10.0,
    sigma_bndcon: float = 1.0,
    sigma_spa: float = 0.25,
    mu: float = 0.1,
) -> np.ndarray:
    """Compute saliency map using the RBD method (Zhu et al. CVPR 2014).

    Returns float32 array of shape (H, W) with values in [0, 1].
    """
    h, w = rgb_img.shape[:2]

    # --- Step 1: SLIC superpixels ---
    sp_labels = slic(
        rgb_img,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        convert2lab=True,
        enforce_connectivity=True,
    )
    n_sp = int(sp_labels.max()) + 1

    # Degenerate: single superpixel => entire image is one region
    if n_sp <= 1:
        return np.zeros((h, w), dtype=np.float32)

    # --- Step 2: Superpixel properties ---
    lab_img = _rgb_to_lab(rgb_img)  # [H, W, 3]
    mean_lab = np.zeros((n_sp, 3), dtype=np.float32)
    centroid = np.zeros((n_sp, 2), dtype=np.float32)  # (row, col) normalised
    for k in range(n_sp):
        mask = sp_labels == k
        mean_lab[k] = lab_img[mask].mean(axis=0)
        ys, xs = np.where(mask)
        centroid[k, 0] = ys.mean() / max(h - 1, 1)
        centroid[k, 1] = xs.mean() / max(w - 1, 1)

    # --- Step 3: Adjacency graph ---
    W, is_boundary = _build_adjacency_graph(sp_labels, mean_lab)

    # --- Step 4: Geodesic distance (Eq. 2) ---
    # Replace 0 (no edge) with inf for shortest_path, keep diagonal as 0
    graph = np.where(W > 0, W, np.inf)
    np.fill_diagonal(graph, 0.0)
    dist_geo = shortest_path(graph, method="FW", directed=False)

    # Handle any remaining inf (disconnected components)
    finite_mask = np.isfinite(dist_geo)
    if not np.all(finite_mask):
        max_finite = float(dist_geo[finite_mask].max()) if finite_mask.any() else 1.0
        dist_geo = np.where(finite_mask, dist_geo, 2.0 * max_finite)

    # --- Step 5: Soft area & boundary length (Eq. 3, 4) ---
    sigma_clr2 = 2.0 * sigma_clr * sigma_clr + 1e-12
    S = np.exp(-(dist_geo ** 2) / sigma_clr2)  # [n_sp, n_sp]
    area = S.sum(axis=1)  # Eq. 3: Area(p)
    len_bnd = S[:, is_boundary].sum(axis=1)  # Eq. 4: Lenbnd(p)

    # --- Step 6: Boundary connectivity (Eq. 5) ---
    bnd_con = len_bnd / np.sqrt(np.maximum(area, 1e-12))

    # --- Step 7: Background probability (Eq. 7) ---
    sigma_bndcon2 = 2.0 * sigma_bndcon * sigma_bndcon + 1e-12
    w_bg = 1.0 - np.exp(-(bnd_con ** 2) / sigma_bndcon2)

    # --- Step 8: Appearance distance & spatial weight ---
    d_app = np.zeros((n_sp, n_sp), dtype=np.float64)
    for i in range(n_sp):
        d_app[i] = np.linalg.norm(mean_lab[i] - mean_lab, axis=1)

    # Spatial weight (Eq. 6): w_spa(p,q) = exp(-d_spa^2 / (2*sigma_spa^2))
    sigma_spa2 = 2.0 * sigma_spa * sigma_spa + 1e-12
    d_spa = np.zeros((n_sp, n_sp), dtype=np.float64)
    for i in range(n_sp):
        d_spa[i] = np.linalg.norm(centroid[i] - centroid, axis=1)
    w_spa = np.exp(-(d_spa ** 2) / sigma_spa2)

    # --- Step 9: Background-weighted contrast (Eq. 8) ---
    wCtr = (d_app * w_spa * w_bg[np.newaxis, :]).sum(axis=1)

    # Foreground weight: normalise wCtr to [0, 1]
    w_fg = _normalize01(wCtr).astype(np.float64)

    # If no foreground signal at all, return zeros
    if w_fg.max() < 1e-8:
        return np.zeros((h, w), dtype=np.float32)

    # --- Step 10: Saliency optimisation (Eq. 9) ---
    # Build smoothness Laplacian from adjacency
    # w_ij = exp(-d_app^2 / (2*sigma_clr^2)) + mu  for adjacent pairs
    W_smooth = np.zeros((n_sp, n_sp), dtype=np.float64)
    for i in range(n_sp):
        for j in range(i + 1, n_sp):
            if W[i, j] > 0:  # adjacent (including boundary links)
                wij = np.exp(-(d_app[i, j] ** 2) / sigma_clr2) + mu
                W_smooth[i, j] = wij
                W_smooth[j, i] = wij

    degree = W_smooth.sum(axis=1)
    L = np.diag(degree) - W_smooth  # Graph Laplacian

    # Linear system: (diag(w_bg) + diag(w_fg) + L) s = w_fg
    A = np.diag(w_bg) + np.diag(w_fg) + L
    b = w_fg.copy()

    # Solve
    try:
        s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.zeros((h, w), dtype=np.float32)

    s = np.clip(s, 0.0, None)
    s = _normalize01(s).astype(np.float32)

    # --- Step 11: Map to pixels ---
    sal_map = s[sp_labels]
    return sal_map


# ---------------------------------------------------------------------------
# I/O: save image/target pairs
# ---------------------------------------------------------------------------


def _save_pairs(
    *,
    split_name: str,
    images: np.ndarray,
    labels: np.ndarray,
    out_root: Path,
    class_indices: list[int],
    max_per_class: int,
    seed: int,
    n_segments: int,
    compactness: float,
    sigma_clr: float,
    sigma_bndcon: float,
    sigma_spa: float,
    mu: float,
    image_folder: str,
    target_folder: str,
) -> dict:
    rng = np.random.default_rng(seed + (0 if split_name == "train" else 1000))

    image_dir = out_root / split_name / image_folder
    mask_dir = out_root / split_name / target_folder
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    num_saved = 0
    class_counts: dict[str, int] = {}

    for label in class_indices:
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        if max_per_class > 0:
            idx = idx[:max_per_class]
        n = int(len(idx))
        if n <= 0:
            continue

        class_name = _LABELS[label]
        class_counts[class_name] = n
        class_image_dir = image_dir / class_name
        class_mask_dir = mask_dir / class_name
        class_image_dir.mkdir(parents=True, exist_ok=True)
        class_mask_dir.mkdir(parents=True, exist_ok=True)

        for li, sample_idx in enumerate(idx):
            rgb_img = images[sample_idx]
            gray_img = _rgb_to_gray_u8(rgb_img[None, ...])[0]
            sal_map = _compute_rbd_saliency(
                rgb_img,
                n_segments=n_segments,
                compactness=compactness,
                sigma_clr=sigma_clr,
                sigma_bndcon=sigma_bndcon,
                sigma_spa=sigma_spa,
                mu=mu,
            )

            fname = f"{split_name}_{num_saved:06d}_c{label}_{class_name}_i{int(sample_idx):05d}.png"
            Image.fromarray(gray_img, mode="L").save(class_image_dir / fname)
            mask_u8 = np.clip(sal_map * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
            Image.fromarray(mask_u8, mode="L").save(class_mask_dir / fname)
            num_saved += 1

        print(f"[{split_name}] class={class_name:>10s} saved={class_counts[class_name]}", flush=True)

    return {"num_saved": num_saved, "class_counts": class_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 RBD saliency pair folders")
    parser.add_argument(
        "--cifar-root",
        default="data/cifar10/cifar-10-batches-py",
        help="Path to CIFAR-10 python batch directory",
    )
    parser.add_argument(
        "--out-root",
        default="data/cifar10_object",
        help="Output directory with train/val + class-wise images/targets folders",
    )
    parser.add_argument("--image-folder", default="images", help="Image folder name under each split")
    parser.add_argument("--target-folder", default="targets", help="Target folder name under each split")
    parser.add_argument(
        "--classes",
        default="all",
        help="Comma-separated class names/indices (e.g., airplane or 0,3). Use 'all' for all classes.",
    )
    parser.add_argument("--train-max-per-class", type=int, default=0, help="0 means use all")
    parser.add_argument("--val-max-per-class", type=int, default=0, help="0 means use all")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-segments", type=int, default=20, help="Target number of SLIC superpixels")
    parser.add_argument("--compactness", type=float, default=10.0, help="SLIC compactness parameter")
    parser.add_argument("--sigma-clr", type=float, default=10.0, help="Color sigma for geodesic soft-assignment and smoothness (Eq.3,10)")
    parser.add_argument("--sigma-bndcon", type=float, default=1.0, help="Boundary connectivity sigma (Eq.7)")
    parser.add_argument("--sigma-spa", type=float, default=0.25, help="Spatial distance sigma for contrast weighting (Eq.6)")
    parser.add_argument("--mu", type=float, default=0.1, help="Smoothness regularization constant (Eq.10)")
    args = parser.parse_args()

    cifar_root = Path(args.cifar_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    class_indices = _resolve_class_indices(args.classes)
    class_names = [_LABELS[i] for i in class_indices]
    print(f"[info] selected classes: {class_names}", flush=True)

    train, val = _load_cifar10_splits(cifar_root)

    rbd_params = dict(
        n_segments=int(args.n_segments),
        compactness=float(args.compactness),
        sigma_clr=float(args.sigma_clr),
        sigma_bndcon=float(args.sigma_bndcon),
        sigma_spa=float(args.sigma_spa),
        mu=float(args.mu),
    )

    print("[info] processing train split...", flush=True)
    train_stats = _save_pairs(
        split_name="train",
        images=train.images,
        labels=train.labels,
        out_root=out_root,
        class_indices=class_indices,
        max_per_class=int(args.train_max_per_class),
        seed=int(args.seed),
        image_folder=str(args.image_folder),
        target_folder=str(args.target_folder),
        **rbd_params,
    )

    print("[info] processing val split...", flush=True)
    val_stats = _save_pairs(
        split_name="val",
        images=val.images,
        labels=val.labels,
        out_root=out_root,
        class_indices=class_indices,
        max_per_class=int(args.val_max_per_class),
        seed=int(args.seed),
        image_folder=str(args.image_folder),
        target_folder=str(args.target_folder),
        **rbd_params,
    )

    summary = {
        "method": "rbd_saliency_zhu2014",
        "cifar_root": str(cifar_root),
        "out_root": str(out_root),
        "params": {
            "classes": class_names,
            "seed": int(args.seed),
            "image_folder": str(args.image_folder),
            "target_folder": str(args.target_folder),
            "train_max_per_class": int(args.train_max_per_class),
            "val_max_per_class": int(args.val_max_per_class),
            **rbd_params,
        },
        "train": train_stats,
        "val": val_stats,
    }
    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
