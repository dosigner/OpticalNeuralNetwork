"""Losses for classification and saliency."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tao2019_fd2nn.optics.fft2c import gamma_flip2d


def classification_loss(
    energies: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_mode: str = "cross_entropy",
    leakage_ratio: torch.Tensor | None = None,
    leakage_weight: float = 0.1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Classification loss on detector energies with optional leakage penalty."""

    logits = energies / max(float(temperature), 1e-8)
    if loss_mode == "mse_onehot":
        onehot = F.one_hot(labels.long(), num_classes=energies.shape[1]).to(energies.dtype)
        probs = logits / logits.sum(dim=1, keepdim=True).clamp_min(1e-8)
        loss = F.mse_loss(probs, onehot)
    else:
        loss = F.cross_entropy(logits, labels.long())
    if leakage_ratio is not None:
        loss = loss + float(leakage_weight) * leakage_ratio.mean()
    return loss


def saliency_mse_loss(
    pred_intensity: torch.Tensor,
    gt_intensity: torch.Tensor,
    *,
    gamma_flip: bool = True,
) -> torch.Tensor:
    """MSE with optional Γ flip alignment for target map."""

    target = gamma_flip2d(gt_intensity) if gamma_flip else gt_intensity
    return F.mse_loss(pred_intensity, target)


def binary_cross_entropy_loss(
    pred_intensity: torch.Tensor,
    gt_intensity: torch.Tensor,
    *,
    gamma_flip: bool = True,
) -> torch.Tensor:
    """BCE loss for saliency detection with sharper decision boundaries."""

    target = gamma_flip2d(gt_intensity) if gamma_flip else gt_intensity
    # Use PyTorch's numerically stable BCE implementation
    return F.binary_cross_entropy(pred_intensity.clamp(1e-6, 1.0 - 1e-6), target)


def iou_loss(
    pred_intensity: torch.Tensor,
    gt_intensity: torch.Tensor,
    *,
    gamma_flip: bool = True,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """IoU-based loss to encourage spatial overlap and discourage blob patterns.

    Returns 1 - IoU so that minimizing loss maximizes IoU.
    """

    target = gamma_flip2d(gt_intensity) if gamma_flip else gt_intensity

    # Soft IoU (differentiable) - treat intensities as soft masks
    intersection = (pred_intensity * target).sum(dim=(-2, -1))
    union = (pred_intensity + target - pred_intensity * target).sum(dim=(-2, -1))
    iou = intersection / (union + eps)

    # Return 1 - IoU as loss (minimize to maximize IoU)
    return (1.0 - iou).mean()


def structure_loss(
    pred_intensity: torch.Tensor,
    gt_intensity: torch.Tensor,
    *,
    gamma_flip: bool = True,
) -> torch.Tensor:
    """Edge-aware structure loss using Sobel gradient matching.

    Encourages learning object boundaries rather than uniform blobs.
    """

    target = gamma_flip2d(gt_intensity) if gamma_flip else gt_intensity

    # Sobel filters for horizontal and vertical gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred_intensity.dtype, device=pred_intensity.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred_intensity.dtype, device=pred_intensity.device)

    # Reshape for conv2d: [1, 1, 3, 3]
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    # Compute gradients for pred and target
    # Ensure 4D: [B, 1, H, W] if needed
    if pred_intensity.ndim == 3:
        pred_4d = pred_intensity.unsqueeze(1)
        target_4d = target.unsqueeze(1)
    elif pred_intensity.ndim == 4 and pred_intensity.shape[1] == 1:
        pred_4d = pred_intensity
        target_4d = target
    else:
        # [B, C, H, W] with C > 1: process per channel and average
        pred_4d = pred_intensity
        target_4d = target

    # Apply Sobel filters with padding
    pred_grad_x = F.conv2d(pred_4d, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred_4d, sobel_y, padding=1)
    target_grad_x = F.conv2d(target_4d, sobel_x, padding=1)
    target_grad_y = F.conv2d(target_4d, sobel_y, padding=1)

    # Gradient magnitude
    pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
    target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)

    # MSE on gradient magnitudes
    return F.mse_loss(pred_grad_mag, target_grad_mag)


def center_bias_penalty(
    pred_intensity: torch.Tensor,
    *,
    sigma_ratio: float = 0.25,
) -> torch.Tensor:
    """Penalize predictions that concentrate mass in center (discourage exploiting center bias).

    Creates a Gaussian weight map that penalizes center-heavy predictions.

    Args:
        pred_intensity: [B, H, W] or [B, 1, H, W] predicted saliency
        sigma_ratio: Gaussian std as fraction of image size (default 0.25)

    Returns:
        Scalar penalty term (higher when prediction is center-biased)
    """

    if pred_intensity.ndim == 4:
        pred = pred_intensity.squeeze(1)  # [B, H, W]
    else:
        pred = pred_intensity

    B, H, W = pred.shape
    device = pred.device
    dtype = pred.dtype

    # Create center Gaussian weight
    y_coords = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    x_coords = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    sigma = float(sigma_ratio)
    gaussian_weight = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))

    # Normalize Gaussian to sum to 1
    gaussian_weight = gaussian_weight / gaussian_weight.sum()

    # Compute weighted sum of prediction (mass in center)
    center_mass = (pred * gaussian_weight.unsqueeze(0)).sum(dim=(-2, -1))

    # Higher penalty when more mass is in center
    return center_mass.mean()


def saliency_structured_loss(
    pred_intensity: torch.Tensor,
    gt_intensity: torch.Tensor,
    *,
    gamma_flip: bool = True,
    loss_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Multi-component structured loss for saliency detection.

    Combines BCE, IoU, structure, and center bias penalty to encourage
    learning structured saliency maps instead of uniform blobs.

    Args:
        pred_intensity: [B, H, W] or [B, 1, H, W] predicted saliency
        gt_intensity: [B, H, W] or [B, 1, H, W] ground truth
        gamma_flip: Apply gamma flip to target
        loss_weights: Dict with keys 'bce', 'iou', 'structure', 'center_penalty'
                      Default: {'bce': 1.0, 'iou': 2.0, 'structure': 1.0, 'center_penalty': 0.1}

    Returns:
        Weighted sum of loss components
    """

    if loss_weights is None:
        loss_weights = {
            'bce': 1.0,
            'iou': 2.0,
            'structure': 1.0,
            'center_penalty': 0.1,
        }

    w_bce = float(loss_weights.get('bce', 1.0))
    w_iou = float(loss_weights.get('iou', 2.0))
    w_struct = float(loss_weights.get('structure', 1.0))
    w_center = float(loss_weights.get('center_penalty', 0.1))

    loss = 0.0

    if w_bce > 0:
        loss = loss + w_bce * binary_cross_entropy_loss(pred_intensity, gt_intensity, gamma_flip=gamma_flip)

    if w_iou > 0:
        loss = loss + w_iou * iou_loss(pred_intensity, gt_intensity, gamma_flip=gamma_flip)

    if w_struct > 0:
        loss = loss + w_struct * structure_loss(pred_intensity, gt_intensity, gamma_flip=gamma_flip)

    if w_center > 0:
        loss = loss + w_center * center_bias_penalty(pred_intensity)

    return loss
