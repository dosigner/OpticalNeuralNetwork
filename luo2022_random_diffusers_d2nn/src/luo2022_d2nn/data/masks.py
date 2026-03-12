"""Binary support mask generation utilities."""

import torch


def make_support_mask(
    amplitude: torch.Tensor,
    strategy: str = "positive_pixels",
) -> torch.Tensor:
    """Create a binary support mask from an amplitude field.

    Parameters
    ----------
    amplitude : torch.Tensor
        Input amplitude tensor (any shape).
    strategy : str
        Masking strategy.  Currently supported:
        - ``"positive_pixels"``: mask is 1 where amplitude > 0, else 0.

    Returns
    -------
    torch.Tensor
        Binary mask with the same shape as *amplitude*.
    """
    if strategy == "positive_pixels":
        return (amplitude > 0).float()
    raise ValueError(f"Unknown mask strategy: {strategy!r}")
