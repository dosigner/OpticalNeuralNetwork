"""Binary grating targets for resolution testing."""

import torch


# Standard test periods (mm) from the paper
SUPPORTED_PERIODS_MM = [7.2, 8.4, 9.6, 10.8, 12.0]


def generate_grating_target(
    period_mm: float,
    dx_mm: float = 0.3,
    active_size: int = 160,
    final_size: int = 240,
) -> torch.Tensor:
    """Create a 3-bar binary grating centered in the field.

    Parameters
    ----------
    period_mm : float
        Full grating cycle (bar + gap) in millimetres.
    dx_mm : float
        Pixel pitch in millimetres.
    active_size : int
        Side length of the un-padded active region in pixels.
    final_size : int
        Side length after zero-padding in pixels.

    Returns
    -------
    torch.Tensor
        Shape ``(1, final_size, final_size)`` with values 0.0 or 1.0.
    """
    period_px = period_mm / dx_mm  # period in pixels
    bar_width_px = period_px / 2.0  # 50 % duty cycle

    # Total width of the 3-bar pattern: 3 bars + 2 gaps = 3*bar + 2*bar = 5*half-periods
    # More precisely: bar gap bar gap bar = 3*bar_width + 2*(period - bar_width)
    # With 50% duty cycle: 3*bar + 2*bar = 5*bar = 2.5 * period
    pattern_width_px = 3 * bar_width_px + 2 * (period_px - bar_width_px)

    # Build the active region
    active = torch.zeros(1, active_size, active_size)

    # Horizontal bars centred in the active region (matching paper Fig. 2)
    center_y = active_size / 2.0
    pattern_start = center_y - pattern_width_px / 2.0

    # Horizontal extent: centre the bars horizontally over the active region
    bar_length = min(pattern_width_px, active_size)
    center_x = active_size / 2.0
    x_start = int(round(center_x - bar_length / 2.0))
    x_end = int(round(center_x + bar_length / 2.0))
    x_start = max(x_start, 0)
    x_end = min(x_end, active_size)

    for bar_idx in range(3):
        # Each bar starts at: pattern_start + bar_idx * period_px
        y_start_f = pattern_start + bar_idx * period_px
        y_end_f = y_start_f + bar_width_px

        y_s = int(round(y_start_f))
        y_e = int(round(y_end_f))
        y_s = max(y_s, 0)
        y_e = min(y_e, active_size)

        if y_e > y_s:
            active[0, y_s:y_e, x_start:x_end] = 1.0

    # Zero-pad to final_size
    pad_each = (final_size - active_size) // 2
    output = torch.nn.functional.pad(active, (pad_each, pad_each, pad_each, pad_each), value=0.0)

    return output
