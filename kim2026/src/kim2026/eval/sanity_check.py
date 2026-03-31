"""Physics sanity checks for D2NN experiments.

Run before/after training to catch metric bugs early.
Checks: vacuum Strehl, energy throughput, unitary CO preservation, Strehl bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from kim2026.eval.focal_utils import (
    WAVELENGTH_M,
    GRID_SIZE,
    WINDOW_M,
    D2NN_ARCH,
    prepare_field,
)
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.training.metrics import complex_overlap, strehl_ratio_correct

STREHL_PAD_FACTOR = 4


@dataclass
class SanityResult:
    name: str
    passed: bool
    value: float
    threshold: str
    message: str


@dataclass
class SanityReport:
    checks: list[SanityResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def print_report(self):
        print("\n" + "=" * 60)
        print("  PHYSICS SANITY CHECKS")
        print("=" * 60)
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            print(f"  [{status}] {c.name}: {c.value:.6f} ({c.threshold})")
            if not c.passed:
                print(f"         {c.message}")
        status = "ALL PASSED" if self.all_passed else "FAILURES DETECTED"
        print(f"\n  Result: {status}")
        print("=" * 60 + "\n")

    def to_dict(self) -> list[dict]:
        return [
            {"name": c.name, "passed": c.passed, "value": c.value,
             "threshold": c.threshold, "message": c.message}
            for c in self.checks
        ]


def _make_zero_phase_model(
    n: int = GRID_SIZE,
    wavelength_m: float = WAVELENGTH_M,
    window_m: float = WINDOW_M,
    arch: dict | None = None,
    device: str = "cpu",
) -> BeamCleanupD2NN:
    """Create a zero-phase D2NN (free-space propagation only)."""
    if arch is None:
        arch = D2NN_ARCH
    model = BeamCleanupD2NN(n=n, wavelength_m=wavelength_m, window_m=window_m, **arch)
    with torch.no_grad():
        for layer in model.layers:
            layer.phase.zero_()
    model.eval()
    return model.to(device)


@torch.no_grad()
def check_vacuum_strehl(
    loader: DataLoader,
    device: str = "cpu",
    strehl_min: float = 0.0,
    strehl_max: float = 1.05,
) -> SanityResult:
    """Verify corrected vacuum Strehl remains passive and finite."""
    d0 = _make_zero_phase_model(device=device)
    all_strehl = []

    for batch in loader:
        u_vac = batch["u_vacuum"].to(device)
        target = prepare_field(u_vac)
        pred = d0(target)
        sr = strehl_ratio_correct(
            pred,
            target.abs(),
            pad_factor=STREHL_PAD_FACTOR,
        )
        all_strehl.append(sr.cpu())
        break  # one batch is enough for sanity check

    mean_strehl = float(torch.cat(all_strehl).mean())
    passed = strehl_min <= mean_strehl <= strehl_max
    return SanityResult(
        name="Vacuum Strehl",
        passed=passed,
        value=mean_strehl,
        threshold=f"[{strehl_min}, {strehl_max}]",
        message=f"Zero-phase D2NN corrected Strehl={mean_strehl:.4f}, expected passive bound",
    )


@torch.no_grad()
def check_throughput(
    model: BeamCleanupD2NN,
    loader: DataLoader,
    device: str = "cpu",
    tp_min: float = 0.90,
    tp_max: float = 1.10,
) -> SanityResult:
    """Verify energy conservation: output/input energy ratio in [tp_min, tp_max]."""
    model.eval()
    total_in, total_out = 0.0, 0.0

    for batch in loader:
        inp = prepare_field(batch["u_turb"].to(device))
        pred = model(inp)
        total_in += inp.abs().square().sum().item()
        total_out += pred.abs().square().sum().item()
        break  # one batch is enough

    tp = total_out / max(total_in, 1e-12)
    passed = tp_min <= tp <= tp_max
    return SanityResult(
        name="Energy Throughput",
        passed=passed,
        value=tp,
        threshold=f"[{tp_min}, {tp_max}]",
        message=f"Throughput={tp:.4f}, energy {'gained' if tp > tp_max else 'lost'}",
    )


@torch.no_grad()
def check_unitary_co(
    model: BeamCleanupD2NN,
    loader: DataLoader,
    device: str = "cpu",
    tolerance: float = 0.01,
) -> SanityResult:
    """Verify D2NN preserves complex overlap (unitary property).

    |CO(D2NN(u_t), D2NN(u_v)) - CO(u_t, u_v)| < tolerance
    """
    model.eval()
    all_delta = []

    for batch in loader:
        u_turb = batch["u_turb"].to(device)
        u_vac = batch["u_vacuum"].to(device)
        inp = prepare_field(u_turb)
        target = prepare_field(u_vac)

        # CO at input
        co_in = complex_overlap(inp, target)

        # CO at D2NN output
        pred = model(inp)
        pred_target = model(target)
        co_out = complex_overlap(pred, pred_target)

        delta = (co_out - co_in).abs()
        all_delta.append(delta.cpu())
        break

    mean_delta = float(torch.cat(all_delta).mean())
    passed = mean_delta < tolerance
    return SanityResult(
        name="Unitary CO Preservation",
        passed=passed,
        value=mean_delta,
        threshold=f"|delta| < {tolerance}",
        message=f"CO delta={mean_delta:.6f}, D2NN may not be unitary",
    )


@torch.no_grad()
def check_strehl_bound(
    model: BeamCleanupD2NN,
    loader: DataLoader,
    device: str = "cpu",
    max_strehl: float = 1.05,
) -> SanityResult:
    """Verify trained D2NN Strehl <= max_strehl (passive device constraint)."""
    model.eval()
    all_strehl = []

    for batch in loader:
        u_turb = batch["u_turb"].to(device)
        u_vac = batch["u_vacuum"].to(device)
        inp = prepare_field(u_turb)
        target = prepare_field(u_vac)

        pred = model(inp)
        sr = strehl_ratio_correct(
            pred,
            target.abs(),
            pad_factor=STREHL_PAD_FACTOR,
        )
        all_strehl.append(sr.cpu())
        break

    mean_strehl = float(torch.cat(all_strehl).mean())
    passed = mean_strehl <= max_strehl
    return SanityResult(
        name="Strehl Upper Bound",
        passed=passed,
        value=mean_strehl,
        threshold=f"<= {max_strehl}",
        message=f"Strehl={mean_strehl:.4f} exceeds passive limit",
    )


def run_pre_training_checks(
    loader: DataLoader,
    device: str = "cpu",
    config: dict | None = None,
) -> SanityReport:
    """Run all pre-training sanity checks."""
    sc = config or {}
    report = SanityReport()
    report.checks.append(check_vacuum_strehl(
        loader, device,
        strehl_min=sc.get("vacuum_strehl_min", 0.0),
        strehl_max=sc.get("vacuum_strehl_max", 1.05),
    ))
    report.print_report()
    return report


def run_post_training_checks(
    model: BeamCleanupD2NN,
    loader: DataLoader,
    device: str = "cpu",
    config: dict | None = None,
) -> SanityReport:
    """Run all post-training sanity checks on a trained model."""
    sc = config or {}
    report = SanityReport()
    report.checks.append(check_throughput(
        model, loader, device,
        tp_min=sc.get("throughput_min", 0.90),
        tp_max=sc.get("throughput_max", 1.10),
    ))
    report.checks.append(check_unitary_co(
        model, loader, device,
        tolerance=sc.get("unitary_co_tolerance", 0.01),
    ))
    report.checks.append(check_strehl_bound(
        model, loader, device,
        max_strehl=sc.get("vacuum_strehl_max", 1.05),
    ))
    report.print_report()
    return report
