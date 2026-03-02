"""Live training log utilities for terminal and file output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import sys
from typing import Any, TextIO

try:
    from colorama import Fore, Style, init as _colorama_init

    _colorama_init(autoreset=True)
    _HAS_COLORAMA = True
except Exception:  # pragma: no cover - graceful fallback without colorama
    _HAS_COLORAMA = False

    class _NoColor:
        BLACK = ""
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""
        RESET = ""

    class _NoStyle:
        BRIGHT = ""
        NORMAL = ""
        RESET_ALL = ""

    Fore = _NoColor()  # type: ignore
    Style = _NoStyle()  # type: ignore


def _fmt_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _fmt_float(v: Any, digits: int = 4) -> str:
    try:
        x = float(v)
        if not math.isfinite(x):
            return "-"
        return f"{x:.{digits}f}"
    except Exception:
        return "-"


def _fmt_mem(v: Any) -> str:
    try:
        return f"{float(v):.2f}GB"
    except Exception:
        return "-"


@dataclass
class LiveLogger:
    """Console/file logger for live training progress."""

    run_dir: Path
    task: str
    total_epochs: int
    log_interval_steps: int = 20
    use_color: bool = True
    show_cuda_memory: bool = True
    stream: TextIO | None = None

    def __post_init__(self) -> None:
        self.stream = self.stream or sys.stdout
        self.log_interval_steps = max(1, int(self.log_interval_steps))
        self._best_metric: float | None = None
        logs_dir = self.run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = logs_dir / "train.log"
        self._append_file(f"[{_fmt_time()}] live logging started task={self.task}")

    def _color(self, text: str, color: str) -> str:
        if not (self.use_color and _HAS_COLORAMA):
            return text
        return f"{color}{text}{Style.RESET_ALL}"

    def _emit(self, line: str) -> None:
        assert self.stream is not None
        print(line, file=self.stream, flush=True)
        self._append_file(line)

    def _append_file(self, line: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

    def start(self, *, experiment_name: str, device: str) -> None:
        tag = self._color("TRAIN", Fore.MAGENTA)
        self._emit(
            f"[{_fmt_time()}] [{tag}] exp={experiment_name} task={self.task} device={device} "
            f"epochs={self.total_epochs} log_every={self.log_interval_steps}steps"
        )

    def on_step(self, data: dict[str, Any]) -> None:
        step = int(data.get("step", 0))
        total_steps = data.get("total_steps")
        if step <= 0:
            return
        is_last = total_steps is not None and step >= int(total_steps)
        if (step % self.log_interval_steps) != 0 and not is_last:
            return

        phase = str(data.get("phase", "train"))
        epoch = int(data.get("epoch", 0))
        total_epochs = int(data.get("total_epochs", self.total_epochs))
        phase_tag = self._color(phase.upper(), Fore.GREEN if phase == "train" else Fore.CYAN)
        progress = f"{step}/{int(total_steps)}" if total_steps is not None else f"{step}"
        base = (
            f"[{_fmt_time()}] [{phase_tag}] "
            f"e{epoch}/{total_epochs} s{progress} "
            f"loss={_fmt_float(data.get('loss'))} avg={_fmt_float(data.get('avg_loss'))}"
        )
        metric_name = data.get("metric_name")
        metric_value = data.get("metric_value")
        if metric_name is not None:
            base += f" {metric_name}={_fmt_float(metric_value)}"
        sps = data.get("samples_per_sec")
        if sps is not None:
            base += f" {float(sps):.1f}sample/s"
        base += f" eta={_fmt_eta(data.get('eta_sec'))}"
        if self.show_cuda_memory:
            base += f" mem={_fmt_mem(data.get('gpu_mem_gb'))}/{_fmt_mem(data.get('gpu_mem_peak_gb'))}"
        self._emit(base)

    def on_epoch_end(self, data: dict[str, Any]) -> None:
        epoch = int(data.get("epoch", 0))
        total_epochs = int(data.get("total_epochs", self.total_epochs))
        if self.task == "classification":
            metric_name = "val_acc"
            metric = float(data.get("val_acc", 0.0))
            line = (
                f"[{_fmt_time()}] [{self._color('EPOCH', Fore.YELLOW)}] "
                f"{epoch}/{total_epochs} "
                f"train_loss={_fmt_float(data.get('train_loss'))} train_acc={_fmt_float(data.get('train_acc'))} "
                f"val_loss={_fmt_float(data.get('val_loss'))} val_acc={_fmt_float(data.get('val_acc'))} "
                f"lr={_fmt_float(data.get('lr'), digits=6)}"
            )
            if "test_acc" in data:
                line += (
                    f" test_loss={_fmt_float(data.get('test_loss'))} "
                    f"test_acc={_fmt_float(data.get('test_acc'))}"
                )
        else:
            metric_name = "val_fmax"
            metric = float(data.get("val_fmax", float("nan")))
            val_fmax_computed = bool(data.get("val_fmax_computed", True))
            line = (
                f"[{_fmt_time()}] [{self._color('EPOCH', Fore.YELLOW)}] "
                f"{epoch}/{total_epochs} "
                f"train_loss={_fmt_float(data.get('train_loss'))} train_fmax={_fmt_float(data.get('train_fmax'))} "
                f"val_loss={_fmt_float(data.get('val_loss'))} val_fmax={_fmt_float(data.get('val_fmax'))} "
                f"lr={_fmt_float(data.get('lr'), digits=6)}"
            )
            if not val_fmax_computed:
                line += " val_fmax=SKIPPED"

        if math.isfinite(metric) and (self._best_metric is None or metric > self._best_metric):
            self._best_metric = metric
            line += f" best_{metric_name}={_fmt_float(metric)} *"
        elif self._best_metric is not None:
            line += f" best_{metric_name}={_fmt_float(self._best_metric)}"
        else:
            line += f" best_{metric_name}=-"
        self._emit(line)

    def finish(self, *, run_dir: Path) -> None:
        done = self._color("DONE", Fore.BLUE)
        self._emit(f"[{_fmt_time()}] [{done}] run_dir={run_dir}")
