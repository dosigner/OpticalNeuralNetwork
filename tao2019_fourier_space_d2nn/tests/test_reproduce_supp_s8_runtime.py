from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_supp_s8_sensitivity_cls.py"
    script_dir = str(script_path.parent)
    src_dir = str(script_path.resolve().parents[1] / "src")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    spec = importlib.util.spec_from_file_location("reproduce_supp_s8_sensitivity_cls", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_overrides_disable_multiprocessing_when_num_workers_zero() -> None:
    mod = _load_script_module()

    overrides = mod._training_overrides(train_batch_size=10, epochs=30, num_workers=0)

    assert overrides["training"]["batch_size"] == 10
    assert overrides["training"]["epochs"] == 30
    assert overrides["training"]["num_workers"] == 0
    assert overrides["training"]["pin_memory"] is False
    assert overrides["training"]["persistent_workers"] is False


def test_training_overrides_keep_pin_memory_enabled_when_num_workers_positive() -> None:
    mod = _load_script_module()

    overrides = mod._training_overrides(train_batch_size=10, epochs=30, num_workers=2)

    assert overrides["training"]["num_workers"] == 2
    assert overrides["training"]["pin_memory"] is True
    assert overrides["training"]["persistent_workers"] is True


def test_parse_args_supports_parallel_train_limit() -> None:
    mod = _load_script_module()

    args = mod._parse_args(["--max-parallel-trains", "3"])

    assert args.max_parallel_trains == 3


def test_latest_completed_run_dir_ignores_incomplete_runs(tmp_path: Path) -> None:
    mod = _load_script_module()
    runs_root = tmp_path / "runs"
    exp_dir = runs_root / "demo_exp"
    incomplete = exp_dir / "260309_010000"
    complete = exp_dir / "260309_020000"
    incomplete.mkdir(parents=True)
    (complete / "checkpoints").mkdir(parents=True)
    (complete / "checkpoints" / "final.pt").write_text("ok", encoding="utf-8")

    resolved = mod._latest_completed_run_dir(runs_root, "demo_exp", checkpoint_name="final.pt")

    assert resolved == complete


def test_launch_parallel_training_detaches_children_and_captures_logs(tmp_path: Path, monkeypatch) -> None:
    mod = _load_script_module()
    config_path = tmp_path / "demo.yaml"
    config_path.write_text("experiment:\n  name: demo_exp\n", encoding="utf-8")
    temp_cfg = tmp_path / "tmp_demo.yaml"
    temp_cfg.write_text("experiment:\n  name: demo_exp\n", encoding="utf-8")
    monkeypatch.setattr(mod, "_write_temp_config", lambda base, overrides: temp_cfg)
    monkeypatch.setattr(mod.time, "sleep", lambda *_args, **_kwargs: None)

    calls: list[dict[str, Any]] = []

    class _FakeProc:
        def __init__(self, args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def poll(self) -> int:
            return 0

    def _fake_popen(args, **kwargs):
        calls.append({"args": args, **kwargs})
        return _FakeProc(args, **kwargs)

    monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)

    mod._launch_parallel_training(
        [config_path],
        project_root=tmp_path,
        train_batch_size=10,
        epochs=1,
        num_workers=0,
        max_parallel_trains=1,
    )

    assert len(calls) == 1
    assert calls[0]["start_new_session"] is True
    assert calls[0]["stderr"] == mod.subprocess.STDOUT
    assert calls[0]["stdout"] is not None
