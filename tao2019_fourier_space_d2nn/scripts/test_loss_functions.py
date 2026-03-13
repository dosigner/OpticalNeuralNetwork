#!/usr/bin/env python3
"""Test different loss function configurations for F-D2NN saliency detection.

This script runs multiple experiments sequentially with different loss function settings:
1. Baseline MSE (current)
2. Structured loss with default weights
3. BCE + IoU only
4. All components with high structure weight
5. All components with high center penalty
"""

import subprocess
import sys
from pathlib import Path
import yaml
import shutil

# Base config
BASE_CONFIG = "src/tao2019_fd2nn/config/saliency_ecssd_f2mm.yaml"

# Test cases: (name, loss_mode, loss_weights_dict)
TEST_CASES = [
    ("mse_baseline", "mse", None),

    ("structured_default", "structured", {
        "bce": 1.0,
        "iou": 2.0,
        "structure": 1.0,
        "center_penalty": 0.1,
    }),

    ("bce_iou_only", "structured", {
        "bce": 1.0,
        "iou": 2.0,
        "structure": 0.0,
        "center_penalty": 0.0,
    }),

    ("high_structure", "structured", {
        "bce": 1.0,
        "iou": 1.0,
        "structure": 3.0,
        "center_penalty": 0.1,
    }),

    ("high_center_penalty", "structured", {
        "bce": 1.0,
        "iou": 2.0,
        "structure": 1.0,
        "center_penalty": 0.5,
    }),

    ("iou_dominant", "structured", {
        "bce": 0.5,
        "iou": 5.0,
        "structure": 0.5,
        "center_penalty": 0.1,
    }),
]


def create_config(base_path: Path, case_name: str, loss_mode: str, loss_weights: dict | None) -> Path:
    """Create a modified config for a test case."""

    # Load base config
    with open(base_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Modify experiment name
    cfg['experiment']['name'] = f"loss_test_{case_name}"

    # Reduce epochs for faster testing (can adjust)
    cfg['training']['epochs'] = 50

    # Set loss mode
    cfg['training']['loss_mode'] = loss_mode

    # Set loss weights if provided
    if loss_weights is not None:
        cfg['training']['loss_weights'] = loss_weights
    elif 'loss_weights' in cfg['training']:
        del cfg['training']['loss_weights']

    # Save to temp config
    temp_config_dir = Path("temp_configs")
    temp_config_dir.mkdir(exist_ok=True)

    temp_config_path = temp_config_dir / f"loss_test_{case_name}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return temp_config_path


def run_experiment(config_path: Path) -> tuple[bool, str]:
    """Run a single experiment and return success status."""

    cmd = [
        sys.executable,
        "-m", "tao2019_fd2nn.cli.train_saliency",
        "--config", str(config_path),
    ]

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
            cwd=Path(__file__).resolve().parents[1],  # tao2019_fourier_space_d2nn/
        )
        return True, "Success"
    except subprocess.CalledProcessError as e:
        return False, f"Failed with exit code {e.returncode}"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def extract_metrics(run_dir: Path) -> dict:
    """Extract key metrics from a run."""

    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return {}

    import json
    with open(metrics_file, 'r') as f:
        return json.load(f)


def main():
    print("="*80)
    print("F-D2NN Loss Function Comparison Test Suite")
    print("="*80)
    print(f"Testing {len(TEST_CASES)} different loss configurations")
    print()

    # Change to repo root
    repo_root = Path(__file__).resolve().parents[1]  # tao2019_fourier_space_d2nn/

    results = []

    for case_name, loss_mode, loss_weights in TEST_CASES:
        print(f"\n{'='*80}")
        print(f"TEST CASE: {case_name}")
        print(f"  Loss mode: {loss_mode}")
        if loss_weights:
            print(f"  Weights: {loss_weights}")
        print(f"{'='*80}")

        # Create config
        config_path = create_config(
            repo_root / BASE_CONFIG,
            case_name,
            loss_mode,
            loss_weights
        )

        # Run experiment
        success, message = run_experiment(config_path)

        # Record result
        results.append({
            'case_name': case_name,
            'loss_mode': loss_mode,
            'loss_weights': loss_weights,
            'success': success,
            'message': message,
        })

        print(f"\n{'='*80}")
        print(f"Result: {message}")
        print(f"{'='*80}\n")

        if not success:
            print(f"WARNING: {case_name} failed, continuing to next test...")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    runs_dir = repo_root / "runs"

    for result in results:
        case_name = result['case_name']
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"\n{status} {case_name}")

        if result['success']:
            # Find the most recent run directory for this case
            pattern = f"loss_test_{case_name}_*"
            matching_dirs = sorted(runs_dir.glob(pattern))

            if matching_dirs:
                run_dir = matching_dirs[-1]
                metrics = extract_metrics(run_dir)

                if metrics:
                    val_fmax_best = metrics.get('val_fmax_best', 'N/A')
                    best_epoch = metrics.get('best_epoch', 'N/A')
                    print(f"  Run dir: {run_dir.name}")
                    print(f"  Val F_max: {val_fmax_best:.4f}" if isinstance(val_fmax_best, float) else f"  Val F_max: {val_fmax_best}")
                    print(f"  Best epoch: {best_epoch}")
        else:
            print(f"  Error: {result['message']}")

    print("\n" + "="*80)
    print(f"Completed {sum(r['success'] for r in results)}/{len(results)} tests successfully")
    print("="*80)

    # Clean up temp configs
    temp_config_dir = Path("temp_configs")
    if temp_config_dir.exists():
        shutil.rmtree(temp_config_dir)


if __name__ == "__main__":
    main()
