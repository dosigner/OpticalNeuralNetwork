#!/usr/bin/env bash
# ============================================================================
# reproduce_experiments.sh — Reproduce kim2026 experiment checkpoints & data
#
# Binary files (.pt, .npy, .npz) are gitignored to save space.
# This script re-generates them from the committed source code and configs.
#
# Usage:
#   ./scripts/reproduce_experiments.sh [experiment_name] [--gpu GPU_ID] [--dry-run]
#
# Examples:
#   ./scripts/reproduce_experiments.sh                    # run ALL experiments
#   ./scripts/reproduce_experiments.sh focal_pib          # single experiment
#   ./scripts/reproduce_experiments.sh focal_pib --gpu 1  # specific GPU
#   ./scripts/reproduce_experiments.sh --dry-run          # show what would run
#   ./scripts/reproduce_experiments.sh list               # list available experiments
#
# Prerequisites:
#   1. conda activate d2nn
#   2. kim2026/data/ must exist (generate with: python -m kim2026.data.generate)
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KIM2026_DIR="$REPO_ROOT/kim2026"
PYTHON="${KIM2026_DIR}/../miniconda3/envs/d2nn/bin/python"
GPU_ID="${GPU_ID:-0}"
DRY_RUN=0
TARGET=""

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)       GPU_ID="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1; shift ;;
        list)        TARGET="list"; shift ;;
        *)           TARGET="$1"; shift ;;
    esac
done

# ── Experiment registry ──────────────────────────────────────────────────────
# Maps experiment name → Python module and output directory
declare -A EXPERIMENTS=(
    ["telescope_sweep"]="autoresearch.d2nn_sweep_telescope|0325-telescope-sweep-cn2-5e14-15cm"
    ["loss_sweep_prelens"]="autoresearch.d2nn_loss_strategy_sweep|0327-loss-sweep-prelens-pib-cn2-5e14"
    ["theorem_verify"]="autoresearch.visualize_optical_path|0327-theorem-verify-defocus-1layer"
    ["co_sweep_strong"]="autoresearch.d2nn_strong_turb_sweep|0328-co-sweep-strong-turb-cn2-5e14"
    ["paper_figures"]="autoresearch.visualize_optical_path|0329-paper-figures-static-d2nn"
    ["focal_pib"]="autoresearch.d2nn_focal_pib_sweep|0330-focal-pib-sweep-4loss-cn2-5e14"
)

# Ordered list for sequential execution
EXPERIMENT_ORDER=(
    "telescope_sweep"
    "loss_sweep_prelens"
    "theorem_verify"
    "co_sweep_strong"
    "paper_figures"
    "focal_pib"
)

# ── Functions ────────────────────────────────────────────────────────────────
print_header() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

list_experiments() {
    echo "Available experiments:"
    echo ""
    printf "  %-25s %-45s %s\n" "NAME" "MODULE" "OUTPUT DIR"
    printf "  %-25s %-45s %s\n" "----" "------" "----------"
    for name in "${EXPERIMENT_ORDER[@]}"; do
        IFS='|' read -r module outdir <<< "${EXPERIMENTS[$name]}"
        printf "  %-25s %-45s %s\n" "$name" "$module" "$outdir"
    done
    echo ""
    echo "Run a specific experiment:  $0 <name> [--gpu N]"
    echo "Run all experiments:        $0 [--gpu N]"
}

check_data() {
    local data_dir="$KIM2026_DIR/data/kim2026/1km_cn2_5e-14_tel15cm_n1024_br75/cache"
    if [[ ! -d "$data_dir" ]] || [[ $(find "$data_dir" -name "*.npz" 2>/dev/null | head -1) == "" ]]; then
        echo "ERROR: Training data not found at $data_dir"
        echo ""
        echo "Generate data first:"
        echo "  cd $KIM2026_DIR && PYTHONPATH=src $PYTHON -m kim2026.data.generate"
        exit 1
    fi
    echo "Data directory OK: $(find "$data_dir" -name "*.npz" | wc -l) files"
}

run_experiment() {
    local name="$1"
    IFS='|' read -r module outdir <<< "${EXPERIMENTS[$name]}"
    local run_dir="$KIM2026_DIR/autoresearch/runs/$outdir"

    print_header "Reproducing: $name"
    echo "  Module:  $module"
    echo "  Output:  $run_dir"
    echo "  GPU:     $GPU_ID"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [DRY RUN] Would execute:"
        echo "    cd $KIM2026_DIR && CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=src $PYTHON -m $module"
        return 0
    fi

    # Run experiment
    cd "$KIM2026_DIR"
    mkdir -p "$run_dir"
    CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH=src "$PYTHON" -m "$module" 2>&1 | tee "$run_dir/reproduce.log"

    # Verify outputs
    local n_pt=$(find "$run_dir" -name "*.pt" 2>/dev/null | wc -l)
    local n_npy=$(find "$run_dir" -name "*.npy" 2>/dev/null | wc -l)
    echo ""
    echo "  Completed: $n_pt checkpoints, $n_npy arrays generated"
}

# ── Main ─────────────────────────────────────────────────────────────────────
if [[ "$TARGET" == "list" ]]; then
    list_experiments
    exit 0
fi

echo "D2NN Experiment Reproducer"
echo "Repository: $REPO_ROOT"
echo "Python:     $PYTHON"
echo "GPU:        $GPU_ID"

# Verify prerequisites
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Activate conda env: conda activate d2nn"
    exit 1
fi

check_data

if [[ -n "$TARGET" ]]; then
    # Run single experiment
    if [[ -z "${EXPERIMENTS[$TARGET]+x}" ]]; then
        echo "ERROR: Unknown experiment '$TARGET'"
        echo "Run '$0 list' to see available experiments"
        exit 1
    fi
    run_experiment "$TARGET"
else
    # Run all experiments
    print_header "Running ALL experiments (${#EXPERIMENT_ORDER[@]} total)"
    for name in "${EXPERIMENT_ORDER[@]}"; do
        run_experiment "$name"
    done
    print_header "All experiments completed"
fi
