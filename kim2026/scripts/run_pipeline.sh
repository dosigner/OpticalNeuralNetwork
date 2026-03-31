#!/usr/bin/env bash
# ============================================================================
# run_pipeline.sh — Unified D2NN experiment pipeline
#
# Phase 1: Train       → d2nn_focal_pib_sweep (4 loss strategies)
# Phase 2: Evaluate    → eval + visualization (6+ figures)
# Phase 3: Analyze     → bucket radius sweep + paper figures
# Phase 4: Verify      → physics sanity checks (embedded in Phase 1/2)
#
# Usage:
#   ./scripts/run_pipeline.sh [--gpu N] [--config PATH] [--phase N] [--dry-run]
#
# Examples:
#   ./scripts/run_pipeline.sh                           # all phases, GPU 0
#   ./scripts/run_pipeline.sh --gpu 1                   # all phases, GPU 1
#   ./scripts/run_pipeline.sh --phase 2                 # eval + viz only
#   ./scripts/run_pipeline.sh --phase 3                 # analysis only
#   ./scripts/run_pipeline.sh --dry-run                 # show commands
#   ./scripts/run_pipeline.sh --config configs/my.yaml  # custom config
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KIM2026_DIR="$REPO_ROOT"
PYTHON="${KIM2026_DIR}/../miniconda3/envs/d2nn/bin/python"
GPU_ID="${GPU_ID:-0}"
CONFIG="autoresearch/configs/focal_pib_sweep.yaml"
DRY_RUN=0
PHASE=0  # 0 = all phases

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)      GPU_ID="$2"; shift 2 ;;
        --config)   CONFIG="$2"; shift 2 ;;
        --phase)    PHASE="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)
            echo "Usage: $0 [--gpu N] [--config PATH] [--phase N] [--dry-run]"
            echo ""
            echo "Phases:"
            echo "  1  Train (d2nn_focal_pib_sweep.py)"
            echo "  2  Evaluate + Visualize"
            echo "  3  Analyze (bucket radius sweep + paper figures)"
            echo "  0  All phases (default)"
            exit 0
            ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Functions ────────────────────────────────────────────────────────────────
header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "  $1"
    echo "╚══════════════════════════════════════════════════════════════╝"
}

run_cmd() {
    local desc="$1"
    local cmd="$2"

    echo "  → $desc"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "    [DRY RUN] $cmd"
        return 0
    fi

    cd "$KIM2026_DIR"
    eval "CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=src $cmd"
}

ensure_phase1_results() {
    local sweep_dir="$KIM2026_DIR/autoresearch/runs/d2nn_focal_pib_sweep"
    local results_path="$sweep_dir/focal_pib_only/results.json"
    if [[ ! -f "$results_path" ]]; then
        echo "ERROR: Phase 1 results not found at $results_path"
        echo "Run Phase 1 first."
        exit 1
    fi
}

check_prerequisites() {
    if [[ ! -x "$PYTHON" ]]; then
        echo "ERROR: Python not found at $PYTHON"
        exit 1
    fi

    local data_dir="$KIM2026_DIR/data/kim2026/1km_cn2_5e-14_tel15cm_n1024_br75/cache"
    if [[ ! -d "$data_dir" ]]; then
        echo "ERROR: Training data not found at $data_dir"
        echo "Generate: cd $KIM2026_DIR && PYTHONPATH=src $PYTHON -m kim2026.data.generate"
        exit 1
    fi
    echo "  Data: OK ($(find "$data_dir" -name "*.npz" | wc -l) files)"
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo "D2NN Experiment Pipeline"
echo "  GPU:    $GPU_ID"
echo "  Config: $CONFIG"
echo "  Phase:  ${PHASE:-all}"
echo "  Python: $PYTHON"

check_prerequisites

# Phase 1: Train
if [[ $PHASE -eq 0 || $PHASE -eq 1 ]]; then
    header "Phase 1: Training (4 loss strategies)"
    CONFIG_ARG=""
    if [[ -n "$CONFIG" ]]; then
        CONFIG_ARG=" --config \"$CONFIG\""
    fi
    run_cmd "Focal PIB sweep" \
        "$PYTHON -m autoresearch.d2nn_focal_pib_sweep$CONFIG_ARG"
fi

# Phase 2: Evaluate + Visualize
if [[ $PHASE -eq 0 || $PHASE -eq 2 ]]; then
    if [[ ! ($PHASE -eq 0 && $DRY_RUN -eq 1) ]]; then
        ensure_phase1_results
    fi
    header "Phase 2: Evaluation & Visualization"
    run_cmd "Quick eval (focal_pib_only)" \
        "$PYTHON scripts/eval_focal_pib_only.py"
    run_cmd "Visualization report (6 figures)" \
        "$PYTHON scripts/visualize_focal_pib_report.py"
fi

# Phase 3: Comprehensive Analysis
if [[ $PHASE -eq 0 || $PHASE -eq 3 ]]; then
    if [[ ! ($PHASE -eq 0 && $DRY_RUN -eq 1) ]]; then
        ensure_phase1_results
    fi
    header "Phase 3: Comprehensive Analysis"
    run_cmd "Bucket radius sweep (4 strategies × 4 radii)" \
        "$PYTHON scripts/eval_bucket_radius_sweep.py"
    run_cmd "Paper figures (theorem verification)" \
        "$PYTHON scripts/generate_focal_paper_figures.py"
fi

header "Pipeline Complete"
echo "  Results: $KIM2026_DIR/autoresearch/runs/d2nn_focal_pib_sweep/"
echo ""
