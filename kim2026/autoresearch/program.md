# FD2NN Autoresearch

Autonomous experimentation for Fourier-space Diffractive Deep Neural Network (FD2NN) beam cleanup optimization, adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar26`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current branch.
3. **Read the in-scope files**: Read these for full context:
   - `autoresearch/program.md` — this file, your instructions.
   - `autoresearch/experiment.py` — **the only file you modify**. Training script with hyperparameters.
   - `src/kim2026/models/fd2nn.py` — FD2NN model architecture (read-only).
   - `src/kim2026/training/losses.py` — available loss functions (read-only).
   - `src/kim2026/training/metrics.py` — evaluation metrics (read-only).
   - `src/kim2026/optics/angular_spectrum.py` — wave propagation (read-only).
   - `src/kim2026/optics/lens_2f.py` — dual-2f lens transforms (read-only).
4. **Verify data exists**: Check that `data/kim2026/1km_cn2e-14_w2m_n1024_dx2mm/cache/` contains `.npz` files and `split_manifest.json` exists.
5. **Initialize results.tsv**: Create with header row only. Baseline will be recorded after first run.
6. **Confirm and go**: Confirm setup looks good, then kick off.

## The Problem

FD2NN is a passive optical device (metasurface stack) that cleans up laser beams distorted by atmospheric turbulence in a Free-Space Optical (FSO) communication link. The system uses a dual-2f Fourier relay to place trainable phase masks in the Fourier plane. Each mask applies `field * exp(j*phi)` — phase-only modulation with unit transmittance.

**Current best complex_overlap: 0.270** (5 layers, 1mm spacing, symmetric_tanh, co:1.0+amp:0.5)
**Baseline (no correction): ~0.098**

The gap between 0.270 and 1.0 is the opportunity.

## Experimentation

Each experiment trains one FD2NN model and evaluates it. Launch with:
```bash
cd /root/dj/D2NN/kim2026 && python -m autoresearch.experiment > autoresearch/run.log 2>&1
```

### What you CAN modify

`autoresearch/experiment.py` is the only file you edit. Everything in the `TUNABLE HYPERPARAMETERS` section is fair game:

| Category | Parameters | Notes |
|----------|-----------|-------|
| Architecture | `NUM_LAYERS`, `LAYER_SPACING_M`, `PHASE_INIT_SCALE` | See guard rails below |
| Phase | `PHASE_CONSTRAINT`, `PHASE_MAX` | Unconstrained is allowed (see below) |
| Optimizer | `LR`, `OPTIMIZER`, `WEIGHT_DECAY`, `GRAD_CLIP_NORM` | Adam or AdamW |
| Scheduler | `SCHEDULER`, `COSINE_T0`, `COSINE_ETA_MIN` | none/cosine/cosine_warm_restarts |
| Training | `EPOCHS`, `BATCH_SIZE` | More epochs likely helps |
| Loss | `LOSS_MODE`, `LOSS_WEIGHTS`, `ROI_*` | composite or roi_complex |
| Regularization | `PHASOR_SMOOTHNESS_WEIGHT` | Recommended > 0 for unconstrained |

You may also modify training logic below the hyperparameters section (add scheduling tricks, custom loss combinations, etc.) as long as you use the existing `kim2026` modules.

### What you CANNOT modify

- Any file under `src/kim2026/` — these implement physical laws (Maxwell's equations, Fourier optics, angular spectrum propagation). They are ground truth.
- The `IMMUTABLE CONSTANTS` section at the top of `experiment.py` — these must match the training data.
- Install new packages or add dependencies.
- The evaluation logic (`evaluate()` function) — this is the ground truth metric.

### What you MUST NOT do

These modifications look tempting to an ML optimizer but violate physics:

| Forbidden | Why |
|-----------|-----|
| Add amplitude modulation (`field * a * exp(j*phi)` where `a != 1`) | Passive metasurface cannot absorb or amplify |
| Add residual connections (`out = layer(x) + x`) | Creates `x(1+exp(jφ))` = amplitude modulation |
| BatchNorm/LayerNorm on complex fields | Destroys energy conservation |
| Weight tying across layers | Creates etalon, useless for correction |
| Set `DUAL_2F_F1_M != DUAL_2F_F2_M` | Output pixel pitch mismatches target — metrics become physically meaningless (code silently discards dx_out) |
| Set `DUAL_2F_NA > 0.3` | Paraxial approximation breaks down |
| Set `DUAL_2F_APPLY_SCALING = True` | Creates artificial gain with ortho FFT |

## Guard Rails

### Physics constraints (hard limits)

| Parameter | Constraint | Reason |
|-----------|-----------|--------|
| `DUAL_2F_F1_M` | Must equal `DUAL_2F_F2_M` | No magnification mismatch |
| `DUAL_2F_NA1/NA2` | <= 0.3 | Paraxial validity for FFT lens model |
| `LAYER_SPACING_M` | <= 0.05 (50mm) | Angular spectrum aliasing limit |
| `DUAL_2F_APPLY_SCALING` | Must be `False` | Energy conservation with ortho FFT |

### Unconstrained phase (allowed with care)

**`PHASE_CONSTRAINT = "unconstrained"` is explicitly allowed.** Here is why:

`exp(jφ)` is 2π-periodic. `exp(j·100.3)` ≡ `exp(j·(100.3 mod 2π))`. The physics is identical.

Benefits of unconstrained over tanh:
- **tanh gradient at saturation**: at raw=3, `1-tanh²(3) = 0.01` → 99% gradient vanishing
- **unconstrained gradient**: always exactly 1.0 for `|d/draw exp(j·raw)|`

The trained phases are saved as `wrapped_phase()` → `[0, 2π)` for fabrication view.

**Required**: When using unconstrained, set `PHASOR_SMOOTHNESS_WEIGHT > 0` (recommended 0.01) to prevent sub-pixel phase discontinuities that would be unfabricatable.

The phasor smoothness loss uses the circular-manifold distance:
```
L_smooth = mean(|exp(jφ_i) - exp(jφ_j)|²) = mean(2(1 - cos(Δφ)))
```
This is 2π-periodic: penalizes local phase jumps but ignores full-wrap boundaries.

### Post-experiment sanity checks (automatic)

The experiment script automatically checks and warns:
- **Throughput** ∈ [0.95, 1.05]: passive device cannot create/destroy energy
- **Strehl** <= 1.05: passive device cannot amplify on-axis intensity

If either warning fires, the result is **physically invalid**. Discard and investigate.

## Goal & Primary Metric

**Maximize `complex_overlap` on the test set.** Higher is better.

```
complex_overlap = |<pred, target>| / (||pred|| * ||target||)
```

This is the coherent mode-matching efficiency — exactly the quantity that determines link budget improvement in a coherent FSO receiver. It captures both amplitude and phase quality in a single number.

- Baseline (no correction): ~0.098
- Current record: 0.270
- Perfect correction: 1.000

Other metrics (strehl, phase_rmse, intensity_overlap) are logged for diagnostics but the keep/discard decision uses `complex_overlap` alone.

## Output Format

The script prints a parseable summary:
```
---
complex_overlap:   0.270000
intensity_overlap: 0.450000
strehl:            0.150000
phase_rmse_rad:    1.230000
amplitude_rmse:    0.340000
encircled_energy:  0.560000
throughput:        0.998000
training_seconds:  180.5
peak_vram_mb:      4096.0
num_params:        1310720
num_layers:        5
baseline_co:       0.098000
```

Extract the key metric:
```bash
grep "^complex_overlap:" autoresearch/run.log
```

## Logging Results

Log every experiment to `autoresearch/results.tsv` (tab-separated).

Header and columns:
```
commit	complex_overlap	throughput	peak_vram_mb	status	description
```

1. git commit hash (short, 7 chars)
2. complex_overlap achieved (0.000000 for crashes)
3. throughput (0.000 for crashes)
4. peak VRAM in MB (0.0 for crashes)
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:
```
commit	complex_overlap	throughput	peak_vram_mb	status	description
a1b2c3d	0.270000	0.998	4096.0	keep	baseline (symmetric_tanh, co:1+amp:0.5)
b2c3d4e	0.285000	0.999	4100.0	keep	unconstrained phase + phasor smoothness 0.01
c3d4e5f	0.260000	0.998	4096.0	discard	cosine schedule lr=1e-3 (overshot)
d4e5f6g	0.000000	0.000	0.0	crash	100 layers OOM
```

## The Experiment Loop

LOOP FOREVER:

1. Look at git state and results.tsv to understand what has been tried
2. Form a hypothesis about what might improve complex_overlap
3. Edit `autoresearch/experiment.py` with the change
4. `git commit -m "experiment: <description>"`
5. Run: `cd /root/dj/D2NN/kim2026 && python -m autoresearch.experiment > autoresearch/run.log 2>&1`
6. Read results: `grep "^complex_overlap:\|^throughput:\|^peak_vram_mb:" autoresearch/run.log`
7. If grep is empty → crash. Run `tail -n 50 autoresearch/run.log` for traceback
8. Check sanity: throughput in [0.95, 1.05], no WARN lines
9. Record in results.tsv
10. If complex_overlap improved → **keep** (advance branch)
11. If complex_overlap equal or worse → **discard** (`git reset --hard HEAD~1`)

### Recommended experiment order

Based on prior analysis, try these roughly in order:

1. **Baseline**: run experiment.py as-is to establish the starting point
2. **Unconstrained + phasor smoothness**: switch to unconstrained phase, add smoothness reg
3. **LR schedule**: try cosine_warm_restarts with T0=20, lr=1e-3
4. **Extended epochs**: increase to 100-200 epochs (current 30 is likely undertrained)
5. **Loss weight sweep**: vary complex_overlap vs amplitude_mse ratio
6. **ROI complex loss**: switch LOSS_MODE to roi_complex
7. **Phase init scale**: try 0.01, 0.3, 0.5
8. **AdamW + weight decay**: try 1e-4 to 1e-3
9. **Gradient clipping**: try 1.0 or 5.0
10. **Layer count**: try 3, 7

### Timeout

Most experiments should complete within 10-30 minutes depending on EPOCHS. If a run exceeds 60 minutes, kill it (`Ctrl+C` or timeout) and treat as failure.

### NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human. The human may be asleep. You are autonomous. If you run out of ideas:
- Re-read the physics modules for new angles
- Combine previous near-misses
- Try more radical loss formulations
- Vary multiple parameters simultaneously

The loop runs until the human interrupts you.
