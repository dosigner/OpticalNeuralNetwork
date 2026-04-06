# D2NN Experiment Chronology Report — Codex Draft Plan

## Goal
Write a comprehensive 300+ page LaTeX report documenting all D2NN/FD2NN experiments from 2026-03-25 to 2026-04-06. The report should include all 261 PNG figures, 37 result sets, and mathematical proofs. Output is a compiled PDF via XeLaTeX.

## Output Format
- **LaTeX** with XeLaTeX compilation (for Korean font support)
- **Korean text** with English technical terms
- **Font**: NanumSquare (main), Consolas (mono). Available via `fc-list :lang=ko`
- **Paper style**: Academic report format (report class, chapters, numbered equations, figure captions)
- **Compile command**: `cd /root/dj/D2NN/kim2026/docs/experiment_report && xelatex -interaction=nonstopmode main.tex`

## File Structure
```
kim2026/docs/experiment_report/
├── main.tex              # Already exists — master document with \input{} for chapters
├── ch01_introduction.tex
├── ch02_beam_reducer_sweep.tex
├── ch03_optics_crisis.tex
├── ch04_static_limit.tex
├── ch05_focal_pib_bug.tex
├── ch06_data_generation.tex
├── ch07_focal_pib_sweep.tex
├── ch08_tp_preserving_loss.tex
├── ch09_loss_optimization.tex
├── ch10_distance_sweep.tex
├── ch11_fd2nn_4f.tex
├── ch12_fd2nn_fundamental_limit.tex
├── ch13_conclusions.tex
├── appendix_configs.tex
└── appendix_full_results.tex
```

## main.tex Setup (already created)
- Uses `\graphicspath{{../../autoresearch/runs/}{../fd2nn_sweep_figures/}}` so figure paths are relative to runs/
- Custom tcolorbox environments: `\begin{keyinsight}`, `\begin{discovery}`, `\begin{breakthrough}`
- Colors defined: navy, accent, codegreen, codered, codegray

## Chapter-by-Chapter Content Specification

### Chapter 1: Introduction (ch01_introduction.tex)
**~20 pages**

Sections:
1. 자유공간 광통신 (FSO) 배경
2. 대기 난류 모델
   - Kolmogorov PSD: $\Phi(\kappa) = 0.023\, r_0^{-5/3}\, \kappa^{-11/3}$
   - von Kármán: $\Phi(\kappa) = 0.023\, r_0^{-5/3}\, (\kappa^2 + \kappa_0^2)^{-11/6}$
   - Fried parameter $r_0$, $D/r_0$ ratio, Rytov variance
3. D2NN (Diffractive Deep Neural Network) 개념
   - Phase-only modulation: $t(x,y) = \exp(j\phi(x,y))$
   - Angular Spectrum Method (ASM) propagation
   - Multi-layer diffraction
4. 연구 목표: static passive 광학 소자로 난류 보상이 가능한가?
5. 보고서 구성 개요 (13 chapters + appendix)

No figures needed. Pure text + equations.

---

### Chapter 2: D2NN Beam-Reducer Sweep (ch02_beam_reducer_sweep.tex)
**~20 pages. Date: 2026-03-25**

Motivation: mm → μm 스케일 전환. 제작 가능한 pitch에서 D2NN 성능 평가.

Key figures (paths relative to graphicspath):
- `0325-telescope-sweep-cn2-5e14-15cm/baseline_co/phase2_fig1_input.png`
- `0325-telescope-sweep-cn2-5e14-15cm/baseline_co/phase2_fig2_d2nn_output.png`
- `0325-telescope-sweep-cn2-5e14-15cm/baseline_co/phase2_fig3_detector.png`
- `0325-telescope-sweep-cn2-5e14-15cm/baseline_co/phase2_fig4_masks.png`
- `0325-telescope-sweep-cn2-5e14-15cm/baseline_co/phase2_fig5_training_curves.png`
- `0325-telescope-sweep-cn2-5e14-15cm/baseline_co/phase2_fig6_statistics.png`

Results table:
| Config | layer_spacing | detector_dist | overlap | Strehl |
|--------|:---:|:---:|:---:|:---:|
| ls10_dd10 | 10mm | 10mm | 0.535 | — |
| ls50_dd50 | 50mm | 50mm | 0.644 | — |
| pitch=5μm | 50mm | 50mm | 0.660 | >2.0 |

Conclusion: plain D2NN plateau at overlap=0.660 → FD2NN 탐색 동기.

---

### Chapter 3: Optics Crisis — f=1mm (ch03_optics_crisis.tex)
**~25 pages. Date: 2026-03-26**

Motivation: FD2NN dual-2f 시스템의 물리적 파라미터 오류 발견.

Key equations:
- Fourier plane pitch: $\text{dx}_f = \frac{\lambda f}{N \cdot \text{dx}}$
- NA condition: $\text{NA} \approx \frac{W}{2f}$

Comparison table:
| Parameter | f=1mm | f=10mm | f=25mm |
|-----------|:---:|:---:|:---:|
| dx_fourier | 1.5μm | 7.6μm | 37.8μm |
| NA | 0.16 | 0.102 | 0.041 |
| Throughput (untrained) | 0.02 | — | 0.76 |
| Fabrication | impossible | e-beam | photolith |

Impact: sweeps 01-05 (f=1mm) 결과 전부 무효. f=25mm으로 재설계.

No PNG figures (this was a parametric analysis, not a training run). Use equation-heavy analysis + tables.

---

### Chapter 4: Static D2NN 근본적 한계 (ch04_static_limit.tex)
**~30 pages. Date: 2026-03-27**

Motivation: static mask가 random turbulence를 보정할 수 있는지 근본적 질문.

Key figure:
- `0327-theorem-verify-defocus-1layer/deterministic_verification.png`

4-loss sweep results:
| Loss | CO | WF RMS (nm) | PIB | Throughput |
|------|:---:|:---:|:---:|:---:|
| PIB-only | 0.016 | 448 | 83.4% | — |
| Strehl-only | 0.015 | 461 | — | — |
| Intensity overlap | 0.178 | 459 | — | — |
| CO+PIB hybrid | 0.294 | 452 | — | — |
| **Baseline** | **0.304** | **460** | — | — |

Mathematical proof (full derivation required):
1. D2NN as fixed unitary operator $H$
2. Optimization: $\min_H \mathbb{E}[|H \hat{u}_{\text{turb}} - \hat{u}_{\text{vac}}|^2]$
3. Wiener filter optimal: $H_{\text{opt}}(\kappa) = \frac{\langle \hat{u}_{\text{vac}} \hat{u}_{\text{turb}}^* \rangle}{\langle |\hat{u}_{\text{turb}}|^2 \rangle}$
4. Random phase: $\theta(\kappa) \sim \text{Uniform}[0, 2\pi]$
5. $\langle e^{-j\theta} \rangle = 0$ → $H_{\text{opt}} = 0$

Theorem verification: deterministic defocus CAN be corrected (fig from 0327 run).

Conclusion: D2NN = mode converter, NOT wavefront corrector.

---

### Chapter 5: PIB Metric Bug (ch05_focal_pib_bug.tex)
**~15 pages. Date: 2026-03-27**

Motivation: PIB 최적화 모델이 실제로는 성능 악화.

Key equations:
- $\text{RP}(r_b) = \text{PIB}(r_b) \times \text{TP} \times E_{\text{in}}$
- PIB = bucket/total → reducing total increases PIB but decreases RP

Figures from `0330-focal-pib-sweep-4loss-cn2-5e14/focal_pib_only/`:
- `fig1_focal_plane_vacuum_vs_turbulent_vs_d2nn.png`
- `fig2_pib_bar_chart_5um_10um_25um_50um.png`
- `cheatsheet_d2nn_mode_conversion.png`

Resolution: metric plane separation principle.
- Coherent metrics (CO, WF RMS) → D2NN output plane
- Incoherent metrics (PIB, Strehl) → focal plane after lens

---

### Chapter 6: Production Data Generation (ch06_data_generation.tex)
**~15 pages. Date: 2026-04-01**

Figures from `0401-datagen-dn100um-lanczos50/`:
- `dataset_visualization_5samples.png`
- `dataset_1d_crosssection_5samples.png`
- `phase1_architecture.png`

Grid spacing comparison table:
| Δn | N_screen | screens | time/real | total (5000) | Nyquist |
|----|:---:|:---:|:---:|:---:|:---:|
| 50μm | 8192 | 77 | 71s | 98h | 3.0× |
| **100μm** | **4096** | **39** | **3.2s** | **4.4h** | **1.5×** |
| 150μm | 2048 | 18 | 1.5s | 2.1h | 0.98× |

Final dataset: 4000/500/500 split, vacuum PIB=94.2%, WFE=26nm.

---

### Chapter 7: Focal PIB Sweep — 모든 Loss 실패 (ch07_focal_pib_sweep.tex)
**~30 pages. Date: 2026-04-01**

Run: `0401-focal-pib-sweep-clean-4loss-cn2-5e14/`

Root-level figures:
- `0401-focal-pib-sweep-clean-4loss-cn2-5e14/figA_all_strategies_pib_comparison.png`
- `0401-focal-pib-sweep-clean-4loss-cn2-5e14/figB_training_curves.png`
- `0401-focal-pib-sweep-clean-4loss-cn2-5e14/received_power_analysis.png`
- `0401-focal-pib-sweep-clean-4loss-cn2-5e14/throughput_analysis.png`

Per-strategy figures (4 strategies × ~6 figures each):
- `focal_pib_only/fig1_focal_plane_vacuum_vs_turbulent_vs_d2nn.png`
- `focal_pib_only/fig2_pib_bar_chart_5um_10um_25um_50um.png`
- `focal_pib_only/fig3_d2nn_output_plane_irradiance_phase_residual.png`
- `focal_pib_only/fig4_wavefront_rms_distribution_50samples.png`
- Same structure for: focal_strehl_only/, focal_intensity_overlap/, focal_co_pib_hybrid/

Results:
| Strategy | PIB@10μm | TP | RP vs Turb |
|----------|:---:|:---:|:---:|
| focal_pib_only | 90.1% | 50.2% | **-41%** |
| focal_strehl_only | 81.8% | 57.5% | **-40%** |
| focal_intensity_overlap | 25.7% | 8.2% | **-97%** |
| focal_co_pib_hybrid | 95.8% | 1.7% | **-98%** |

Critical discovery: ALL conventional losses destroy throughput → RP worse than baseline.

---

### Chapter 8: TP-Preserving Loss Breakthrough (ch08_tp_preserving_loss.tex)
**~25 pages. Date: 2026-04-02**

Run: `0402-focal-new-losses-pitchrescale-3strat-cn2-5e14/`

Root figures:
- `0402-focal-new-losses-pitchrescale-3strat-cn2-5e14/received_power_analysis.png`
- `0402-focal-new-losses-pitchrescale-3strat-cn2-5e14/absolute_bucket_power_histogram.png`

Per-strategy figures (3 strategies).

Results:
| Strategy | PIB | TP | RP vs Turb |
|----------|:---:|:---:|:---:|
| Absolute Bucket | 80.0% | 98.6% | **+5.9%** |
| TP-Penalized PIB | 80.2% | 98.4% | **+5.8%** |
| Multiplane | 83.1% | 39.8% | -55% |

Loss design principle: TP must be INTRINSIC to loss, not post-hoc penalty.

---

### Chapter 9: Loss Optimization (ch09_loss_optimization.tex)
**~30 pages. Date: 2026-04-03**

Run: `0403-combined-6strat-pitchrescale-cn2-5e14/`

Root figures:
- `0403-combined-6strat-pitchrescale-cn2-5e14/cross_strategy_received_power.png`
- `0403-combined-6strat-pitchrescale-cn2-5e14/encircled_energy_per_strategy.png`

6 strategies compared. Winner: `focal_raw_received_power`
- $L = -\log\left(\frac{E_{\text{bucket}}}{E_{\text{input}}} + \epsilon\right)$
- PIB=80.8%, TP=98.6%, RP **+6.4%**

Per-strategy figures (6 × ~6 figs each).

Include gradient analysis of why log-form is optimal.

---

### Chapter 10: Distance Sweep (ch10_distance_sweep.tex)
**~35 pages. Date: 2026-04-05**

Run: `0405-distance-sweep-rawrp-f6p5mm/`

Root figure:
- `0405-distance-sweep-rawrp-f6p5mm/distance_sweep_summary_6panel.png`

Per-distance figures (5 distances × focal_raw_received_power/):
- L100m, L500m, L1000m, L2000m, L3000m
- Each with: fig1_focal_plane, fig2_pib_bar, fig3_d2nn_output, fig4_wavefront_rms, etc.

Results (key table):
| Distance | D/r₀ | Turb Bucket | D2NN Bucket | Improvement | Regime |
|----------|:---:|:---:|:---:|:---:|---------|
| 100m | 1.26 | 382 | 728 | +91% | Weak |
| 500m | 3.31 | 500 | 531 | +6.2% | Moderate |
| 1km | 5.02 | 166 | 181 | +9.0% | Moderate |
| 2km | 7.61 | 9.3 | 16.4 | +77% | Strong |
| 3km | 9.70 | 0.64 | 2.08 | +223% | Strong |

Three-regime analysis with physical mechanisms for each.

---

### Chapter 11: FD2NN 4f Multi-layer (ch11_fd2nn_4f.tex)
**~30 pages. Date: 2026-04-05~06**

Run: `0405-fd2nn-4f-sweep-pitchrescale/`

Figures (from `../fd2nn_sweep_figures/`):
- `fig1_pib_bar.png`
- `fig2_heatmap.png`
- `fig3_loss_curves.png`
- `fig4_tp_co.png`
- `fig5_regime.png`
- `fig6_phase_masks.png`

Architecture:
```
Input → Lens1(f) → Mask1 → ASM(z) → ... → Mask5 → Lens2(f) → Focal Lens → PIB
```

Results:
| Config | fPIB@10μm | Baseline | Δ |
|--------|:---:|:---:|:---:|
| f10mm_z1mm | 0.377 | 0.575 | -0.198 |
| f10mm_z3mm | 0.290 | 0.575 | -0.285 |
| f10mm_z5mm | 0.251 | 0.575 | -0.324 |
| f25mm_z1mm | **0.586** | 0.575 | **+0.010** |
| f25mm_z3mm | 0.430 | 0.575 | -0.145 |
| f25mm_z5mm | 0.419 | 0.575 | -0.156 |

Physics: Rayleigh range, diffraction mixing, f-z dilemma.

---

### Chapter 12: FD2NN Fundamental Limit Proof (ch12_fd2nn_fundamental_limit.tex)
**~25 pages**

Full mathematical proof:
1. PSD describes amplitude statistics, not phase
2. Static Fourier mask: $H(\kappa) = e^{j\Delta\phi(\kappa)}$
3. Needed: $H_{\text{ideal}}(\kappa) = e^{-j\theta_{\text{turb}}(\kappa)}$ (per-realization)
4. Wiener filter derivation: $H_{\text{opt}} \to 0$
5. f-z trade-off analysis
6. Why spatial D2NN works better (mode conversion vs phase correction)

Comparison: Spatial D2NN PIB +0.240 vs FD2NN PIB +0.010.

---

### Chapter 13: Conclusions (ch13_conclusions.tex)
**~10 pages**

Summary of all findings. Future directions: adaptive D2NN, MPLC comparison.

---

### Appendix A: Configuration Files (appendix_configs.tex)
**~10 pages**

All YAML configs used in experiments.

### Appendix B: Full Numerical Results (appendix_full_results.tex)
**~20 pages**

All results.json data in tabular form for every experiment.

---

## Compilation Instructions

```bash
cd /root/dj/D2NN/kim2026/docs/experiment_report
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex  # twice for TOC/references
```

## Key Constraints
1. All figure paths use `\graphicspath` from main.tex — no absolute paths in chapters
2. Korean text requires XeLaTeX (not pdfLaTeX)
3. Figures exist at `../../autoresearch/runs/` relative to the .tex files
4. Use `[H]` float placement for figures to keep them near text
5. Every experiment section follows: 동기 → 설계 → 결과 → 분석 → 교훈 → 다음 실험 연결
6. Use tcolorbox environments for key findings: `\begin{keyinsight}`, `\begin{discovery}`, `\begin{breakthrough}`
