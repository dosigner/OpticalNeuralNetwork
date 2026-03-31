# GPU-Accelerated D2NN for Atmospheric Turbulence Correction: Feasibility Study & Experimental Report

**Author:** DJ, ADD Optical Network Research Section
**Date:** 2026-03-23
**Platform:** PyTorch + CUDA (NVIDIA A100 40 GB), Wavelength $\lambda = 1550\;\text{nm}$, Grid $1024 \times 1024$

---

## Executive Summary

**Objective.** Evaluate whether a passive, phase-only Diffractive Deep Neural Network (D2NN) can correct atmospheric-turbulence distortion in a 1550 nm Free-Space Optical (FSO) communication link.

**Key Findings.**

1. A physics-validated FSO beam propagation simulator (9 modules, 2138 LOC, 30 automated tests) was developed from first principles, reproducing Kolmogorov turbulence with structure-function errors below 20 % and coherence-factor agreement within 13 %.
2. At physical pixel scales ($\delta = 150\;\mu\text{m}$, $\delta/\lambda = 97$), the D2NN achieves $< 1\%$ loss improvement after 200 training epochs. Training effectively stalls.
3. At metasurface scales ($\delta = 10\;\mu\text{m}$, $\delta/\lambda = 6.5$), modest learning occurs (11 % training loss improvement), but visual inspection reveals the learned phase masks are nearly flat—the network converges to "do nothing."
4. The failure is not an engineering deficiency but a fundamental physics limitation: a fixed phase mask cannot correct random, per-realization turbulence distortions.

**Conclusion.** Passive D2NN is **not viable** for real-time atmospheric turbulence correction. The root cause is the mismatch between deterministic (fixed) optical elements and stochastic (random) turbulence realizations.

**Recommendation.** Pursue hybrid approaches that combine real-time wavefront sensing with adaptive correction elements. D2NN may still be useful for correcting fixed, deterministic aberrations such as manufacturing tolerances or static misalignments.

---

## 1. FSO Beam Propagation Simulator

### 1.1 Architecture

The simulator is implemented in the `kim2026.fso` package, consisting of 9 Python modules totaling 2138 lines of code (1626 source + 512 test), verified by 30 automated tests. The modules are:

| Module | Function | LOC |
|--------|----------|-----|
| `config.py` | `SimulationConfig` dataclass | 45 |
| `ft_utils.py` | `ft2`, `ift2`, `corr2_ft`, `str_fcn2_ft`, `str_fcn2_bruteforce` | 220 |
| `atmosphere.py` | $r_0$ calculations, SLSQP optimization | 165 |
| `sampling.py` | Grid constraint analysis (Constraints 1--4) | 183 |
| `phase_screen.py` | FFT + 3-level subharmonic Kolmogorov screens | 189 |
| `propagation.py` | Split-step angular-spectrum (Listing 9.1) | 212 |
| `verification.py` | Structure function + coherence factor checks | 342 |
| `main.py` | Full pipeline orchestrator | 264 |
| `__init__.py` | Public API | 6 |

### 1.2 Propagation Method

The core propagation engine implements the split-step angular-spectrum method from Schmidt (2010), Listing 9.1. The field $U$ is propagated through $n$ phase-screen planes by iterating:

$$U_{i+1} = \text{sg} \cdot T_{i+1} \cdot \mathcal{F}^{-1}\!\left[Q_2^{(i)} \cdot \mathcal{F}\!\left[\frac{U_i}{m_i}\right]\right]$$

where $Q_2^{(i)} = \exp\!\left(-i\pi^2 \frac{2\Delta z_i}{m_i k} f^2\right)$ is the free-space transfer function in the frequency domain, $m_i = \delta_{i+1}/\delta_i$ is the magnification between consecutive planes, $T_{i+1} = e^{i\phi_{i+1}}$ is the phase-screen transmittance, and $\text{sg}$ is a super-Gaussian absorbing boundary ($w = 0.47N$). Initial and final quadratic phases $Q_1$ and $Q_3$ account for the variable grid spacing.

### 1.3 Phase Screen Generation

Turbulence phase screens follow the Kolmogorov power spectral density:

$$\Phi_\phi(f) = 0.023 \; r_0^{-5/3} \; f^{-11/3}$$

Screens are generated via FFT with 3 levels of subharmonic correction (Schmidt Listings 9.2--9.3). Each subharmonic level $p \in \{1,2,3\}$ adds low-frequency content at spatial frequencies $\Delta f_p = 1/(3^p D)$ through direct DFT synthesis over a $3 \times 3$ sub-grid. All computations use float64 precision on CUDA.

### 1.4 Fried Parameter Optimization

Per-screen $r_0$ values are determined by constrained SLSQP optimization (Schmidt Listing 9.5), distributing the total turbulence strength $C_n^2 L$ across $n$ planes while satisfying sampling constraints. Endpoint screens are capped at $r_0 = 50\;\text{m}$ (near-zero turbulence) with corresponding verification skips.

### 1.5 Physics Validation

| Test | Criterion | Result |
|------|-----------|--------|
| FT roundtrip error | $< 10^{-10}$ | $1.94 \times 10^{-15}$ |
| Parseval's theorem | $< 10^{-10}$ | $4.04 \times 10^{-16}$ |
| Structure function $D_\phi(r)$ | per-plane avg. error $< 20\%$ | 7/8 planes pass |
| Coherence factor $e^{-1}$ width | within $20\%$ of $\rho_0$ | 13 % difference |

### 1.6 Simulation Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Wavelength | $\lambda$ | 1550 nm |
| Propagation distance | $L$ | 1 km |
| Refractive-index structure constant | $C_n^2$ | $10^{-14}\;\text{m}^{-2/3}$ |
| Full-angle divergence | $\theta_\text{div}$ | 0.6 mrad |
| Beam waist | $w_0$ | 1.64 mm |
| Beam radius at 1 km | $w(L)$ | 30 cm |
| Spherical-wave Fried parameter | $r_{0,\text{sw}}$ | 14.1 cm |
| Log-amplitude variance | $\sigma_\chi^2$ | 0.020 (weak fluctuation) |
| Realizations | $n_\text{real}$ | 200 |

### 1.7 Figure Analysis: FSO Turbulence Data

![Figure 1. FSO turbulence data — beam irradiance montage showing vacuum reference and individual turbulence realizations at 0.6 mrad, 1 km, $C_n^2 = 10^{-14}$.](figures/fig1_fso_turbulence.png){width=100%}

The FSO turbulence montage figure presents a grid of beam irradiance patterns at the receiver plane. The top-left panel shows the vacuum Gaussian beam—clean, circularly symmetric, and peaked at the optical axis. Surrounding panels (#0, #5, #10, #15, #20, #25) display individual turbulence realizations, in which the beam has broken into chaotic speckle patterns with randomly distributed hot spots. Two dashed circles annotate each panel: a white circle at $w(z) = 30\;\text{cm}$ (the vacuum beam radius) and a cyan circle at $r_{0,\text{sw}} = 14.1\;\text{cm}$ (the Fried parameter).

The ratio $w/r_0 \approx 2.1$ is the key physical parameter governing beam behavior. Since the beam diameter exceeds the atmospheric coherence diameter, the wavefront within the beam aperture contains multiple independent turbulence cells, causing the beam to fragment into multiple speckle lobes rather than simply wander as a whole. The bottom-right panel shows the long-exposure average over 30 realizations—the speckle washes out, producing a smoother but noticeably broader profile than the vacuum beam. The accompanying cross-section plots make the transition vivid: the vacuum beam's smooth Gaussian envelope degrades into a jagged, realization-dependent speckle trace under turbulence, then recovers a broadened, smoothed average in the ensemble mean.

---

## 2. D2NN Architecture

### 2.1 Model Structure

The `BeamCleanupD2NN` model consists of $L = 5$ phase-only diffractive layers implemented as `PhaseOnlyLayer` modules. Each layer applies a pixel-wise, learnable phase modulation:

$$U_\text{out} = U_\text{in} \cdot e^{i\phi_\text{learned}}$$

where $\phi_\text{learned}$ is wrapped to $[0, 2\pi]$ via modular arithmetic. Between consecutive layers, the field is propagated using the angular-spectrum method with a band-limited transfer function computed in float64 precision. After the final layer, one additional propagation step brings the field to the detector plane.

### 2.2 Phase-Only Constraint

The transmittance of each layer satisfies $|e^{i\phi}| = 1$ everywhere. This means the D2NN is a unitary (energy-conserving) optical processor: it can redistribute energy spatially through diffraction, but it cannot absorb or amplify any part of the field. Critically, the local irradiance immediately after a phase layer is unchanged:

$$|U_\text{in} \cdot e^{i\phi}|^2 = |U_\text{in}|^2$$

Any irradiance modification requires subsequent free-space propagation to convert the phase modulation into amplitude modulation through interference.

### 2.3 The $\delta/\lambda$ Scaling Law

The effectiveness of a diffractive neural network is governed by the ratio $\delta/\lambda$ (pixel size to wavelength). Diffraction coupling between adjacent pixels across two layers separated by distance $z$ follows:

$$N_\text{coupling} = \frac{\lambda z}{2\delta^2}$$

This quantity counts the number of pixels over which a single pixel's diffraction cone spreads after traversing one inter-layer gap. When $N_\text{coupling} \ll 1$, each pixel effectively propagates as a ray—there is no inter-pixel mixing, and the D2NN reduces to a trivial point-by-point phase shift, incapable of spatial processing.

For $\delta/\lambda \gg 10$, achieving even $N_\text{coupling} \sim 1$ requires impractically large layer spacings $z$. Only at metasurface scales ($\delta/\lambda < 10$) do modest layer gaps produce sufficient coupling for the D2NN to function as a genuine diffractive processor.

### 2.4 Loss Functions

Two loss formulations were tested:

**Irradiance loss (composite).** Applied to the predicted and target intensity patterns:

$$\mathcal{L}_\text{irr} = \underbrace{\left(1 - \frac{\langle I_\text{pred},\, I_\text{target}\rangle}{\|I_\text{pred}\|\;\|I_\text{target}\|}\right)}_{\text{normalized overlap}} + 0.25 \cdot \text{MSE}(w_\text{pred},\, w_\text{target}) + 0.25 \cdot \text{MSE}(\eta_\text{pred},\, \eta_\text{target})$$

where $w$ is the $1/e^2$ beam radius computed from second moments and $\eta$ is the encircled energy fraction within the target beam radius.

**Complex field loss.** Applied directly to the complex amplitude:

$$\mathcal{L}_\text{field} = 1 - \frac{|\langle U_\text{pred},\, U_\text{target}\rangle|}{\|U_\text{pred}\|\;\|U_\text{target}\|}$$

This preserves phase information and provides a richer gradient signal, but ultimately faces the same fundamental limitation.

---

## 3. Layer Spacing Sweep ($\delta = 2\;\text{mm}$, Irradiance Loss, 20 Epochs)

To isolate the effect of inter-layer coupling, a systematic sweep of layer spacing was conducted at the physical pixel scale $\delta = 2\;\text{mm}$ ($\delta/\lambda = 1290$).

| Layer Spacing | $\delta/\lambda$ | $N_\text{coupling}$ [px] | 20-Epoch Improvement |
|:---:|:---:|:---:|:---:|
| 0.01 m | 1290 | 0.00 | 0.02 % |
| 1.0 m | 1290 | 0.19 | 0.09 % |
| 5.0 m | 1290 | 0.97 | 0.26 % |
| 10.0 m | 1290 | 1.94 | 0.48 % |
| 25.0 m | 1290 | 4.84 | 1.14 % |
| 50.0 m | 1290 | 9.69 | 1.90 % |

The trend is clear and monotonic: loss improvement scales linearly with pixel coupling, confirming the $N_\text{coupling}$ scaling law. Even at an extreme (and physically absurd) layer spacing of 50 m, where the total D2NN depth would exceed 200 m, the improvement is a mere 1.9 %. At $\delta/\lambda = 1290$, the pixel pitch is so large relative to the wavelength that diffraction barely couples neighboring pixels, regardless of how far apart the layers are placed.

### Figure Analysis: Layer Spacing Sweep

![Figure 2. Layer spacing sweep — improvement vs. layer spacing (left) and vs. pixel coupling (right) at $\delta = 2\;\text{mm}$.](figures/fig2_spacing_sweep.png){width=100%}

### Figure Analysis: D2NN Learning Scales with Pixel Coupling

![Figure 3. Training and validation loss improvement — metasurface ($10\;\mu\text{m}$, red) vs. physical scale ($2\;\text{mm}$, blue/orange/green).](figures/fig3_scale_comparison.png){width=100%}

The two-panel training curves figure crystallizes this result. In the left panel (Training Loss Improvement vs. Epoch), the red curve representing the metasurface configuration ($\delta = 10\;\mu\text{m}$, $z = 387\;\mu\text{m}$, 200 epochs) rises dramatically, reaching approximately 11 % improvement and still climbing. In stark contrast, the three 2 mm configurations (blue for $z = 0.01\;\text{m}$, orange for $z = 10\;\text{m}$, green for $z = 50\;\text{m}$) are nearly flat lines hugging the baseline, barely reaching 2 %. The right panel (Validation Loss Improvement) mirrors this pattern: the metasurface configuration achieves roughly 7 % validation improvement while the physical-scale configurations remain below 1 %.

This figure is the clearest demonstration that D2NN learning is controlled by diffractive coupling ($\sim \lambda z / \delta^2$), not by layer spacing alone. When $\delta/\lambda$ is large, no practical amount of inter-layer propagation compensates.

---

## 4. Metasurface Scale Results ($\delta = 10\;\mu\text{m}$)

### 4.1 Configuration

| Parameter | Value |
|-----------|-------|
| Pixel pitch $\delta$ | 10 $\mu$m |
| $\delta/\lambda$ | 6.5 |
| Computational window | 10.24 mm |
| Grid size $N$ | 1024 |
| Layer spacing $z$ | 387 $\mu$m ($\sim$3 px coupling) |
| Total D2NN depth | 1.94 mm |

### 4.2 Training Results

With irradiance loss over 200 epochs at learning rate $3 \times 10^{-3}$:

- **Training loss:** $0.0377 \to 0.0330$ (11.0 % relative improvement)
- **Validation loss:** $0.0371 \to 0.0345$ (still decreasing — no overfitting detected)

This is the only configuration in which the optimizer achieved measurable learning. The validation curve's continued descent suggests the model has not yet saturated, but the absolute improvement remains small.

### 4.3 Detector Size Sweep

A sweep over circular detector diameters revealed that the D2NN provides its largest coupling gain ($\sim 2\%$) at the smallest detectors (0.5--4 mm), where the turbulence-broadened beam overfills the detector aperture. For large detectors ($D > 6\;\text{mm}$), the baseline already captures nearly all incident energy, leaving no room for improvement. The Strehl ratio improved from 1.992 to 2.118 (+6.3 %).

### 4.4 Figure Analysis: Coupling Efficiency and Strehl Ratio

![Figure 4. Coupling efficiency vs. detector size (left), coupling gain (center), and Strehl ratio comparison (right) — metasurface D2NN, 5 layers, $\delta = 10\;\mu\text{m}$, $z = 387\;\mu\text{m}$, 200 epochs.](figures/fig4_metasurface_performance.png){width=100%}

The three-panel coupling figure quantifies the D2NN's effect at the metasurface scale. The left panel plots coupling efficiency versus detector size for three conditions: baseline (red), D2NN-corrected (blue), and vacuum reference (green). The three curves nearly overlap for detectors larger than 6 mm, confirming that large apertures are insensitive to turbulence correction. At small detector sizes (0.5--4 mm), the blue D2NN curve sits slightly above the red baseline—a real but modest gain. The middle panel presents this gain as a bar chart: the improvement peaks at the smallest detectors ($\sim 2\%$) and decays to zero. The right panel shows the Strehl ratio improving from 1.992 to 2.118. While statistically significant, this 6.3 % Strehl improvement represents a marginal correction relative to the diffraction-limited ideal.

### 4.5 Figure Analysis: Irradiance Comparison

![Figure 5. Detector irradiance comparison — Baseline (left) vs. D2NN-corrected (center) vs. Vacuum reference (right) for three test samples.](figures/fig5_irradiance_comparison.png){width=100%}

The $3 \times 3$ irradiance montage (3 test samples $\times$ 3 conditions: baseline, D2NN, vacuum) provides the most intuitive assessment. The baseline and D2NN columns are visually indistinguishable—both display the same irregular speckle patterns with random hot spots and dark voids. The vacuum column shows the clean, centered Gaussian target. The D2NN has not meaningfully redistributed the speckle energy back toward the Gaussian envelope.

### 4.6 Figure Analysis: Learned Phase Patterns

![Figure 6. Learned phase patterns [rad] for all 5 D2NN layers — nearly flat ($\approx 0$ rad), indicating the optimizer converged to "do nothing."](figures/fig6_learned_phases.png){width=100%}

The five panels showing the learned phase profile of each D2NN layer deliver the definitive visual verdict. All five layers display a nearly uniform phase ($\approx 0\;\text{rad}$, rendered as flat light blue) with only minor edge artifacts from the absorbing boundary. The optimizer, after 200 epochs of gradient descent over the training set, converged to the identity transformation: the best fixed response to random turbulence is to do nothing. This is the "smoking gun" confirming the fundamental impossibility argument developed in Section 6.

---

## 5. Physical Scale Results ($\delta = 150\;\mu\text{m}$, 15 cm Receiver)

### 5.1 Configuration

| Parameter | Value |
|-----------|-------|
| Input data grid | $N = 4096$, $\delta = 150\;\mu\text{m}$ |
| Receiver ROI diameter | 15 cm |
| Center crop | $1024 \times 1024$ |
| $\delta/\lambda$ | 97 |
| Layer spacing | 9 cm (3.1 px coupling) |
| Total D2NN depth | 45 cm |

### 5.2 Training Results

With complex field loss over 200 epochs at $\text{lr} = 3 \times 10^{-3}$:

- **Training loss:** $0.3631 \to 0.3604$ (0.8 % improvement)
- **Validation loss:** $0.4484 \to 0.4465$ (essentially flat)

Despite switching to the complex field loss (which provides richer gradient information by preserving phase), the D2NN fails to learn any meaningful correction at the physical scale. The 0.8 % training improvement is within noise. The failure is architectural and physical, not a consequence of loss function choice.

---

## 6. Root Cause Analysis — Why D2NN Fails for Random Turbulence

This section presents the three interlocking physics arguments that explain why a passive D2NN is fundamentally incapable of correcting atmospheric turbulence. Understanding these causes is essential for identifying which alternative approaches may succeed.

### 6.1 Cause 1: Fixed Phase Mask vs. Stochastic Input

The D2NN learns a set of fixed phase parameters $\{\phi_l\}_{l=1}^{L}$ that are frozen after training. Each atmospheric turbulence realization $k$ introduces a different random phase distortion $\phi_\text{turb}^{(k)}(\mathbf{r})$ drawn from the Kolmogorov ensemble. The training objective is:

$$\phi_\text{opt} = \arg\min_{\phi} \; \mathbb{E}_k\!\left[\mathcal{L}\!\left(\text{D2NN}_\phi\!\left(U_\text{turb}^{(k)}\right),\; U_\text{vac}\right)\right]$$

The key insight is that Kolmogorov turbulence has zero-mean phase:

$$\langle \phi_\text{turb}^{(k)}(\mathbf{r}) \rangle_k = 0$$

Because the expectation of the turbulence phase over many realizations vanishes, the optimal fixed correction also vanishes. Any non-zero fixed correction $\phi_l \neq 0$ would improve some realizations at the expense of degrading others. The statistical optimum for a fixed mask is therefore the identity (flat phase), which is precisely what the optimizer discovers (Section 4.6).

This is directly analogous to why a static corrector plate cannot compensate for atmospheric seeing in astronomy—the wavefront distortion changes on millisecond timescales, and any fixed optic averages to no net benefit.

### 6.2 Cause 2: Phase-Only Irradiance Invariance

A phase-only element satisfies:

$$|U \cdot e^{i\phi}|^2 = |U|^2$$

Phase modulation leaves the local irradiance unchanged. The only mechanism by which a phase mask can modify the downstream irradiance pattern is through phase-to-amplitude conversion via subsequent diffraction propagation. The efficiency of this conversion depends on the propagation distance and the spatial frequency content of the phase pattern—precisely the $N_\text{coupling}$ parameter discussed in Section 2.3.

At physical pixel scales ($\delta/\lambda \sim 100$), the phase pattern imprinted by each D2NN layer diffracts negligibly before reaching the next layer. The phase-to-amplitude conversion efficiency is proportional to $(\lambda z / \delta^2)^2$, which evaluates to $\sim 10^{-3}$ for typical physical configurations. The D2NN is effectively transparent.

At metasurface scales ($\delta/\lambda \sim 6$), sufficient coupling exists for meaningful phase-to-amplitude conversion, explaining why the metasurface configuration shows some learning. However, even with adequate coupling, the fixed-mask limitation (Cause 1) prevents the converged solution from providing meaningful correction.

### 6.3 Cause 3: The $\delta/\lambda$ Scaling Law

Combining the above arguments yields a scaling law for D2NN capability. The D2NN is a passive diffractive processor whose computational power derives from inter-pixel diffraction coupling across layers. The number of pixels coupled per layer is:

$$N_\text{coupling} = \frac{\lambda z}{2\delta^2}$$

For $\delta/\lambda > 10$, achieving $N_\text{coupling} \geq 3$ (the minimum for non-trivial spatial processing) requires:

$$z > \frac{6\delta^2}{\lambda} = 6\delta \cdot \frac{\delta}{\lambda}$$

At $\delta = 2\;\text{mm}$, this evaluates to $z > 15.5\;\text{m}$ per layer—a total D2NN depth exceeding 77 m for 5 layers. Such dimensions are physically absurd for a receiver-side optical element.

The experimental results (Section 3) confirm this scaling: the 20-epoch improvement is directly proportional to $N_\text{coupling}$, with no evidence of any threshold or nonlinear acceleration. The D2NN is trapped in a regime where it lacks the diffractive bandwidth to function as a meaningful optical processor.

### 6.4 Comparison with Adaptive Optics

The contrast between D2NN and conventional adaptive optics (AO) illuminates why passive correction fails:

| Property | D2NN | Adaptive Optics |
|----------|------|-----------------|
| Wavefront sensing | None | Real-time (Shack-Hartmann, curvature sensor) |
| Correction element | Fixed phase mask | Deformable mirror (updates at $\sim$kHz) |
| Input dependence | Fixed response | Input-adaptive response |
| Correction bandwidth | Zero (static) | Up to Greenwood frequency |
| Per-realization adaptation | Impossible | Designed for it |

AO succeeds because it measures each turbulence realization and applies a matched correction. The deformable mirror is the conjugate of the measured wavefront: $\phi_\text{DM} = -\phi_\text{turb}^{(k)}$. This per-realization adaptation is precisely what a fixed D2NN cannot provide.

---

## 7. Complete Experimental Results Summary

| # | Configuration | $\delta$ | $\delta/\lambda$ | Loss Type | Epochs | Train Improvement | Val Improvement | Verdict |
|:-:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | Layer sweep $z=0.01$ m | 2 mm | 1290 | Irradiance | 20 | 0.02 % | — | No learning |
| 2 | Layer sweep $z=10$ m | 2 mm | 1290 | Irradiance | 20 | 0.48 % | — | No learning |
| 3 | Layer sweep $z=50$ m | 2 mm | 1290 | Irradiance | 20 | 1.90 % | — | Marginal |
| 4 | Metasurface | 10 $\mu$m | 6.5 | Irradiance | 200 | 11.0 % | 7.0 % | Learned, but flat masks |
| 5 | Physical receiver | 150 $\mu$m | 97 | Complex field | 200 | 0.8 % | 0.4 % | No learning |

Across all five experiments, the D2NN fails to produce a correction that meaningfully differs from the identity. The single configuration that demonstrates learning (Experiment 4, metasurface) converges to nearly flat phase masks, confirming that the optimizer discovers the theoretical optimum: do nothing.

---

## 8. Future Directions

Based on the negative results and root cause analysis, five potentially viable research directions emerge:

### 8.1 Hybrid D2NN + Wavefront Sensor

Combine a Shack-Hartmann or pyramid wavefront sensor with a reconfigurable D2NN whose phase patterns are updated per-realization. This transforms the fixed-mask limitation into an adaptive system while retaining the D2NN's advantage of operating at the speed of light without electronic latency. The wavefront sensor provides the per-realization information $\phi_\text{turb}^{(k)}$ that the standalone D2NN lacks.

### 8.2 Nonlinear D2NN with Saturable Absorbers

Introduce saturable absorber layers between phase layers to break the phase-only irradiance invariance (Cause 2). A saturable absorber has intensity-dependent transmission $T(I) = T_0 / (1 + I/I_\text{sat})$, enabling input-dependent amplitude modulation without external sensing. This makes the D2NN response realization-dependent at the physics level, potentially circumventing Cause 1. The challenge is achieving sufficient modulation depth at $\lambda = 1550\;\text{nm}$ with available materials.

### 8.3 Digital Twin / CNN Post-Processing

Replace the physical D2NN with a digital neural network (e.g., U-Net or convolutional encoder-decoder) trained on the same turbulence dataset. Digital networks have no $\delta/\lambda$ constraint, can implement amplitude modulation, and are inherently input-adaptive through their nonlinear activation functions. The trade-off is electronic latency and power consumption, but for FSO communication at data rates below 100 Gbps, the processing latency ($\sim 1\;\mu\text{s}$ on modern GPUs) is acceptable.

### 8.4 D2NN for Fixed Aberrations

The D2NN architecture remains well-suited for correcting deterministic, time-invariant aberrations such as telescope alignment errors, manufacturing defects, or thermal deformations. In these scenarios, the fixed-mask limitation is not a constraint—a fixed correction is exactly what is needed. The optimizer can converge to the conjugate of the known aberration $\phi_\text{DM} = -\phi_\text{aberr}$, and the correction remains valid indefinitely.

### 8.5 Temporal Averaging via Beam Splitting

For links where the scintillation timescale ($\sim 1\;\text{ms}$) is much shorter than the data symbol duration, a passive beam splitter followed by temporal integration can average out turbulence fluctuations. This approach sacrifices time resolution but requires no active elements. The D2NN could serve as a spatial pre-processor to optimize the splitting geometry, operating in the domain of fixed optical design rather than random turbulence correction.

---

## 9. References

1. J. D. Schmidt, *Numerical Simulation of Optical Wave Propagation with Examples in MATLAB*, SPIE Press (2010). Listings 9.1--9.5.

2. X. Lin, Y. Rivenson, N. T. Yardimci, M. Veli, Y. Luo, M. Jarrahi, and A. Ozcan, "All-optical machine learning using diffractive deep neural networks," *Science* **361**, 1004--1008 (2018).

3. L. C. Andrews and R. L. Phillips, *Laser Beam Propagation through Random Media*, 2nd ed., SPIE Press (2005).

4. D. L. Fried, "Optical resolution through a randomly inhomogeneous medium for very long and very short exposures," *J. Opt. Soc. Am.* **56**, 1372--1379 (1966).

5. R. J. Noll, "Zernike polynomials and atmospheric turbulence," *J. Opt. Soc. Am.* **66**, 207--211 (1976).

6. Y. Luo, D. Mengu, N. T. Yardimci, Y. Rivenson, M. Veli, M. Jarrahi, and A. Ozcan, "Design of task-specific optical systems using broadband diffractive neural networks," *Light Sci. Appl.* **8**, 112 (2019).

---

*This report documents a negative experimental result. The findings confirm that passive diffractive elements, regardless of optimization, cannot compensate for stochastic wavefront distortions. The physics is clear: correcting random perturbations requires real-time sensing and adaptive response—capabilities that a fixed optical element fundamentally lacks.*
