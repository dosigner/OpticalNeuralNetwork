# Peer Review: Static Diffractive Deep Neural Network Limits in Atmospheric Turbulence Compensation

**Reviewer**: Anonymous (Optica / Nature Photonics style review)
**Date**: 2026-03-27
**Recommendation**: Major Revision

---

## General Assessment

This manuscript addresses an important and timely question: can a static (fixed-weight) diffractive deep neural network compensate random atmospheric turbulence? The central theoretical contribution -- that unitary invariance prevents a static D2NN from improving field-level metrics (CO, WF RMS) while permitting improvement of intensity-based metrics (PIB) through detection nonlinearity -- is physically sound in its ideal formulation. The 247x PIB improvement is a striking demonstration.

However, the paper suffers from several significant gaps between the idealized mathematical framework and the actual numerical implementation, insufficiently rigorous statistical methodology, and missing comparisons with known passive optical techniques. These issues must be addressed before the manuscript is suitable for a high-impact venue.

---

## 1. Mathematical Correctness (수학적 정확성)

### Theorems 1-4: Statement and Proof

**Theorem 1 (Inner Product Preservation)**: Correctly stated and proven. This is a standard result from functional analysis. No issues.

**Theorem 2 (Distance Preservation)**: The theorem itself is correct. However, the **Corollary 2a (WF RMS approximate preservation)** has a serious logical gap.

- The small aberration approximation $U_\text{aberr} \approx U_\text{vac}(1 + i\Delta\phi)$ requires $|\Delta\phi| \ll 1$ (i.e., phase errors much less than 1 radian).
- At D/r_0 = 5.02, the typical RMS wavefront error is approximately $\sigma_\phi \approx 1.03 (D/r_0)^{5/6} \approx 4.1$ rad (Noll 1976). The reported 460 nm at lambda=1.55 um corresponds to $2\pi \times 0.460/1.55 \approx 1.86$ rad RMS. Individual Zernike modes can have peak-to-valley values of 5-10 rad.
- **The small aberration approximation is categorically invalid at D/r_0 = 5.** The first-order Taylor expansion $e^{i\phi} \approx 1 + i\phi$ has relative errors exceeding 50% when phi > 1 rad.
- The authors need to either: (a) prove the WF RMS result without the small aberration approximation, (b) restrict the claim to weak turbulence (D/r_0 < 1), or (c) clearly label this as a heuristic argument confirmed numerically but not rigorously proven for moderate turbulence.

**Theorem 3 (Expected CO invariance under random turbulence)**: The statement is trivially true given Theorem 1 -- if CO is preserved realization-by-realization, the expectation is preserved. This is correctly noted. However, calling this a separate "theorem" is somewhat inflated; it is a one-line corollary of Theorem 1.

**Theorem 4**: The paper references "Theorem 4" in the troubleshooting document but only presents Theorems 1-3 and a counterexample in the main manuscript. The numbering is inconsistent. The counterexample (Section 2.6) serves a theorem-like role but is not formally stated as one. This should be cleaned up.

### Unitarity Assumption

The claim that D2NN is unitary rests on two sub-claims:

1. **Phase masks are unitary**: This is exact for pure phase modulation with no absorption. Valid.
2. **Free-space propagation is unitary (Parseval's theorem)**: This is true for the continuous angular spectrum integral over infinite domain. However, in the numerical implementation:
   - The FFT-based angular spectrum method operates on a **finite, periodic** grid of 1024x1024 pixels.
   - Energy that diffracts beyond the computational window wraps around (periodic boundary) or, if windowed, is lost.
   - The evanescent wave filter $\sqrt{\lambda^{-2} - |f|^2}$ introduces a cutoff for $|f| > 1/\lambda$.

   The authors acknowledge this as "quasi-unitary" but dismiss it based on throughput = 1.0000. This dismissal is premature -- see Section 2 below.

---

## 2. Physical Validity (물리적 타당성)

### Is D2NN truly unitary in practice?

The throughput = 1.0000 evidence is **necessary but not sufficient** for unitarity. Throughput measures $\|HU\|^2 / \|U\|^2 = 1$, which confirms norm preservation. But unitarity requires $H^\dagger H = I$, which is a much stronger condition (preservation of ALL inner products, not just norms).

Specific concerns:

1. **Finite aperture clipping**: After the 75:1 beam reducer, the 2.048 mm D2NN window receives a field that may have energy outside this window (especially for tip/tilt-heavy turbulence realizations that shift the beam centroid). Any clipping at the D2NN boundary breaks unitarity.

2. **FFT aliasing**: The angular spectrum propagation at 10 mm layer spacing with dx = 2 um and lambda = 1.55 um gives a maximum propagation angle of $\theta_\text{max} = \arcsin(\lambda/(2 dx)) = \arcsin(0.3875) \approx 22.8°$. While the diffraction angles from the phase masks must be checked, unconstrained phase masks (stated as [-inf, +inf]) can in principle create arbitrarily steep gradients, pushing energy beyond the Nyquist frequency. The periodic FFT would alias this energy back, preserving total power (throughput = 1.0) but destroying the unitary structure.

3. **Numerical precision**: 32-bit floating point accumulates errors over 5 layers of FFT-multiply-IFFT operations. While individually small, these errors could be significant for the CO preservation claim at the 4th decimal place.

**Recommendation**: The authors should verify unitarity more rigorously by computing $\|H^\dagger H - I\|$ numerically (or at least checking inner product preservation for multiple test pairs), not just throughput.

### "PIB can improve but CO cannot"

This claim is physically correct in the unitary limit. However, the **experimental evidence actually contradicts exact CO preservation**: test CO values range from 0.273 to 0.330 vs. baseline 0.304 (Table in Section 4.1). The ~8% variation is attributed to "overfitting" but could also reflect non-unitary numerical effects. The authors need to disentangle these two explanations.

### Does the 2-pixel counterexample generalize?

The 2-pixel example elegantly demonstrates that PIB can change under unitary transformation. However:

- In 2 pixels, any unitary is a single rotation/reflection -- maximally expressive.
- In 1024x1024, the D2NN has 5M parameters but the full unitary group U(1024^2) has ~10^{12} parameters. D2NN explores only a tiny submanifold of all possible unitaries.
- The question is not whether *some* unitary can improve PIB, but whether the *D2NN-reachable* unitaries can do so. The experimental results (PIB = 83.4%) answer this affirmatively, but the counterexample alone does not constitute a proof of reachability.

The paper would benefit from discussing the expressivity of D2NN within U(N^2) -- how much of the full unitary group is accessible with K=5 layers?

---

## 3. Experimental Methodology (실험 방법론)

### Schmidt angular spectrum propagation

The choice of Schmidt's method with 18 phase screens over 1 km is appropriate for this propagation regime ($D/r_0 = 5$, Rytov variance $\sigma_R^2 = 1.23 C_n^2 k^{7/6} L^{11/6}$). However:

- The Rytov variance should be computed and reported. For the given parameters: $\sigma_R^2 = 1.23 \times 5 \times 10^{-14} \times (4.05 \times 10^6)^{7/6} \times 1000^{11/6}$. This appears to be in the moderate-to-strong scintillation regime, yet the paper models turbulence as **phase-only** ($U_\text{turb} = U_\text{vac} \cdot e^{i\phi}$). Amplitude scintillation (intensity fluctuations) is completely ignored in the theoretical framework but is present in the simulation (split-step propagation naturally produces scintillation). This is a significant inconsistency.

### Statistical sufficiency

- **500 realizations (400/50/50)**: For D/r_0 = 5, the CO distribution has substantial variance. With only 50 test realizations, the standard error of the mean CO is roughly $\sigma_\text{CO}/\sqrt{50}$. If $\sigma_\text{CO} \sim 0.1$, then SE ~ 0.014, which is comparable to the reported "improvements" of 0.025. **The claimed 8% CO improvement is likely not statistically significant.** The authors should report confidence intervals and p-values.

- The 400/50/50 split is unusual. Standard practice would be at least 100 test realizations for reliable statistics. The small validation set (50) also makes early stopping unreliable.

### Deterministic verification

Only 3 Zernike modes tested (defocus Z4, coma Z7, astigmatism Z5). This is insufficient:

- Missing: spherical aberration (Z11), trefoil (Z9), and higher-order modes that test the multi-layer D2NN more severely.
- The single-layer success (phase conjugation) is trivial and does not validate the D2NN framework.
- The multi-layer failure is predicted by theory but should be tested more extensively (different layer spacings, different numbers of layers) to map out the boundary of the unitary constraint.

### PIB bucket size

The 50 um bucket size is mentioned without physical justification. For a single-mode fiber at 1.55 um with NA ~0.14, the mode field diameter is ~10 um. For a multimode fiber, it could be 50-62.5 um. The choice of bucket size critically affects the baseline PIB (0.34%) and the improvement factor. The authors should:
- State what physical detector/fiber this corresponds to
- Show sensitivity of results to bucket size (25 um, 50 um, 100 um)

---

## 4. Interpretation of Results (결과 해석)

### PIB 247x increase

The 247x figure ($0.34\% \to 83.4\%$) is **technically correct but potentially misleading**:

- The baseline 0.34% means almost no energy falls in the bucket without D2NN. This is expected for a turbulence-broadened beam (D/r_0 = 5) with a 50 um bucket that subtends only $(50/2048)^2 \approx 0.06\%$ of the computational area.
- Any reasonable focusing optic (even a simple lens) would dramatically improve this baseline. The relevant comparison is not "no D2NN" but "single lens" or "best passive optical element."
- The 247x number would collapse if the bucket were larger or if a pre-focusing lens were included in the baseline.

**The paper must compare against a simple focusing lens baseline to demonstrate that D2NN provides benefit beyond trivial focusing.**

### CO+PIB hybrid Pareto optimum

The CO+PIB result (PIB = 0.508, CO = 0.294) is interesting. However:

- CO = 0.294 vs. baseline 0.304 is actually a **3.3% degradation**, not "maintenance." This is within statistical noise given 50 test realizations, but should not be described as "CO maintained."
- Whether this represents a true Pareto optimum requires sweeping the loss weight and plotting the full Pareto frontier, not just one operating point.

### Validation vs. Test discrepancy

The val CO improvement (+21%) vs. test CO near-baseline is attributed to overfitting, which is plausible. However:

- With 50 validation samples, the D2NN (5M parameters) can partially memorize the validation set through the training dynamics (validation CO is checked but not backpropagated -- but the training/validation distributions are identical, so training on 400 samples overfits the distribution).
- A stronger test of Theorem 3 would use **completely independent turbulence statistics** (e.g., different Cn^2 or different propagation geometry) to show that CO is invariant.

### WF RMS constancy (460 +/- 5 nm)

- 5 nm variation on a 460 nm baseline is 1.1%. For "exact preservation" (Theorem 2), this should be closer to machine precision (~0.001%). The 1% variation likely reflects a combination of: (a) the invalidity of the small aberration approximation, (b) numerical non-unitarity, (c) the fact that WF RMS is not exactly equal to L2 distance (it involves intensity weighting that changes after D2NN).
- The authors should separate these effects and discuss which dominates.

---

## 5. Missing Considerations (누락된 고려사항)

### Critical omissions:

1. **Scintillation (amplitude fluctuations)**: The theoretical framework assumes pure phase perturbation ($U = U_\text{vac} e^{i\phi}$), but the simulation uses split-step propagation that naturally generates amplitude scintillation. At the reported turbulence strength, log-amplitude variance is non-negligible. The theorems technically apply (they hold for any unitary transformation of any input field), but the physical interpretation in Section 2 that frames turbulence as "phase-only" is misleading and inconsistent with the simulation.

2. **Polarization effects**: Scalar approximation is stated but not justified. At 2 um pixel pitch with lambda = 1.55 um, the feature-to-wavelength ratio is ~1.3, approaching the regime where vector diffraction effects become significant. This should be quantified or at least bounded.

3. **Temporal coherence / bandwidth**: FSO systems use modulated signals, not monochromatic CW. Even a "narrowband" laser has finite linewidth. Chromatic dispersion through 5 layers of diffractive elements (which are inherently wavelength-dependent) could significantly degrade performance. This limitation deserves more than a passing mention.

4. **Detector noise and quantization**: Real photodetectors add shot noise, dark current, and readout noise. The SNR implications of the PIB improvement should be discussed -- concentrating energy in fewer pixels also concentrates photon noise.

5. **Fabrication tolerances**: Phase mask quantization (e.g., 8-level or 16-level lithography), alignment errors between layers, and surface roughness are not discussed. These break the exact phase-only condition and could degrade the unitary property.

6. **Comparison with existing passive methods**: The paper makes no comparison with:
   - Simple spatial filters (pinhole at focus)
   - Mode sorters / photonic lanterns
   - Static aberration corrector plates (for telescope-specific aberrations)
   - Lucky imaging / aperture synthesis (post-processing)

   Without these comparisons, it is impossible to assess the practical utility of D2NN for FSO.

---

## 6. Three Principal Weaknesses (논문의 약점 3개)

### Weakness 1: Gap between ideal unitary theory and numerical implementation

The entire theoretical edifice (Theorems 1-4) rests on exact unitarity ($H^\dagger H = I$). The numerical D2NN on a finite 1024x1024 grid with FFT-based propagation is at best approximately unitary. The authors provide only throughput = 1.0000 as evidence, which proves norm preservation but not full unitarity. The small aberration approximation used for Theorem 2 Corollary is invalid at D/r_0 = 5 (RMS phase error ~1.9 rad >> 1). This gap between theory and implementation undermines the central claims. **A Nature Photonics reviewer would require rigorous numerical verification of unitarity (e.g., inner product preservation tests) and either a proof of WF RMS preservation that does not require the small aberration limit, or honest acknowledgment that the WF RMS result is empirical rather than theoretical.**

### Weakness 2: Insufficient statistical rigor and missing baselines

The 50-sample test set is too small for reliable conclusions at the precision claimed (CO differences of 0.025 on a baseline of 0.304). No confidence intervals, p-values, or bootstrap estimates are provided. The 247x PIB improvement lacks a meaningful baseline comparison -- a simple lens would also produce dramatic PIB improvement from the 0.34% starting point. Without a lens baseline, the D2NN's practical advantage over trivial optics is unquantified. **This is a critical gap for any experimental optics paper.**

### Weakness 3: Scintillation is present in simulations but absent from theory

The split-step propagation naturally generates amplitude scintillation at D/r_0 = 5, yet the theoretical framework and physical interpretation frame turbulence as phase-only. This inconsistency means the "WF RMS" metric (which assumes phase-only aberration) is not well-defined for the actual simulated fields. The theorems still hold (they are about unitary operators on arbitrary fields), but the physical narrative is misleading. For a strong-turbulence FSO paper, ignoring scintillation effects in the discussion is a significant omission.

---

## 7. Suggestions for Improvement (개선 제안)

### Additional experiments/analyses needed:

1. **Unitarity verification**: Compute $\langle HU_i, HU_j \rangle$ for multiple field pairs and compare with $\langle U_i, U_j \rangle$. Report the maximum deviation. This directly tests the foundation of all theorems.

2. **Lens baseline for PIB**: Add a "simple focusing lens" baseline and a "D2NN + focusing lens" configuration. Show that D2NN provides benefit beyond what a single thin lens achieves.

3. **Larger test set**: Increase to at least 200-500 test realizations. Report mean, standard deviation, and 95% confidence intervals for all metrics.

4. **Bucket size sensitivity**: Sweep PIB bucket size from 10 um to 200 um and show how the improvement factor depends on this choice.

5. **Pareto frontier**: Sweep the CO/PIB loss weight continuously and plot the full Pareto curve, not just one point.

6. **D/r_0 sweep**: Test at D/r_0 = 1, 2, 5, 10, 20 to map out performance as a function of turbulence strength. This would reveal whether the results are specific to moderate turbulence or general.

7. **Scintillation analysis**: Compute the scintillation index ($\sigma_I^2$) for the simulated fields and discuss its impact on the theoretical framework.

8. **Expressivity analysis**: How many layers K are needed to approach the theoretical PIB maximum? Plot PIB vs. K to understand D2NN representational capacity.

### Claims that need qualification:

1. "D2NN is unitary" -> "D2NN is approximately unitary under the stated simulation conditions"
2. "WF RMS is preserved (Theorem 2 corollary)" -> "WF RMS is empirically observed to be approximately preserved; the small aberration proof is not strictly valid at D/r_0 = 5"
3. "247x PIB improvement" -> Contextualize with lens baseline and state the absolute PIB values prominently
4. "CO+PIB hybrid maintains CO" -> "CO+PIB hybrid shows CO within statistical uncertainty of the baseline (3.3% degradation, not statistically significant)"
5. "This is the only physical channel" -> "Within the unitary approximation, nonlinear detection is the only channel..." (fabrication imperfections, finite aperture clipping, and nonlinear optical materials provide alternative channels)

---

## Summary

The paper presents an elegant theoretical framework connecting unitary invariance of D2NN to the impossibility of improving field-level turbulence metrics, with the detection nonlinearity as the "escape hatch" for intensity metrics. The core insight is correct and valuable. However, the manuscript needs substantial strengthening in three areas: (1) rigorous verification that the numerical D2NN is actually unitary, not just norm-preserving; (2) proper statistical methodology with adequate sample sizes and meaningful baselines; and (3) honest treatment of the gap between phase-only theory and amplitude+phase reality at moderate turbulence. With these revisions, the paper could make a solid contribution to the understanding of passive diffractive optics for FSO.

---

*Reviewer note: The bilingual (Korean + English) presentation is appreciated for the intended audience but would need full English translation for Optica/Nature Photonics submission.*
