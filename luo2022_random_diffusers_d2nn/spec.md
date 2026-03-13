
# PRD: Reproducing and Extending *Computational Imaging Without a Computer: Seeing Through Random Diffusers at the Speed of Light*

## 1. Document purpose

This PRD defines a **physically grounded, reproducible implementation plan** for the all-optical diffractive imaging system reported in:

- **Main paper:** *Computational imaging without a computer: seeing through random diffusers at the speed of light* (Luo et al., eLight 2022)
- **Supplement:** *Supplementary Information* for the same paper

The goal is twofold:

1. **Baseline reproduction** of the paper’s numerical and experimental results, especially:
   - Fig. 1(a), Fig. 1(b)
   - Fig. 2(a), Fig. 2(b)
   - Fig. 3
   - Fig. 4(a), Fig. 4(b), Fig. 4(c)
   - Fig. 5(a), Fig. 5(b)
   - Fig. 6
   - Fig. 7

2. **Controlled extensions** beyond MNIST:
   - CIFAR-10 evaluation
   - correlation-length shift tests
   - turbulence-like phase screens
   - rotating diffuser inference
   - optional volumetric / multi-slice scatterer simulations

This document is written to minimize physics mistakes. Where the paper leaves an implementation detail ambiguous, that ambiguity is called out explicitly and resolved conservatively.

---

## 2. Source-of-truth hierarchy

### 2.1 Primary source of truth
Use the uploaded main paper and supplementary PDF as the authoritative baseline for:
- geometry
- wavelength / frequency
- diffuser statistics
- loss definition
- figure definitions
- evaluation metrics
- experimental apparatus
- visualization rules

### 2.2 What this PRD adds
This PRD adds:
- a rigorous software architecture
- a parameter/config system
- sweep design
- visualization contracts
- figure-by-figure reproduction procedures
- dynamic diffuser / turbulence extensions
- subagent definitions
- skill definitions
- explicit guardrails for optical correctness

### 2.3 Exact reproduction vs faithful reproduction
There is a crucial distinction:

- **Exact methodological reproduction** is achievable from the paper.
- **Exact bitwise reproduction of every plotted number/image** is **not guaranteed** because several details are not fully specified in the paper, including:
  - exact random seeds
  - exact MNIST subset selection policy behind the stated 50,000-train split
  - exact phase parameterization during optimization
  - exact training image ordering
  - exact fabricated layer material parameters for phase-height conversion (not fully spelled out in the accessible text)
  - exact choice of example digits and diffuser instances shown in visual panels

Therefore, the target should be:
- **exact physics**
- **exact geometry**
- **exact experiment definitions**
- **quantitative agreement within predefined tolerances**
- **qualitative agreement in figure structure and trends**

That is the only honest standard.

---

## 3. Non-negotiable physical assumptions

The baseline paper is built on these assumptions. Do not violate them if the goal is faithful reproduction.

### 3.1 Scalar coherent wave optics
Use a scalar coherent field model throughout the baseline simulation.

### 3.2 Thin phase diffuser model
The baseline diffuser is an **optically thin phase mask**:
- it modulates phase
- it does **not** model multiple scattering inside a volume
- it does **not** model baseline amplitude attenuation in simulation

This is the paper’s main baseline assumption. Experimental diffusers may introduce amplitude loss because of thickness-dependent absorption, but that is an experimental deviation, not part of the training forward model.

### 3.3 Intensity detection at the output
The network output is the output-plane **intensity**:
\[
o(x,y) = |u(x,y)|^2
\]

This matters a lot for dynamic diffusers:
- for a 2D camera with exposure time \(T\), the detector records **time-averaged intensity**
- for a raster-scanned single-pixel detector, the measured image is **not** a snapshot; each pixel is sampled at a different time

### 3.4 Plane-wave illumination in simulation
The baseline forward model assumes plane-wave illumination at the object plane.

### 3.5 Monochromatic THz baseline
The baseline system is monochromatic:
- frequency: **400 GHz**
- wavelength in air: **~0.75 mm**

### 3.6 Passive phase-only diffractive layers
Each trainable layer is phase-only:
\[
t_m(x,y) = \exp(j \phi_m(x,y))
\]

### 3.7 Raw-data metrics; enhanced display images only for visualization
The paper explicitly separates:
- **raw data** for PCC and grating-period quantification
- **contrast-enhanced images** for visual display only

Never compute quantitative metrics from contrast-enhanced images.

---

## 4. Baseline optical specification

## 4.1 Core geometry (baseline system)

| Parameter | Value in paper | Practical code value |
|---|---:|---:|
| Illumination frequency | 400 GHz | 400 GHz |
| Wavelength in air | ~0.75 mm | compute from \(c/f\) |
| Object to diffuser distance | 40 mm | 40.0 mm |
| Object to diffuser (normalized) | ~53 \(\lambda\) | derived |
| Diffuser to first diffractive layer | 2 mm | 2.0 mm |
| Layer-to-layer spacing | 2 mm | 2.0 mm |
| Layer spacing (normalized) | ~2.7 \(\lambda\) | derived |
| Output plane after last layer | 7 mm | 7.0 mm |
| Output distance (normalized) | ~9.3 \(\lambda\) | derived |
| Baseline number of trainable layers | 4 | 4 |

**Important note:** the normalized values in the paper are rounded; use metric distances in the code, and compute normalized labels from the exact wavelength used in your implementation.

## 4.2 Spatial grid

| Parameter | Baseline value |
|---|---:|
| Grid size per layer | 240 × 240 |
| Pixel pitch | 0.3 mm |
| Physical layer width | 72 mm |
| Active object size after resize | 160 × 160 pixels |
| Active object width | 48 mm |
| Padding | zero-pad from 160 to 240 |

### Sampling sanity check
With \(dx = 0.3\) mm and \(\lambda \approx 0.75\) mm:
- Nyquist frequency = \(1/(2dx) \approx 1.667\) cycles/mm
- Propagating cutoff = \(1/\lambda \approx 1.333\) cycles/mm

This is good: the propagating spectrum fits inside the sampled Fourier grid.

---

## 5. Baseline diffuser model

## 5.1 Phase transmittance
The diffuser is modeled as:
\[
t_D(x,y) = \exp\left(j \frac{2\pi \Delta n}{\lambda} D(x,y)\right)
\]

where:
- \(\Delta n \approx 0.74\)
- \(D(x,y)\) is the diffuser height map

## 5.2 Random height map
The height map is generated as:
\[
D(x,y) = W(x,y) * K(\sigma)
\]
with:
\[
W(x,y) \sim \mathcal{N}(\mu, \sigma_0)
\]

### Baseline parameters for training/testing diffusers
| Parameter | In wavelengths | In mm (if \(\lambda=0.75\) mm) |
|---|---:|---:|
| Mean height \(\mu\) | 25 \(\lambda\) | 18.75 mm |
| Std of raw random field \(\sigma_0\) | 8 \(\lambda\) | 6.00 mm |
| Gaussian smoothing std \(\sigma\) | 4 \(\lambda\) | 3.00 mm |
| Mean correlation length \(L\) | ~10 \(\lambda\) | ~7.5 mm |

### Smaller-correlation test diffuser (Supplementary Fig. S5)
| Parameter | Value |
|---|---:|
| Smoothing std \(\sigma\) | 2 \(\lambda\) |
| Mean correlation length \(L\) | ~5 \(\lambda\) |

## 5.3 Correlation-length model
The phase-autocorrelation function is:
\[
R_d(x,y) = \exp\left(-\pi \frac{x^2+y^2}{L^2}\right)
\]

This is used to estimate the mean diffuser correlation length.

## 5.4 Unique diffuser criterion
Two diffusers are considered distinct if the average pixel-wise absolute phase difference exceeds:
\[
\Delta \phi > \pi/2
\]
after subtracting each diffuser’s mean phase.

**Implementation rule:** keep this uniqueness test exactly, because it governs “new diffuser” generation.

## 5.5 Important implementation note about the mean height
The mean height \(\mu\) contributes mostly a global phase term. Since global phase does not affect intensity, subtracting the mean phase is numerically harmless for propagation. However:
- keep the original diffuser-generation process for baseline fidelity
- use mean-subtracted phase only for the uniqueness test, matching the paper

---

## 6. Forward propagation model

## 6.1 Reference model (paper)
The paper states the forward model using a Rayleigh-Sommerfeld propagation kernel:
\[
u_0(x,y,z_0)= t_D(x,y)\,[h(x,y,0) * w(x,y,z_0)]
\]
\[
w(x,y,z)= \frac{z}{r^2}\left(\frac{1}{2\pi r}+\frac{1}{j\lambda}\right)\exp\left(j\frac{2\pi r}{\lambda}\right)
\]
with:
\[
r=\sqrt{x^2+y^2+z^2}
\]

Layer modulation:
\[
t_m(x,y,z_m)=\exp(j\phi_m(x,y,z_m))
\]

Propagation through each layer:
\[
u_m(x,y,z_m) = t_m(x,y,z_m)\,[u_{m-1}(x,y,z_{m-1}) * w(x,y,\Delta z_m)]
\]

Output intensity:
\[
o(x,y)=|u_M * w(x,y,\Delta z_d)|^2
\]

## 6.2 Production implementation choice: BL-ASM
For software, implement the production propagator as **band-limited angular spectrum method (BL-ASM)** because it is FFT-efficient and stable.

Use:
\[
H(f_x,f_y;z)=
\exp\left(j 2\pi z \sqrt{\frac{1}{\lambda^2}-f_x^2-f_y^2}\right)
\]
for propagating components, with proper band-limiting and evanescent handling.

### Required policy
- **BL-ASM is the production propagator**
- **Rayleigh-Sommerfeld (RS) propagation is the regression oracle**

### Acceptance test
For a fixed padded grid, verify that BL-ASM and RS agree on normalized output intensity within a tight tolerance on canonical cases:
- free-space propagation without diffuser
- propagation through a baseline diffuser
- one-layer phase modulation
- four-layer network after random initialization

## 6.3 Padding rule
Because FFT propagation is circular:
- use at least **2× zero-padding** in both dimensions for production
- use **4× padding** for regression/oracle tests if compute allows

Do not skip padding; otherwise wrap-around contamination will silently corrupt the figures.

## 6.4 Lens-based imaging baseline
For the lens baseline used in Fig. 1(b) and Supplementary Fig. S2:
- Fresnel lens focal length:
  \[
  f = 145.6 \lambda
  \]
- pupil diameter:
  \[
  104 \lambda
  \]
- lens placed:
  \[
  2f
  \]
  away from the object
- object-to-diffuser propagation:
  \[
  z_0 = 53 \lambda
  \]
- propagate diffuser output to lens plane
- multiply by lens transmission
- propagate to image plane \(2f\) behind the lens

### Lens transmission
\[
t_L(\tilde x,\tilde y)= A(\tilde x,\tilde y)\,\exp\left(-j\frac{\pi}{\lambda f}(\tilde x^2+\tilde y^2)\right)
\]
with pupil:
\[
A(\tilde x,\tilde y)=
\begin{cases}
1 & \sqrt{\tilde x^2+\tilde y^2}<52\lambda \\
0 & \text{otherwise}
\end{cases}
\]

---

## 7. Training model

## 7.1 Dataset protocol (baseline)
The paper states:
- 50,000 MNIST images for training
- 10,000 MNIST images for testing
- each MNIST image upscaled from 28 × 28 to 160 × 160 by bilinear interpolation
- then zero-padded to 240 × 240

### Ambiguity note
Standard MNIST has 60,000 training images and 10,000 test images. The paper explicitly says 50,000 are used for training, but does not specify which 50,000. Therefore the reproduction protocol should define and freeze this.

### Recommended reproducible split
- `train = first 50,000` images of the standard MNIST training set
- `val = last 10,000` images of the standard MNIST training set
- `test = standard 10,000` test images

This keeps the published 50,000-train statement while making the implementation deterministic.

## 7.2 Batch and diffuser usage
- training batch size in objects: **B = 4**
- at the start of each epoch, sample **n** diffusers
- keep those **n diffusers fixed for the entire epoch**
- in each iteration, each image in the batch is duplicated across the **n** diffusers
- therefore each iteration processes **B × n** distorted inputs

**This detail is essential for Fig. 5. Do not resample diffusers every mini-batch.**

## 7.3 Loss
The total loss averages over all \(B \times n\) outputs:
\[
\mathrm{Loss} = \frac{1}{B n}\sum_{b=1}^B \sum_{i=1}^n \left[-P(o_{bi},h_b) + E(o_{bi},h_b)\right]
\]

where \(P\) is PCC and \(E\) is the energy-efficiency penalty.

### PCC
\[
P = \frac{\sum (O-\bar O)(G-\bar G)}{\sqrt{\sum (O-\bar O)^2 \sum (G-\bar G)^2}}
\]

### Binary mask
\[
\hat h_b(x,y)=
\begin{cases}
1,& h_b(x,y)>0 \\
0,& \text{otherwise}
\end{cases}
\]

### Energy penalty
Use the paper’s structure:
- penalize light outside the object support
- reward light inside the object support
- hyperparameters:
  - \(\alpha = 1\)
  - \(\beta = 0.5\)

In code, implement:
\[
E(o_{bi},h_b)=
\frac{\sum_{x,y}\left[\alpha(1-\hat h_b)o_{bi} - \beta \hat h_b o_{bi}\right]}{\sum_{x,y}\hat h_b}
\]

## 7.4 Optimizer and learning-rate schedule
The paper states:
\[
L_r = 0.99^{Ite}\times 10^{-3}
\]
with Adam.

### Important ambiguity
The paper text labels `Ite` as the current iteration number. But elsewhere the paper says the end-of-training learning rate is approximately \(3\times 10^{-4}\), which matches:
\[
10^{-3}\times 0.99^{100} \approx 3.66\times 10^{-4}
\]
That is consistent with **epoch-wise decay over 100 epochs**, not mini-batch-wise decay.

### Required resolution
For faithful reproduction, use:
- **epoch-wise multiplicative decay by 0.99**
- initial LR = **1e-3**
- final LR after 100 epochs ≈ **3.7e-4**

Log this explicitly as a resolved ambiguity.

## 7.5 Training duration and environment
The paper reports:
- 100 epochs
- Python 3.7.3
- TensorFlow 1.13.0
- GTX 1080 Ti
- Intel i9-7900X
- 64 GB RAM
- typical training time ~24 h for `n=20`

Modern PyTorch or JAX implementations are acceptable for reimplementation, but the resolved config and saved seeds must be preserved.

---

## 8. Figure-specific reproduction requirements

## 8.1 Fig. 1(a): training and design schematic

### Purpose
Show the baseline geometry and the training/blind-test logic.

### Must show
- object plane
- phase diffuser
- 4 diffractive layers
- output plane
- object-to-diffuser distance: 40 mm (~53 \(\lambda\))
- diffuser-to-first-layer: 2 mm
- layer-to-layer spacing: 2 mm
- last-layer-to-output: 7 mm
- training over 100 epochs with `n` diffusers per epoch
- blind testing with 20 new diffusers

### Acceptance
The geometry and labels must match the paper conceptually and dimensionally.

## 8.2 Fig. 1(b): object distortion and all-optical reconstruction

### Panel rows
1. target objects
2. free-space propagation through diffuser, measured at output plane, without diffractive layers
3. lens-based imaging through diffuser
4. diffractive-network output

### Inputs
Use sample MNIST test digits equivalent in class to those shown (the paper shows digits such as 2, 5, 6 in the figure montage).

### Important caveat
The exact example digit shapes in the paper are not specified by index; reproduce the **figure logic and panel layout**, not the exact stroke shape.

### Output policy
- show raw data for metrics
- for display, use the same visual enhancement rule as the paper

---

## 8.3 Fig. 2(a), Fig. 2(b): simulation with known vs new diffusers

### Model
- 4-layer network
- `n=20` diffusers per epoch
- 100 epochs
- baseline MNIST training
- evaluation on:
  - digits: 0, 2, 7
  - resolution targets: 10.8 mm, 12.0 mm
- known diffusers: three selected from the last epoch’s `n=20`
- new diffusers: three selected from a newly generated set of 20

### Known/new diffuser naming
- known: `K1, K2, K3`
- new: `B1, B2, B3`

### Must show
- diffuser phase maps on the top row
- output reconstructions below
- PCC under each reconstructed image

### Reference simulation PCC values visible in the paper
These are useful as **reference ranges**, not as an absolute bitwise target.

#### Fig. 2(a) known diffusers
- digit 0: 0.7884, 0.7814, 0.7953
- digit 2: 0.7895, 0.7746, 0.7901
- digit 7: 0.7741, 0.7681, 0.7758
- 10.8 mm target: 0.7517, 0.7512, 0.7268
- 12.0 mm target: 0.7371, 0.7390, 0.7449

#### Fig. 2(b) new diffusers
- digit 0: 0.7778, 0.7762, 0.7867
- digit 2: 0.7771, 0.7788, 0.7801
- digit 7: 0.7684, 0.7949, 0.7723
- 10.8 mm target: 0.7322, 0.7558, 0.7507
- 12.0 mm target: 0.7471, 0.7313, 0.7259

### Acceptance
- mean simulation PCC should be in the same regime (~0.73–0.80)
- qualitative reconstruction quality should match
- known and new diffuser results should be comparable

---

## 8.4 Fig. 3: period recovery under different training `n`

### Purpose
Quantify generalization to previously unseen grating-like objects that were not in MNIST training.

### Test periods
Use amplitude-encoded binary resolution targets with:
- 7.2 mm
- 8.4 mm
- 9.6 mm
- 10.8 mm
- 12.0 mm

### Training conditions
Train four separate 4-layer networks with:
- `n = 1`
- `n = 10`
- `n = 15`
- `n = 20`

### Evaluation cases
- Fig. 3(a): last `n` diffusers from training epoch 100
- Fig. 3(b): 20 new diffusers

### Period quantification
For each reconstructed resolution target:
1. average the output intensity along \(y\)
2. fit the resulting 1D profile using a sum of three Gaussians
3. compute the resolved period:
   \[
   \hat p = \frac{\max(b_1,b_2,b_3)-\min(b_1,b_2,b_3)}{2}
   \]

### Acceptance
- recovered periods should track the true period line
- the figure trend should show successful resolution under both known and new diffusers
- all `n` values should resolve the periods, though `n=1` may be less robust elsewhere

---

## 8.5 Fig. 4(a), Fig. 4(b), Fig. 4(c): experiment

### Experimental baseline
The paper uses:
- CW coherent THz at 0.4 THz
- raster-scanned single-pixel detector
- 42 × 42 mm output field of view for hand-written digits
- detector aperture 0.5 × 0.25 mm
- scan step size 1 mm
- aluminum-foil-coated binary objects
- fabricated known diffusers K1–K3
- fabricated new diffusers B1–B3
- fabricated 4-layer diffractive network

### Fig. 4(a), Fig. 4(b)
Show experimental all-optical reconstructions for:
- digits: 0, 2, 7
- known diffusers K1–K3
- new diffusers B1–B3

### Visible PCC values in the paper
#### Known diffusers
- digit 0: 0.4207, 0.4639, 0.4603
- digit 2: 0.5273, 0.5080, 0.4551
- digit 7: 0.4943, 0.5124, 0.4990

#### New diffusers
- digit 0: 0.4218, 0.3982, 0.3702
- digit 2: 0.5242, 0.4274, 0.5225
- digit 7: 0.4892, 0.5189, 0.5185

### Fig. 4(c)
Show experimental reconstruction of 10.8 mm and 12.0 mm resolution targets, with resolved period numbers printed in red.

#### Visible period values in the paper
For 10.8 mm target:
- known: 10.92, 10.91, 10.72 mm
- new: 11.73, 11.29, 10.68 mm

For 12.0 mm target:
- known: 11.81, 12.66, 12.34 mm
- new: 12.51, 12.07, 12.10 mm

#### Mean values stated in the text
- 10.8 mm target:
  - known diffusers: **10.851 ± 0.121 mm**
  - new diffusers: **11.233 ± 0.531 mm**
- 12.0 mm target:
  - known diffusers: **12.269 ± 0.431 mm**
  - new diffusers: **12.225 ± 0.245 mm**

### Acceptance
- experimental PCC is expected to be lower than numerical PCC
- measured periods should remain close to target periods
- known/new diffuser gap should remain small

### Known experimental degraders to track
The paper attributes experimental degradation to:
- nonuniform incident THz field
- fabrication imperfections
- mechanical misalignment
- lower SNR after strong diffraction by diffuser

These must be treated as first-class logged variables.

---

## 8.6 Fig. 5(a), Fig. 5(b): network memory

### Purpose
Quantify how much the network “remembers” the last diffusers seen during training.

### Training conditions
Train separate 4-layer networks with:
- `n=1`
- `n=10`
- `n=15`
- `n=20`

Each network is trained for 100 epochs.

### Fig. 5(a)
For each finalized network:
- compute the mean PCC over 10,000 MNIST test objects for every **known diffuser**
- order diffusers by training introduction index
- plot the solid line across all `100n` diffusers
- overlay the dashed mean over diffusers from epochs 1–99
- use zoomed inset for the last 50 diffusers where appropriate

### Diffuser indexing contract
If a diffuser is introduced in epoch `e` and has local index `i` inside that epoch, define:
\[
\text{global\_diffuser\_index} = (e-1)n + i
\]

### Fig. 5(b)
For each finalized network:
- compute PCC on diffusers introduced in the last 10 epochs (total `10n`)
- compute PCC on 20 new diffusers
- show mean ± std over diffusers

### Required trend
- epoch-100 diffusers should show elevated PCC
- `n=1` should show fading memory / overfitting
- `n=10,15,20` should generalize similarly on new diffusers

---

## 8.7 Fig. 6: known vs new vs no diffuser

### Purpose
Demonstrate that the trained network becomes a general-purpose imager, not just a diffuser-specific inverter.

### Procedure
For each network trained with:
- `n=1`
- `n=10`
- `n=15`
- `n=20`

evaluate the same test object under:
1. last known diffuser
2. a new diffuser
3. no diffuser

### Required output
- image montage with three output columns
- PCC bar chart with:
  - known diffusers (epoch 100)
  - known diffusers (epochs 1–99)
  - new diffusers
  - no diffuser

### Required trend
- no-diffuser performance should be higher than new-diffuser performance
- `n=1` should overfit more strongly
- `n=10,15,20` should be similar and stronger than `n=1` on new diffusers

---

## 8.8 Fig. 7: depth advantage

### Purpose
Show that additional trainable diffractive layers improve reconstruction fidelity.

### Sweep
For each:
- `n=1`
- `n=10`
- `n=15`
- `n=20`

train networks with:
- 2 layers
- 4 layers
- 5 layers

### Required outputs
For each `n`, plot mean ± std PCC under:
- known diffusers (epoch 100)
- known diffusers (epochs 1–99)
- new diffusers

### Required trend
Increasing depth should improve average PCC, especially relative to shallow networks.

### Important ambiguity
The paper does not fully spell out whether total physical depth was held constant across layer-count sweeps. The conservative reproduction choice is:
- keep the same 2 mm inter-layer spacing
- keep the same 7 mm output gap
- allow total system depth to vary with layer count

Log this choice explicitly.

---

## 9. Experimental hardware requirements (for Fig. 4 fidelity)

## 9.1 THz system essentials
The paper reports:
- source: WR2.2 AMC / multiplier chain
- source RF input: 11.111 GHz sinusoid, 10 dBm
- multiplication factor: ×36
- output radiation: 0.4 THz CW
- detector LO input: 11.083 GHz sinusoid, 10 dBm
- down-converted IF: 1 GHz
- electrical modulation: 1 kHz square wave
- lock-in based detection

## 9.2 Output scanning
- single-pixel detector on XY motorized stage
- scan raw amplitude, then convert to linear scale using calibration
- do not normalize arbitrarily between panels before PCC

## 9.3 Experimental objects
The paper used:
- 3 handwritten digits
- 2 resolution targets
- aluminum foil coating to realize binary transmittance

## 9.4 Practical recommendation
If the goal is numerical reproduction first and experiment later, structure the codebase so that:
- simulation and experimental pipelines use the same figure composer
- only the data source changes
- calibration is a preprocessing stage, not baked into figure code

---

## 10. Visualization policy

## 10.1 Separate raw and display images
Store for every output:
- `raw` image: used for metrics
- `display` image: used for plots only

## 10.2 Contrast enhancement rule
To mimic the paper:
- apply MATLAB-like `imadjust` behavior to display images only
- saturate bottom/top 1% of pixel values
- map to [0, 1]

Apply this only to figures where the paper says it was used:
- Fig. 1(b)
- Fig. 4
- Supplementary Fig. S2
- Supplementary Fig. S5

Do **not** apply it before PCC or grating-period estimation.

## 10.3 Scale bars
Use the paper’s style:
- phase maps: color bar from 0 to \(2\pi\)
- output intensity: grayscale with labels 0 and 1
- spatial scale bar: **10 \(\lambda\)**

At 400 GHz:
- \(10\lambda \approx 7.5\) mm

## 10.4 Plot contracts
Every figure function must define:
- input dataset / diffuser IDs
- raw arrays used
- display transform used
- fixed axis labels
- fixed colormap
- seed used for sample selection
- saved panel order

## 10.5 Required summary dashboards
In addition to paper figures, generate:
- PCC histogram by diffuser type
- diffuser correlation-length histogram
- training LR curve
- RS vs BL-ASM regression error report
- dynamic diffuser speed vs PCC curves
- CIFAR-10 classwise reconstruction summaries

---

## 11. Parameter config design

Use hierarchical YAML configs with a resolved config snapshot saved for every run.

## 11.1 Recommended config tree

```text
configs/
  dataset/
    mnist.yaml
    cifar10_gray.yaml
    cifar10_binary.yaml
    resolution_targets.yaml
  optics/
    thz_400ghz.yaml
  diffuser/
    random_phase_L10.yaml
    random_phase_L5.yaml
    turbulence_vonkarman.yaml
    rotating_random_phase.yaml
  model/
    d2nn_2layer.yaml
    d2nn_4layer.yaml
    d2nn_5layer.yaml
  training/
    n1.yaml
    n10.yaml
    n15.yaml
    n20.yaml
  experiment/
    fig1b.yaml
    fig2.yaml
    fig3.yaml
    fig4_exp.yaml
    fig5.yaml
    fig6.yaml
    fig7.yaml
    cifar10_eval.yaml
    rotating_diffuser.yaml
  visualization/
    paper_style.yaml
    raw_only.yaml
sweeps/
  fig3_n_sweep.yaml
  fig5_memory.yaml
  fig7_depth.yaml
  cifar10_ablation.yaml
  turbulence_sweep.yaml
  rotating_diffuser_sweep.yaml
```

## 11.2 Canonical baseline config

```yaml
experiment:
  id: fig2_baseline_n20
  mode: simulate
  seed: 20220126
  save_dir: runs/${experiment.id}

optics:
  frequency_ghz: 400.0
  wavelength_mm: 0.75
  coherent: true
  scalar_model: true
  detector_type: intensity

grid:
  nx: 240
  ny: 240
  pitch_mm: 0.3
  pad_factor: 2
  crop_after_propagation: true

geometry:
  object_to_diffuser_mm: 40.0
  diffuser_to_layer1_mm: 2.0
  layer_to_layer_mm: 2.0
  num_layers: 4
  last_layer_to_output_mm: 7.0

dataset:
  name: mnist
  train_count: 50000
  val_count: 10000
  test_count: 10000
  source_resolution_px: 28
  resize_to_px: 160
  final_resolution_px: 240
  resize_mode: bilinear
  amplitude_encoding: grayscale
  mask_strategy: positive_pixels
  deterministic_split: first_50000_train_last_10000_val

diffuser:
  type: thin_random_phase
  delta_n: 0.74
  height_mean_lambda: 25.0
  height_std_lambda: 8.0
  smoothing_sigma_lambda: 4.0
  target_correlation_length_lambda: 10.0
  uniqueness_delta_phi_min_rad: 1.5707963267948966

model:
  type: d2nn_phase_only
  num_layers: 4
  phase_parameterization: wrapped_phase
  init_phase_distribution: uniform_0_2pi

training:
  epochs: 100
  batch_size_objects: 4
  diffusers_per_epoch: 20
  optimizer: adam
  learning_rate_initial: 1.0e-3
  learning_rate_schedule:
    type: epoch_multiplicative
    gamma: 0.99
  loss:
    type: pcc_plus_energy
    alpha: 1.0
    beta: 0.5

evaluation:
  metrics: [pcc]
  use_raw_images_for_metrics: true
  known_diffuser_count_from_last_epoch: 20
  blind_test_new_diffuser_count: 20

visualization:
  paper_style: true
  contrast_enhancement:
    type: percentile_stretch
    lower_percentile: 1.0
    upper_percentile: 99.0
  save_raw: true
  save_display: true
```

## 11.3 Fig. 3 sweep config

```yaml
sweep:
  name: fig3_n_sweep
  parameters:
    training.diffusers_per_epoch: [1, 10, 15, 20]
    model.num_layers: [4]
    dataset.name: [resolution_targets]
  fixed:
    evaluation.resolution_periods_mm: [7.2, 8.4, 9.6, 10.8, 12.0]
    evaluation.known_diffusers_from_last_epoch: true
    evaluation.new_diffusers_count: 20
```

## 11.4 Fig. 7 depth sweep config

```yaml
sweep:
  name: fig7_depth
  parameters:
    training.diffusers_per_epoch: [1, 10, 15, 20]
    model.num_layers: [2, 4, 5]
  fixed:
    geometry.layer_to_layer_mm: 2.0
    geometry.last_layer_to_output_mm: 7.0
```

## 11.5 Rotating diffuser sweep config

```yaml
experiment:
  id: rotating_diffuser_snapshot_vs_raster

dynamic_medium:
  enabled: true
  type: rotating_diffuser
  angular_velocity_rad_s: [0.0, 0.1, 1.0, 5.0, 10.0]
  rotation_axis: center
  interpolation: bicubic
  large_canvas_margin_factor: 1.5

detector:
  mode: [snapshot, raster_scan]
  exposure_ms: [0.1, 1.0, 10.0]
  raster:
    step_mm: 1.0
    dwell_ms: 1.0
    scan_order: row_major
    synchronize_rotation: false
```

## 11.6 Turbulence sweep config

```yaml
dynamic_medium:
  enabled: true
  type: thin_phase_turbulence
  model: modified_von_karman
  r0_mm: [5.0, 10.0, 20.0, 40.0]
  L0_mm: [100.0, 300.0]
  l0_mm: [0.3, 1.0]
  frozen_flow_velocity_mm_s: [0.0, 10.0, 50.0]
  num_phase_screens: 1
```

---

## 12. Repository and artifact structure

```text
repo/
  src/
    physics/
      rs_reference.py
      bl_asm.py
      lens_baseline.py
      phase_screen.py
      turbulence.py
    models/
      d2nn.py
      phase_parameterization.py
    data/
      mnist.py
      cifar10.py
      resolution_targets.py
      masks.py
    train/
      trainer.py
      losses.py
      schedules.py
    eval/
      pcc.py
      grating_period.py
      dynamic_metrics.py
    figures/
      fig1.py
      fig2.py
      fig3.py
      fig4.py
      fig5.py
      fig6.py
      fig7.py
    experiment/
      run_experiment.py
      multirun.py
      registry.py
  configs/
  sweeps/
  runs/
    <run_id>/
      resolved_config.yaml
      seeds.json
      diffuser_registry.csv
      metrics/
      raw_outputs/
      display_outputs/
      figures/
      checkpoints/
      logs/
      stl/
      qa/
```

### Mandatory saved artifacts per run
- resolved YAML config
- exact seeds
- diffuser registry with generation parameters and correlation lengths
- known/new diffuser IDs
- per-diffuser PCC tables
- raw and display versions of all saved images
- figure-panel metadata
- if fabrication is involved: STL files and phase-height mapping metadata

---

## 13. CIFAR-10 extension plan

## 13.1 Non-negotiable physics note
The baseline system is **monochromatic**. Therefore a direct “RGB reconstruction” claim at one wavelength is physically wrong.

### Correct options
1. **CIFAR-10 grayscale** under the same monochromatic THz model  
2. **Multi-pass or multi-wavelength color extension** as a separate research branch

Do not mix these two.

## 13.2 Recommended CIFAR-10 baselines

### Track A: CIFAR-10 grayscale (numerical, same optics)
- convert RGB to luminance:
  \[
  Y = 0.299R + 0.587G + 0.114B
  \]
- resize 32 × 32 to 160 × 160
- zero-pad to 240 × 240
- use the same optical grid and geometry

### Track B: CIFAR-10 binary / silhouette (closest to current THz hardware)
Because the paper’s experimental objects are binary amplitude masks:
- use object masks / silhouettes / strong thresholds
- optionally use halftoning if grayscale physical masks are desired

### Track C: color extension (research branch, not baseline)
Only valid if you adopt:
- multi-wavelength propagation
- sequential channel exposures
- or spectrally multiplexed diffractive design

This is not the same system as the paper.

## 13.3 Loss issue for CIFAR-10
The MNIST loss uses a binary support mask \(\hat h_b\). For CIFAR-10 grayscale images, many background pixels are nonzero, which weakens the meaning of the energy penalty.

### Required resolution
Run at least two CIFAR-10 variants:
1. `cifar10_gray_masked`
   - compute a foreground mask by thresholding / segmentation
   - keep PCC + energy loss
2. `cifar10_gray_pcc_only`
   - drop the energy penalty
   - test whether MNIST sparsity was critical

### Recommendation
Use this ablation as part of the official CIFAR-10 evaluation. Otherwise conclusions will be confounded by the loss definition.

## 13.4 CIFAR-10 metrics
Use:
- PCC
- SSIM
- class-balanced mean PCC
- per-class variance
- failure-case gallery

LPIPS can be reported for numerical natural-image evaluation, but it is not part of the paper baseline and should not replace PCC.

---

## 14. Turbulence-like diffuser extension

## 14.1 What is physically valid here
If you want a “turbulence-like diffuser” in the same software framework, the correct analog is a **thin phase screen** or a **multi-screen split-step model**.

## 14.2 Thin-screen turbulence (recommended first)
Use a modified von Kármán phase screen:
\[
W_\phi(\kappa) = 0.023\, r_0^{-5/3}(\kappa^2+\kappa_0^2)^{-11/6}\exp(-\kappa^2/\kappa_m^2)
\]
where:
- \(r_0\): Fried parameter
- \(\kappa_0 = 2\pi/L_0\): outer-scale cutoff
- \(\kappa_m \propto 1/l_0\): inner-scale cutoff

### Required output variables
- \(r_0\)
- \(L_0\)
- \(l_0\)
- RMS phase
- phase structure function

## 14.3 Important caveat
At the paper’s short 40 mm object-to-diffuser distance, this is best interpreted as a **synthetic phase-screen benchmark**, not a literal atmospheric THz turbulence experiment.

If you want physically meaningful long-path turbulence, increase path length and define a propagation scenario consistent with the wavelength and \(C_n^2\).

## 14.4 Frozen-flow dynamic turbulence
For time-varying turbulence, use Taylor frozen flow:
\[
\phi(x,y,t)=\phi_0(x-v_xt, y-v_yt)
\]

This plugs directly into the dynamic snapshot/raster models in Section 15.

---

## 15. Rotating diffuser inference

This section is critical because it is easy to get wrong physically.

## 15.1 Dynamic diffuser model
Let the static height map be \(D_0(x,y)\). For a rigidly rotating diffuser:
\[
D(x,y,t)=D_0(R_{-\theta(t)}[x-x_c, y-y_c]) 
\]
with:
\[
\theta(t)=\omega t + \theta_0
\]

and
\[
t_D(x,y,t)=\exp\left(j\frac{2\pi\Delta n}{\lambda}D(x,y,t)\right)
\]

## 15.2 Snapshot detector model
If the detector is a 2D camera or an ideal simultaneous sensor:
\[
I_{\text{snapshot}}(x,y;t_0)=\frac{1}{T_{\text{exp}}}\int_{t_0}^{t_0+T_{\text{exp}}}|u(x,y,t)|^2dt
\]

### Key point
The detector averages **intensity**, not complex field.

## 15.3 Raster-scan detector model
The paper’s experimental apparatus is not a snapshot camera; it is a sequential raster scan. Therefore for sample \(k\) measured at \((x_k,y_k)\) and time window \([t_k, t_k+T_{\text{dwell}}]\):
\[
I_{\text{raster}}[k] = \frac{1}{T_{\text{dwell}}}\int_{t_k}^{t_k+T_{\text{dwell}}}|u(x_k,y_k,t)|^2dt
\]

Then reconstruct the final image by placing each scalar sample back onto the scan grid.

### This means:
A rotating diffuser during a scan does **not** produce a simple time-averaged frame. It produces a **space-time stitched output**. This must be modeled separately.

## 15.4 Required rotating-diffuser experiments
Run both modes:
1. `snapshot mode`
2. `raster-scan mode`

### For each mode, sweep:
- angular velocity \(\omega\)
- exposure time / dwell time
- diffuser radius and crop margin
- synchronized vs unsynchronized acquisition

## 15.5 Required rotating-diffuser visualizations
- output montage vs angle
- PCC vs angle
- PCC vs angular velocity
- snapshot vs raster comparison
- temporal average intensity vs single-angle intensity
- failure-case gallery for motion blur / raster distortion

## 15.6 Border-handling rule
Never rotate the diffuser on the same tight square canvas without margin. Use a larger canvas and center crop after rotation; otherwise you will introduce artificial edge wrap and nonphysical interpolation artifacts.

---

## 16. Optional volumetric diffuser extension

The paper explicitly states that the baseline thin-phase model ignores multiple scattering inside a volume. Therefore a volumetric extension is **not** part of baseline reproduction.

If added, implement it as:
- multi-slice phase/amplitude screens
- split-step propagation
- optional absorption term

### Multi-slice model
For slices \(m=1,\dots,M_s\):
\[
u_{m+1}= \mathcal{P}(\Delta z)\left[a_m(x,y)\exp(j\phi_m(x,y))u_m\right]
\]

Use this only as an **extension experiment**, clearly labeled non-baseline.

---

## 17. Figure generation requirements

## 17.1 Reproducibility rules
Each figure script must:
- load a resolved config
- log exact data and diffuser IDs
- save both PNG and vector PDF
- save a JSON sidecar with panel metadata
- save raw arrays used to compose the figure

## 17.2 Figure-specific files
```text
figures/
  fig1a_geometry.pdf
  fig1b_distortion_comparison.pdf
  fig2_known_new_simulation.pdf
  fig3_period_sweep.pdf
  fig4_experiment.pdf
  fig5_memory.pdf
  fig6_conditions.pdf
  fig7_depth.pdf
```

## 17.3 Statistical output tables
For every published figure with quantitative content, also save a CSV:
- `fig3_periods.csv`
- `fig4_periods_experimental.csv`
- `fig5_memory_stats.csv`
- `fig6_pcc_summary.csv`
- `fig7_depth_summary.csv`

---

## 18. Subagents

The project should be split into specialized subagents so that physics, training, evaluation, and figure generation are independently auditable.

## 18.1 Subagent list

| Subagent | Responsibility | Inputs | Outputs | Hard success criterion |
|---|---|---|---|---|
| `GroundingAgent` | Extract exact paper parameters and ambiguities | PDFs, config defaults | validated baseline spec | no undocumented parameter drift |
| `PropagationAgent` | Implement RS and BL-ASM propagators | optics config, fields | propagated fields, QA report | BL-ASM matches RS within tolerance |
| `DiffuserAgent` | Generate, register, and validate diffusers | diffuser config, seeds | phase/height maps, correlation stats | target \(L\) and uniqueness respected |
| `DatasetAgent` | Prepare MNIST/CIFAR/resolution targets and masks | raw datasets | tensors, masks, metadata | deterministic split and preprocessing |
| `TrainingAgent` | Train D2NN models and save checkpoints | model+training config | checkpoints, logs, histories | convergent training with tracked seeds |
| `EvaluationAgent` | Run PCC / period / generalization tests | checkpoints, test sets | metrics tables | exact metric protocol on raw outputs |
| `FigureAgent` | Reproduce Figs. 1–7 and extension figures | raw outputs, metrics | publication-ready figures | figure contracts and labels satisfied |
| `ExperimentAgent` | Handle calibrated THz scans and experimental metadata | raw scans, calibration | linear-scale images, QC report | calibration traceable and reversible |
| `DynamicMediaAgent` | Simulate rotating diffusers and turbulence | dynamic-medium config | time-resolved outputs | snapshot/raster physics handled correctly |
| `FabricationAgent` | Convert trained phase maps to height/STL | phase maps, material params | STL files, height maps | correct phase-height conversion logged |
| `QAAgent` | Audit every run against physics and reproduction contracts | all outputs | pass/fail audit | all required checks green |
| `ReportAgent` | Assemble run summaries and compare against paper | metrics, figures | markdown/PDF summaries | no unsupported claim slips through |

## 18.2 Subagent orchestration order
1. `GroundingAgent`
2. `PropagationAgent`
3. `DiffuserAgent`
4. `DatasetAgent`
5. `TrainingAgent`
6. `EvaluationAgent`
7. `FigureAgent`
8. `DynamicMediaAgent` / `ExperimentAgent`
9. `FabricationAgent`
10. `QAAgent`
11. `ReportAgent`

---

## 19. Skills

Skills are atomic capabilities that the subagents call. Each skill must have a narrow contract and unit tests.

## 19.1 `skill.generate_random_phase_diffuser`
**Purpose:** Create baseline thin random phase diffusers.

**Inputs**
- wavelength
- \(\Delta n\)
- \(\mu, \sigma_0, \sigma\)
- grid size and pitch
- seed
- uniqueness registry

**Outputs**
- height map \(D(x,y)\)
- phase map \(\phi_D(x,y)\)
- transmittance \(t_D(x,y)\)
- estimated correlation length
- uniqueness report

**Must validate**
- correlation length near target
- uniqueness \(\Delta\phi>\pi/2\)

---

## 19.2 `skill.estimate_diffuser_correlation_length`
**Purpose:** Estimate \(L\) from generated phase maps.

**Outputs**
- autocorrelation map
- fitted \(L\)
- batch histogram over a diffuser set

---

## 19.3 `skill.propagate_rs_reference`
**Purpose:** High-fidelity oracle propagation from the paper’s RS formulation.

**Use cases**
- regression tests
- unit validation of BL-ASM
- sanity checks on edge cases

---

## 19.4 `skill.propagate_bl_asm`
**Purpose:** Fast production propagation.

**Inputs**
- complex field
- wavelength
- dx, dy
- z
- pad factor
- bandlimit policy

**Outputs**
- propagated field
- optional energy-conservation summary

**Guardrails**
- no implicit circular wrap
- no silent aliasing
- explicit evanescent policy

---

## 19.5 `skill.forward_d2nn`
**Purpose:** End-to-end object→diffuser→layers→output forward pass.

**Outputs**
- complex field at each layer
- output intensity
- optional intermediate diagnostics

---

## 19.6 `skill.compute_pcc_and_energy_loss`
**Purpose:** Compute the exact training objective.

**Outputs**
- PCC
- energy penalty
- total loss
- per-sample decomposition

---

## 19.7 `skill.build_resolution_targets`
**Purpose:** Generate the line-based amplitude targets used in Figs. 2–4.

**Required periods**
- 7.2
- 8.4
- 9.6
- 10.8
- 12.0 mm

**Implementation**
Use the paper’s three-bar target structure.

---

## 19.8 `skill.quantify_grating_period`
**Purpose:** Estimate resolved period from reconstructed outputs using the same rule as the paper.

**Procedure**
- average along \(y\)
- fit three Gaussians
- compute \(\hat p\)

---

## 19.9 `skill.compose_paper_style_figure`
**Purpose:** Make exact panel layouts and labels that mirror the paper.

**Inputs**
- raw arrays
- display arrays
- panel metadata
- labels
- scale bars

**Outputs**
- figure PDF/PNG
- panel metadata JSON
- raw panel array NPZ

---

## 19.10 `skill.export_phase_to_height_and_stl`
**Purpose:** Convert learned wrapped phase to printable height maps.

**Formula**
If the fabrication material has refractive-index difference \(\Delta n_{\text{mat}}\):
\[
h(x,y)= \frac{\lambda}{2\pi \Delta n_{\text{mat}}}\,\phi(x,y)
\]
after wrapping \(\phi \in [0,2\pi)\).

**Important note**
The accessible paper text does not fully spell out the diffractive-layer material parameter. If the same material is used as the diffuser and experimentally confirmed, \(\Delta n \approx 0.74\) is a reasonable starting point; otherwise measure it.

---

## 19.11 `skill.simulate_rotating_diffuser`
**Purpose:** Generate dynamic snapshot or raster outputs under rigid diffuser rotation.

**Outputs**
- time-resolved outputs
- exposure-averaged outputs
- raster-stitched images
- PCC-vs-speed curves

---

## 19.12 `skill.simulate_turbulence_screen`
**Purpose:** Generate static or dynamic von Kármán phase screens.

**Outputs**
- phase screen
- PSD diagnostics
- \(r_0\), \(L_0\), \(l_0\) metadata
- frozen-flow animation support

---

## 19.13 `skill.run_multirun_sweep`
**Purpose:** Execute parameter sweeps with deterministic run indexing.

**Outputs**
- sweep manifest
- resolved configs
- aggregated metric tables

---

## 19.14 `skill.audit_reproduction_contract`
**Purpose:** Automatically decide whether a run qualifies as a baseline reproduction candidate.

**Checks**
- geometry match
- diffuser stats match
- training schedule match
- raw-vs-display separation
- metric protocol match
- figure contract match

---

## 20. Acceptance criteria

## 20.1 Baseline physics acceptance
A run is not allowed to claim “baseline reproduction” unless all are true:
- geometry matches the paper
- monochromatic coherent scalar model used
- thin phase diffuser baseline respected
- layer grid = 240 × 240, pitch = 0.3 mm
- MNIST resized 28→160 and padded to 240
- raw metrics used
- known/new diffuser logic respected
- epoch-wise diffuser refresh respected

## 20.2 Numerical acceptance
Minimum numerical acceptance targets:
- Fig. 2-like mean PCC on known/new diffusers in the same regime as the paper
- Fig. 3 recovered periods closely track true periods
- Fig. 5 memory trend reproduced
- Fig. 6 no-diffuser improvement reproduced
- Fig. 7 depth advantage reproduced

## 20.3 Experimental acceptance
Minimum experimental acceptance targets:
- Fig. 4 digit PCC in the same broad regime as the paper (~0.37–0.53 across shown panels)
- Fig. 4(c) mean measured periods close to 10.8 mm and 12 mm targets
- known/new diffuser gap remains modest

## 20.4 Extension acceptance
CIFAR-10 or dynamic-diffuser claims are valid only if:
- baseline pipeline passes first
- the extension is clearly labeled as non-baseline
- the detector model is physically correct
- the loss modifications are documented

---

## 21. Known ambiguities and how to resolve them

| Ambiguity | Why it matters | Required resolution |
|---|---|---|
| MNIST 50k split not fully specified | affects exact data order and metrics | freeze first-50k split and log it |
| LR schedule says “iteration” but end-LR suggests epoch decay | changes optimization drastically | use epoch-wise 0.99 decay |
| exact phase parameterization not specified | affects optimization path | choose one explicit parameterization and log it |
| exact example digits/diffuser IDs not specified | affects panel appearance | use deterministic seeded sample selection |
| diffractive-layer material index not fully explicit in accessible text | affects STL height mapping | measure it or document assumption |
| depth-sweep geometry not fully explicit | affects Fig. 7 comparison | keep 2 mm inter-layer and 7 mm output gap; log it |

---

## 22. Recommended execution plan

## Phase 0 — physics verification
- implement RS and BL-ASM
- verify agreement
- verify diffuser statistics
- verify lens baseline

## Phase 1 — numerical baseline
- reproduce Fig. 1(b), Fig. 2, Fig. 3
- reproduce Fig. 5, Fig. 6, Fig. 7
- save all raw arrays and metrics

## Phase 2 — experimental baseline
- calibrate THz scans
- reproduce Fig. 4
- quantify period means/SD

## Phase 3 — baseline robustness
- reproduce Supplementary L-shift test (\(L\approx10\lambda \rightarrow 5\lambda\))
- optional phase-island analysis

## Phase 4 — CIFAR-10
- grayscale numerical
- binary/halftone experimental
- pcc-only vs masked-loss ablation

## Phase 5 — dynamic media
- rotating diffuser snapshot mode
- rotating diffuser raster mode
- frozen-flow turbulence screens

## Phase 6 — advanced research
- multi-slice volumetric scatterers
- multispectral / multi-channel color extension
- learned robustness to dynamic diffusers

---

## 23. Immediate action list

1. Lock the **baseline YAML** exactly.
2. Implement **RS oracle + BL-ASM production** and regression tests.
3. Implement **diffuser registry** with uniqueness and correlation-length checks.
4. Freeze the **MNIST 50k split** and seeds.
5. Reproduce **Fig. 2 / 3 / 5 / 6 / 7 numerically** before touching experiments.
6. Only after baseline passes, start:
   - CIFAR-10 grayscale
   - \(L=5\lambda\) shift tests
   - rotating diffuser snapshot/raster experiments
   - turbulence screens

---

## 24. Bottom-line guardrails

- Do **not** treat rotating-diffuser raster scans as snapshot averages.
- Do **not** claim RGB CIFAR-10 reconstruction under the single-wavelength THz baseline.
- Do **not** compute PCC or grating period on contrast-enhanced images.
- Do **not** resample diffusers every mini-batch if you want Fig. 5.
- Do **not** call a volumetric scatterer experiment a baseline reproduction.
- Do **not** skip padding in FFT propagation.
- Do **not** ignore the LR-schedule ambiguity; resolve it explicitly.
- Do **not** claim “exact paper reproduction” without logging unresolved assumptions.

---

## 25. Reference quantitative checkpoints

Use these as sanity ranges after the baseline is implemented.

### Numerical
- Fig. 2-style simulation PCCs: roughly **0.73–0.80**
- known and new diffuser results should be close
- grating periods should recover true periods across 7.2–12 mm

### Experimental
- Fig. 4-style digit PCCs: roughly **0.37–0.53**
- period means:
  - around **10.851 mm** / **11.233 mm** for the 10.8 mm target
  - around **12.269 mm** / **12.225 mm** for the 12 mm target
  under known / new diffuser groups respectively

### Generalization trends
- `n=10,15,20` should be similar on new diffusers
- `n=1` should overfit more
- no-diffuser case should improve fidelity
- deeper networks should improve PCC

---

## 26. Final recommendation

For the cleanest path:

- First reproduce the baseline with **MNIST + thin random phase diffuser + 4 layers + n in {1,10,15,20}**
- Make **BL-ASM** the main engine, but validate it against **RS**
- Treat **CIFAR-10 grayscale** as the first extension
- Treat **rotating diffuser** as a **time-dependent intensity-detection problem**, not a static image problem
- Treat **turbulence-like diffusers** as **thin or multi-screen phase-screen extensions**, clearly separated from the paper baseline
- Keep every claim tied to a saved config, seed set, diffuser registry, and raw metric table

If this discipline is followed, the project can be both scientifically faithful and extensible without slipping into physically wrong shortcuts.
