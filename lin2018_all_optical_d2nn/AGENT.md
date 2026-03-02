# D2NN Domain Agent Instructions

## Non-negotiables
1. Use SI units internally (meters, radians).
2. Document tensor shapes in public APIs.
3. Deterministic output for fixed config + seed.
4. Keep physics, training, visualization, and export modules separated.
5. Generate figures via reusable plotting functions.

## Required capabilities
- Angular Spectrum propagation with cached transfer functions
- Trainable phase-only and optional complex modulators
- Detector-energy classification with leakage penalty
- Imaging loss (MSE) and SSIM evaluation
- Height-map export and optional Lumerical integration

## Definition of done
- Type hints and docstrings on each module
- At least one unit test per key module
- CLI path to execute each module
- Example config that runs end to end
