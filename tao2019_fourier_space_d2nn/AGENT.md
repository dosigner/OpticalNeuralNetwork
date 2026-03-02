# AGENT — tao2019 F-D2NN Domain Rules

## Non-negotiable
- Use centered FFT convention consistently.
- Keep all physics lengths in meters.
- Apply gamma flip in saliency loss by default.
- Use config snapshots and deterministic seeds for every run.

## Physics checks
- Validate `ifft2c(fft2c(u)) ~= u`.
- Validate double FFT gamma inversion behavior.
- Mask evanescent components by default in ASM.
- Apply NA cutoff when configured.

## Model checks
- Phase-only modulation: `u * exp(i*phi)`.
- Classification default phase range: `0~pi`.
- Saliency default phase range: `0~2pi`.
- SBN: `u * exp(i * phi_max * I/(1+I))`.

## Output requirements
- Save metrics, checkpoints, config, and figures in each run dir.
