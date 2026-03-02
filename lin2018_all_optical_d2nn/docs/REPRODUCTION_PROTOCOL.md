# Reproduction Protocol

1. Fix seed and deterministic runtime flags.
2. Run config-driven training command.
3. Save resolved config, metrics, checkpoints, and figures in `runs/`.
4. Re-run with same config and verify metrics hash equality.
