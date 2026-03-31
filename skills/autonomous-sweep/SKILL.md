---
name: autonomous-sweep
description: Autonomous D2NN/FD2NN parameter sweep with self-validation and self-healing. Validates physics constraints, runs sweeps, detects failures, auto-recovers, and only surfaces validated results.
---

# Autonomous Sweep with Self-Validation

Run parameter sweeps that automatically validate against physical constraints before launching, self-heal from errors, and only surface results when they pass validation gates.

## Workflow

### Phase 1: Physics Constraint Definition
Define sanity checks for this sweep:
- Energy conservation >95% through each propagation step
- Physically realistic focal lengths (25mm-200mm for lab optics)
- Proper pixel pitch units (no m/mm confusion)
- No aliasing conditions (sampling theorem satisfied)
- Loss values finite and gradients flow correctly

### Phase 2: Pre-Sweep Validation Agent
Spawn a validation sub-agent that:
1. Reads the experiment config
2. Checks ALL physical parameters against constraints
3. If validation fails: fix parameters and re-validate
4. Only proceed when all checks pass

### Phase 3: Self-Healing Sweep Execution
For each sweep configuration:
1. Launch the training run
2. Monitor for failures: BadZipFile, NaN loss, OOM, killed process
3. On failure:
   - Diagnose root cause (concurrent writes? OOM? bad params?)
   - Fix the issue automatically
   - Restart that specific run
4. Log all interventions

### Phase 4: Post-Sweep Analysis Agent
After all runs complete:
1. Collect results from all run directories
2. Generate comparison figures and summary table
3. Validate results against physics (energy budgets, realistic Strehl)
4. Compare against paper baselines with matching hyperparameters
5. Only present results after all validation checks pass

## Self-Healing Rules
| Error | Diagnosis | Auto-Fix |
|-------|-----------|----------|
| BadZipFile | Concurrent data generation | Kill duplicate processes, restart |
| NaN loss | Learning rate too high or bad init | Reduce LR by 10x, restart |
| OOM | Batch size too large | Halve batch size, restart |
| Process killed | System resource limit | Wait 60s, check GPU, restart |
| Energy loss >10% | Wrong propagation pipeline | STOP - flag for human review |

## Output Format
```
=== Autonomous Sweep Report ===
Validation: PASSED (all 6 physics checks green)
Runs: 8/8 completed (2 auto-recovered)
  - roi_256: BadZipFile at epoch 12 -> killed dup process, restarted, completed
  - roi_1024: OOM at epoch 1 -> halved batch 1024->512, completed

| Config | Primary Metric | Baseline | Delta | Physics Valid |
|--------|---------------|----------|-------|--------------|

Best config: ... (exceeds baseline by X%)
Interventions log: /path/to/sweep_interventions.log
```

## Important
- Energy loss >10% at any stage = STOP and ask human
- Never auto-fix propagation pipeline issues (beam reducer vs zoom)
- Maximum 3 auto-recovery attempts per run before escalating
- All interventions logged for reproducibility
