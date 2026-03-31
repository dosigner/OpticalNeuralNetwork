---
name: sweep
description: Run D2NN/FD2NN parameter sweeps with physics validation. Validates parameters before launch, monitors progress, and generates comparison figures.
---

# Sweep Skill

Run parameter sweeps with built-in physics validation.

## Workflow

### Step 1: Parameter Validation (MANDATORY before any sweep)
Before launching ANY sweep, validate all physical parameters:

```
Checklist:
□ Focal length realistic? (typical: 25mm-200mm for lab optics, NOT 1mm)
□ Pixel pitch units correct? (μm, not m or mm)
□ NA consistent with aperture and focal length?
□ ROI crop size won't cause >10% energy loss?
□ Propagation method correct? (beam reducer vs zoom propagate)
□ No concurrent data generation processes running?
□ Batch size matches paper baseline for fair comparison?
```

Flag any parameter that would cause >10% energy loss, aliasing, or unphysical beam behavior.

### Step 2: Pre-flight Check
```bash
# Check no conflicting processes
ps aux | grep -E "(train|sweep)" | grep -v grep
# Check GPU availability
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
# Check disk space
df -h /root/dj/D2NN
```

### Step 3: Launch
- For short sweeps (<30min): run directly with monitoring
- For long sweeps: provide copy-paste nohup command for user to run manually
- Always log to a timestamped file: `{sweep_name}_{date}.log`

### Step 4: Post-sweep Analysis
After sweep completes:
1. Collect all `results.json` / `test_metrics.json` from run directories
2. Generate comparison table (sorted by primary metric)
3. Generate comparison figures (convergence curves, metric bar charts)
4. Flag any runs with NaN loss or anomalous metrics
5. Compare against paper baselines if available

## Output Format
```
═══ Sweep Validation ═══
✅ Focal length: 25mm (realistic)
✅ Pixel pitch: 37.8μm (correct units)
✅ NA: 0.16 (consistent with f and aperture)
⚠️ ROI crop: 1024 — verify energy budget
✅ Propagation: beam reducer (correct for this setup)
✅ No conflicting processes
✅ GPU available: 22GB free

═══ Sweep Config ═══
Sweep name: ...
Parameters: ...
Estimated time: ...
Command: (copy-paste ready)

═══ Results (after completion) ═══
| Config | Primary Metric | Secondary | Status |
|--------|---------------|-----------|--------|
```

## Notes
- Never launch a sweep without Step 1 validation passing
- If user provides f=1mm, STOP and ask to confirm — this is almost certainly wrong
- BadZipFile error = concurrent write conflict, kill duplicate processes first
