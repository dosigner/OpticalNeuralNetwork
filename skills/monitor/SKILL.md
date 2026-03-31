---
name: monitor
description: Check status of running D2NN training/sweep experiments. Reads log files, reports progress, detects errors.
---

# Training Monitor Skill

Quick status check for running experiments.

## Workflow

1. Find all active training/sweep processes:
```bash
ps aux | grep -E "(train|sweep|autoresearch)" | grep -v grep
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
```

2. Read latest log files in the experiment directory

3. For each active run, report:

```
═══ Training Status ═══
| Run | Epoch | Best Loss | Current LR | ETA | GPU Mem | Status |
|-----|-------|-----------|------------|-----|---------|--------|

═══ Errors/Warnings ═══
(any NaN losses, OOM errors, file conflicts)

═══ Completed Runs ═══
(runs that finished since last check)
```

## Quick Commands
```bash
# One-liner status check
tail -1 /root/dj/D2NN/kim2026/autoresearch/*.log 2>/dev/null

# GPU utilization
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

# Find latest results
find /root/dj/D2NN/kim2026/runs -name "results.json" -newer /root/dj/D2NN/kim2026/runs -mmin -60
```

## Notes
- Read logs, don't spawn background agents (they get killed)
- If a process is stuck (no log update >30min), flag it
- For long-running experiments, suggest copy-paste nohup commands rather than running via Bash tool
