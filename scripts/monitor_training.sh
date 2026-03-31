#!/bin/bash
# Headless training monitor — run with: bash scripts/monitor_training.sh
# Or schedule with cron: */30 * * * * bash /root/dj/D2NN/scripts/monitor_training.sh >> /tmp/d2nn_monitor.log

echo "═══ D2NN Training Monitor — $(date '+%Y-%m-%d %H:%M:%S') ═══"
echo ""

# GPU Status
echo "── GPU ──"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "No GPU found"
echo ""

# Active training processes
echo "── Active Processes ──"
ps aux | grep -E "(train|sweep|autoresearch)" | grep -v grep | awk '{printf "PID:%-8s CPU:%-5s MEM:%-5s CMD:%s\n", $2, $3, $4, $11}' || echo "No active training processes"
echo ""

# Latest log entries
echo "── Latest Logs ──"
for log in /root/dj/D2NN/kim2026/autoresearch/*.log /root/dj/D2NN/kim2026/runs/*/*.log; do
    if [ -f "$log" ]; then
        mod_time=$(stat -c %Y "$log" 2>/dev/null)
        now=$(date +%s)
        age=$(( (now - mod_time) / 60 ))
        if [ $age -lt 120 ]; then  # Only show logs updated in last 2 hours
            echo "  📄 $(basename $log) (${age}min ago):"
            tail -3 "$log" | sed 's/^/    /'
            echo ""
        fi
    fi
done

# Check for errors in recent logs
echo "── Errors (last 2h) ──"
find /root/dj/D2NN/kim2026 -name "*.log" -mmin -120 -exec grep -l -i "error\|nan\|exception\|killed" {} \; 2>/dev/null || echo "No errors found"
echo ""

# Completed runs (results.json created in last 2 hours)
echo "── Recently Completed ──"
find /root/dj/D2NN/kim2026/runs -name "results.json" -mmin -120 2>/dev/null | while read f; do
    echo "  ✅ $(dirname $f | xargs basename)"
done || echo "No recent completions"

echo ""
echo "═══ End Monitor ═══"
