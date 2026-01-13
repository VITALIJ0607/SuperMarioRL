#!/bin/bash
# GPU-Monitor während Training

echo "=== GPU Monitoring ==="
echo "Drücke Ctrl+C zum Beenden"
echo ""

while true; do
    clear
    date
    echo ""
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
    echo ""
    echo "Prozesse:"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
    sleep 2
done
