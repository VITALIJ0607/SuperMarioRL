#!/usr/bin/env python3
"""
GPU Monitoring Logger für Remote Server
Zeichnet GPU-Auslastung über Zeit auf und zeigt Statistiken
"""
import subprocess
import time
import datetime
from collections import deque

def get_gpu_stats():
    """Holt GPU Stats von nvidia-smi für ALLE GPUs"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse alle GPUs
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                gpu_util, mem_used, power, temp = line.split(', ')
                gpus.append((int(gpu_util), int(mem_used), float(power), int(temp)))
        
        return gpus
    except Exception as e:
        print(f"Fehler beim Abrufen der GPU-Stats: {e}")
        return []

def main():
    print("="*70)
    print("GPU MONITORING LOGGER - Multi-GPU Edition")
    print("="*70)
    print("Überwacht GPU-Auslastung alle 2 Sekunden")
    print("Drücke Ctrl+C zum Beenden und Statistiken anzuzeigen")
    print("="*70)
    print()
    
    # Ringbuffer für letzte 30 Sekunden pro GPU
    num_gpus = 0
    recent_utils = {}
    
    # Statistiken pro GPU
    samples = {}
    high_util_count = {}
    start_time = time.time()
    
    # Logfile
    logfile = f"gpu_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    try:
        with open(logfile, 'w') as f:
            # Header wird dynamisch erstellt
            header_written = False
            
            while True:
                gpus = get_gpu_stats()
                
                if gpus:
                    num_gpus = len(gpus)
                    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                    
                    # Initialisiere Strukturen beim ersten Durchlauf
                    if not header_written:
                        header = "Timestamp," + ",".join([f"GPU{i}_Util_%,GPU{i}_Memory_MB,GPU{i}_Power_W,GPU{i}_Temp_C" for i in range(num_gpus)])
                        f.write(header + "\n")
                        header_written = True
                        
                        for i in range(num_gpus):
                            recent_utils[i] = deque(maxlen=15)
                            samples[i] = []
                            high_util_count[i] = 0
                    
                    # Log zu File
                    log_line = timestamp + "," + ",".join([f"{util},{mem},{power},{temp}" for util, mem, power, temp in gpus])
                    f.write(log_line + "\n")
                    f.flush()
                    
                    # Anzeige für alle GPUs
                    print(f"{timestamp}", end="")
                    for i, (gpu_util, mem_used, power, temp) in enumerate(gpus):
                        # Statistiken
                        samples[i].append(gpu_util)
                        recent_utils[i].append(gpu_util)
                        if gpu_util > 50:
                            high_util_count[i] += 1
                        
                        # Balken
                        bar = '█' * (gpu_util // 5)
                        avg_recent = sum(recent_utils[i]) / len(recent_utils[i])
                        
                        print(f"\nGPU{i}: {gpu_util:3d}% {bar:20s} | Avg(30s): {avg_recent:5.1f}% | "
                              f"VRAM: {mem_used:5d}MB | Power: {power:5.1f}W | Temp: {temp}°C", end="")
                    
                    print()  # Neue Zeile nach allen GPUs
                
                time.sleep(2)
                
    except KeyboardInterrupt:
        print("\n")
        print("="*70)
        print("STATISTIKEN")
        print("="*70)
        
        runtime = time.time() - start_time
        runtime_min = runtime / 60
        
        print(f"Laufzeit: {runtime_min:.1f} Minuten\n")
        
        # Statistiken pro GPU
        for i in range(num_gpus):
            if samples[i]:
                avg_util = sum(samples[i]) / len(samples[i])
                max_util = max(samples[i])
                min_util = min(samples[i])
                high_util_percent = (high_util_count[i] / len(samples[i])) * 100
                
                print(f"GPU {i} ({len(samples[i])} Samples):")
                print(f"  Durchschnittliche Auslastung: {avg_util:.1f}%")
                print(f"  Max Auslastung: {max_util}%")
                print(f"  Min Auslastung: {min_util}%")
                print(f"  Zeit mit >50%: {high_util_percent:.1f}%")
                
                # Bewertung
                if avg_util < 10:
                    print("  ⚠️  WARNUNG: Sehr niedrig!")
                elif avg_util < 25:
                    print("  ✅ OK für PPO (viel CPU-Rollout-Zeit)")
                else:
                    print("  ✅ Sehr gute Auslastung!")
                print()
        
        print(f"Logfile gespeichert: {logfile}")
        print("="*70)

if __name__ == "__main__":
    main()
