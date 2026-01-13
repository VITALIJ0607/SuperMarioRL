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
    """Holt GPU Stats von nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_util, mem_used, power, temp = result.stdout.strip().split(', ')
        return int(gpu_util), int(mem_used), float(power), int(temp)
    except Exception as e:
        print(f"Fehler beim Abrufen der GPU-Stats: {e}")
        return None, None, None, None

def main():
    print("="*70)
    print("GPU MONITORING LOGGER")
    print("="*70)
    print("Überwacht GPU-Auslastung alle 2 Sekunden")
    print("Drücke Ctrl+C zum Beenden und Statistiken anzuzeigen")
    print("="*70)
    print()
    
    # Ringbuffer für letzte 30 Sekunden
    recent_utils = deque(maxlen=15)  # 15 * 2s = 30s
    
    # Statistiken
    samples = []
    high_util_count = 0  # GPU > 50%
    start_time = time.time()
    
    # Logfile
    logfile = f"gpu_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    try:
        with open(logfile, 'w') as f:
            f.write("Timestamp,GPU_Util_%,Memory_MB,Power_W,Temp_C\n")
            
            while True:
                gpu_util, mem_used, power, temp = get_gpu_stats()
                
                if gpu_util is not None:
                    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                    
                    # Log zu File
                    f.write(f"{timestamp},{gpu_util},{mem_used},{power},{temp}\n")
                    f.flush()
                    
                    # Statistiken
                    samples.append(gpu_util)
                    recent_utils.append(gpu_util)
                    if gpu_util > 50:
                        high_util_count += 1
                    
                    # Anzeige
                    bar = '█' * (gpu_util // 5)  # Balken
                    avg_recent = sum(recent_utils) / len(recent_utils)
                    
                    print(f"{timestamp} | GPU: {gpu_util:3d}% {bar:20s} | "
                          f"Avg(30s): {avg_recent:5.1f}% | "
                          f"VRAM: {mem_used:5d}MB | "
                          f"Power: {power:5.1f}W | "
                          f"Temp: {temp}°C")
                
                time.sleep(2)
                
    except KeyboardInterrupt:
        print("\n")
        print("="*70)
        print("STATISTIKEN")
        print("="*70)
        
        runtime = time.time() - start_time
        runtime_min = runtime / 60
        
        if samples:
            avg_util = sum(samples) / len(samples)
            max_util = max(samples)
            min_util = min(samples)
            high_util_percent = (high_util_count / len(samples)) * 100
            
            print(f"Laufzeit: {runtime_min:.1f} Minuten ({len(samples)} Samples)")
            print(f"Durchschnittliche GPU-Auslastung: {avg_util:.1f}%")
            print(f"Max GPU-Auslastung: {max_util}%")
            print(f"Min GPU-Auslastung: {min_util}%")
            print(f"Zeit mit GPU > 50%: {high_util_percent:.1f}%")
            print(f"\nLogfile gespeichert: {logfile}")
            print()
            
            # Bewertung
            if avg_util < 10:
                print("⚠️  WARNUNG: Sehr niedrige durchschnittliche GPU-Auslastung!")
                print("   Das ist bei PPO normal (90% Rollout auf CPU)")
                print("   Check ob GPU während Training-Phase hochgeht")
            elif avg_util < 25:
                print("✅ GPU-Auslastung OK für PPO (viel CPU-Rollout-Zeit)")
            else:
                print("✅ Sehr gute GPU-Auslastung!")
            
            if high_util_percent < 5:
                print("\n⚠️  GPU geht fast nie über 50%!")
                print("   Mögliche Probleme:")
                print("   - Training läuft tatsächlich auf CPU")
                print("   - n_steps zu hoch (lange Rollouts)")
                print("   - batch_size zu klein")
            elif high_util_percent > 15:
                print("\n✅ GPU wird regelmäßig stark genutzt!")
        
        print("="*70)

if __name__ == "__main__":
    main()
