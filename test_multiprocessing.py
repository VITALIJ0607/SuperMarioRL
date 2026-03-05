"""
Test-Script um CPU-Auslastung während Training zu überprüfen
Zeigt ob SubprocVecEnv wirklich alle Cores nutzt
"""

import psutil
import time
import threading
import sys
import os

# Füge Projektverzeichnis zum Path hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import nur make_env
from super_mario_rl_low_spec import make_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Anzahl Environments für Test (6 für 6-Core CPU)
NUM_ENVS = 6

# Monitoring Flag
monitoring = True
cpu_samples = []

def monitor_cpu():
    """Monitore CPU-Auslastung pro Core"""
    global monitoring, cpu_samples
    print("\n" + "="*70)
    print("CPU-MONITORING GESTARTET")
    print("="*70)
    
    while monitoring:
        # CPU-Auslastung pro Core
        per_cpu = psutil.cpu_percent(interval=1, percpu=True)
        cpu_samples.append(per_cpu)
        
        # Live-Ausgabe
        print(f"\rCPU-Kerne: {' | '.join([f'Core {i}: {p:5.1f}%' for i, p in enumerate(per_cpu)])}", end='')
    
    print("\n")

def test_env(env_class, num_envs, name):
    """Teste ein Environment und monitore CPU"""
    global monitoring, cpu_samples
    
    print(f"\n{'='*70}")
    print(f"TEST: {name} mit {num_envs} Environments")
    print(f"{'='*70}\n")
    
    # Reset samples
    cpu_samples = []
    monitoring = True
    
    # Starte CPU-Monitoring in separatem Thread
    monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
    monitor_thread.start()
    
    # Erstelle Environments
    env_id = 'SuperMarioBros-1-1-v0'
    print(f"Erstelle {env_class.__name__}...")
    envs = env_class([make_env(env_id, i) for i in range(num_envs)])
    
    # Lasse Training für 30 Sekunden laufen
    print(f"Sammle {num_envs} Rollouts (dauert ~30 Sekunden)...")
    obs = envs.reset()
    
    start_time = time.time()
    steps = 0
    
    while time.time() - start_time < 30:
        actions = [envs.action_space.sample() for _ in range(num_envs)]
        obs, rewards, dones, infos = envs.step(actions)
        steps += num_envs
    
    elapsed = time.time() - start_time
    
    # Stoppe Monitoring
    monitoring = False
    time.sleep(1.5)  # Warte auf letztes Sample
    
    envs.close()
    
    # Analysiere CPU-Auslastung
    print(f"\n{'='*70}")
    print(f"ERGEBNIS: {name}")
    print(f"{'='*70}")
    print(f"Steps: {steps} in {elapsed:.1f}s")
    print(f"FPS: {steps/elapsed:.1f}")
    
    if cpu_samples:
        import numpy as np
        cpu_array = np.array(cpu_samples)
        avg_per_core = cpu_array.mean(axis=0)
        
        print(f"\nDurchschnittliche CPU-Auslastung pro Core:")
        for i, usage in enumerate(avg_per_core):
            bar = '█' * int(usage / 5)  # Bar chart
            print(f"  Core {i}: {usage:5.1f}% {bar}")
        
        print(f"\nGesamte CPU-Auslastung: {avg_per_core.mean():.1f}%")
        print(f"Aktive Cores (>20%): {sum(avg_per_core > 20)} von {len(avg_per_core)}")
    
    time.sleep(2)
    return avg_per_core if cpu_samples else None

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTIPROCESSING TEST")
    print("="*70)
    print("\nDieser Test vergleicht DummyVecEnv vs. SubprocVecEnv")
    print("und zeigt die CPU-Auslastung pro Core.\n")
    
    num_cores = psutil.cpu_count(logical=True)
    print(f"System: {num_cores} CPU-Cores erkannt")
    
    # Test 1: DummyVecEnv (sequentiell, nur 1 Core)
    print("\n" + "="*70)
    print("TEST 1: DummyVecEnv (OHNE Multiprocessing)")
    print("Erwartung: Nur 1 Core bei ~100%, andere idle")
    print("="*70)
    input("Drücke Enter zum Starten...")
    
    dummy_cpu = test_env(DummyVecEnv, NUM_ENVS, "DummyVecEnv")
    
    # Test 2: SubprocVecEnv (parallel, alle Cores)
    print("\n" + "="*70)
    print("TEST 2: SubprocVecEnv (MIT Multiprocessing)")
    print(f"Erwartung: Alle {NUM_ENVS} Cores bei ~70-90% Auslastung")
    print("="*70)
    input("Drücke Enter zum Starten...")
    
    subproc_cpu = test_env(SubprocVecEnv, NUM_ENVS, "SubprocVecEnv")
    
    # Vergleich
    if dummy_cpu is not None and subproc_cpu is not None:
        import numpy as np
        print("\n" + "="*70)
        print("VERGLEICH")
        print("="*70)
        
        dummy_active = sum(dummy_cpu > 20)
        subproc_active = sum(subproc_cpu > 20)
        
        print(f"\nDummyVecEnv:")
        print(f"  Aktive Cores: {dummy_active}")
        print(f"  Durchschnitt: {dummy_cpu.mean():.1f}%")
        
        print(f"\nSubprocVecEnv:")
        print(f"  Aktive Cores: {subproc_active}")
        print(f"  Durchschnitt: {subproc_cpu.mean():.1f}%")
        
        if subproc_active >= NUM_ENVS:
            print("\n✅ SUCCESS: SubprocVecEnv nutzt alle Cores parallel!")
        else:
            print(f"\n⚠️  WARNING: SubprocVecEnv nutzt nur {subproc_active} von {NUM_ENVS} Cores")
        
        speedup = subproc_cpu.mean() / max(dummy_cpu.mean(), 1)
        print(f"\nTheoretischer Speedup: {speedup:.1f}×")
