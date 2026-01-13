"""
GPU Debug Script - Überprüft ob PPO wirklich GPU nutzt
"""
import torch
import sys

print("=" * 60)
print("GPU DEBUG CHECK")
print("=" * 60)

# 1. PyTorch CUDA Check
print(f"\n1. PyTorch CUDA:")
print(f"   Version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Device count: {torch.cuda.device_count()}")
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")

# 2. Test GPU Tensor
print(f"\n2. GPU Tensor Test:")
try:
    test_tensor = torch.randn(1000, 1000).cuda()
    print(f"   ✅ Tensor auf GPU erstellt: {test_tensor.device}")
    print(f"   ✅ GPU funktioniert!")
    del test_tensor
except Exception as e:
    print(f"   ❌ Fehler: {e}")
    sys.exit(1)

# 3. Importiere und teste Stable Baselines3
print(f"\n3. Stable Baselines3 GPU Test:")
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gym
    
    # Einfaches CartPole Environment
    env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
    
    # Model auf CUDA
    print("   Erstelle PPO Model mit device='cuda'...")
    model = PPO('MlpPolicy', env, verbose=0, device='cuda')
    
    # Check device
    print(f"   Model device: {model.device}")
    print(f"   Policy auf GPU: {next(model.policy.parameters()).is_cuda}")
    print(f"   Policy device: {next(model.policy.parameters()).device}")
    
    # Korrigierte Check-Logik
    if next(model.policy.parameters()).is_cuda:
        print(f"   ✅ Model ist auf GPU!")
    else:
        print(f"   ❌ Model ist NICHT auf GPU!")
        sys.exit(1)
    
    # Kurzes Training
    print(f"\n4. Kurzer Trainings-Test:")
    print("   -> Öffne nvidia-smi in anderem Terminal!")
    print("   -> GPU-Util sollte hochgehen...")
    
    model.learn(total_timesteps=5000)
    print(f"   ✅ Training erfolgreich!")
    
    env.close()
    
except Exception as e:
    print(f"   ❌ Fehler: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALLE CHECKS ERFOLGREICH!")
print("GPU sollte funktionieren. Falls nvidia-smi während")
print("des Tests GPU-Util > 50% zeigte, ist alles OK.")
print("=" * 60)
