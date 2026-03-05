import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import numpy as np
import os
import torch

# GPU-Konfiguration für maximale Auslastung
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Nutze primäre GPU (beide verfügbar: 0,1)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchrones Execution
torch.set_num_threads(32)  # 40 CPU-Cores: 32 für optimale Performance
torch.set_num_interop_threads(8)  # Parallelität für Ops


# Wrapper um überflüssige Dimensionen zu entfernen
class SqueezeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Passe observation_space an - entferne letzte Dimension wenn sie 1 ist
        old_shape = env.observation_space.shape
        if old_shape[-1] == 1:
            new_shape = old_shape[:-1]
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=new_shape,
                dtype=env.observation_space.dtype
            )
    
    def observation(self, observation):
        return np.squeeze(observation)


# Compatibility Wrapper für alte gym Environments die kein seed in reset() akzeptieren
class GymCompatibilityWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        # Entferne seed und options aus kwargs da alte Envs das nicht unterstützen
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        obs = self.env.reset()
        # Alte gym Envs geben nur obs zurück, neue erwarten (obs, info)
        return obs, {}
    
    def step(self, action):
        # Alte gym: (obs, reward, done, info)
        obs, reward, done, info = self.env.step(action)
        # Neue gym: (obs, reward, terminated, truncated, info)
        # done wird zu terminated, truncated ist False
        return obs, reward, done, False, info


# Custom Reward Wrapper für Fortschritt in x-Richtung
class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RewardShapingWrapper, self).__init__(env)
        self.prev_x = 0
        
    def reset(self, **kwargs):
        self.prev_x = 0
        return self.env.reset()
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        # MAXIMAL VEREINFACHT: Nur rohe X-Position als Reward!
        if 'x_pos' in info:
            # Belohnung = Fortschritt seit letztem Step
            x_progress = (info['x_pos'] - self.prev_x)
            reward = x_progress * 1.0  # VERSTÄRKTES Signal für klares Lernen!
            self.prev_x = info['x_pos']
        
        # Großer Bonus bei Level-Completion
        if done and info.get('flag_get', False):
            reward += 100.0
            
        return state, reward, done, info


def make_env(env_id, rank, seed=0):
    """
    Utility function für parallele Environments
    """
    def _init():
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = RewardShapingWrapper(env)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=4)
        env = SqueezeObservation(env)  # NACH FrameStack! Entfernt (4,84,84,1) -> (4,84,84)
        
        # WICHTIG: Max Episode Length begrenzen (verhindert 8019-step Timeouts)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=2000)  # ~100 Sekunden
        
        env = GymCompatibilityWrapper(env)  # NACH gym-Wrappern, konvertiert zu neuem API
        # Seed wird jetzt beim reset() gesetzt (gym >= 0.26)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        return env
    return _init


if __name__ == "__main__":
    # ===== HARDWARE-SPEZIFISCHE KONFIGURATION =====
    # CPU: 40 Cores, RAM: 196GB, GPU: 2x RTX Pro 6000 Blackwell 96GB
    # =============================================
    NUM_ENVS = 32  # Optimal für 40 CPU Cores (32 parallele Envs)
    TOTAL_TIMESTEPS = 20_000_000
    SAVE_FREQ = 100_000
    EPISODES = 5000
    N_STEPS = 2048  # 32*2048 = 65536 steps per rollout - mehr diverse Samples!
    
    # Erstelle parallele Environments
    # SubprocVecEnv: Nutzt alle CPU-Cores für echte Parallelisierung
    env_id = 'SuperMarioBros-1-1-v0'  # Spezifisches Level 1-1 (einfachstes Level)
    
    # Verwende SubprocVecEnv für bessere Parallelisierung
    try:
        envs = SubprocVecEnv([make_env(env_id, i) for i in range(NUM_ENVS)])
        print(f"✅ SubprocVecEnv lädt... (nutzt mehrere CPU-Cores)")
    except Exception as e:
        print(f"⚠️  SubprocVecEnv fehlgeschlagen, fallback zu DummyVecEnv: {e}")
        envs = DummyVecEnv([make_env(env_id, i) for i in range(NUM_ENVS)])
    
    # Erstelle Evaluation Environment
    eval_env = DummyVecEnv([make_env(env_id, 0)])
    
    # Erstelle PPO Model mit optimierten Hyperparametern
    # PPO_5: Fix für Explained Variance = 0% Problem
    model = PPO(
        policy='CnnPolicy',
        env=envs,
        n_steps=N_STEPS,                # 2048 steps × 32 envs = 65536 samples!
        batch_size=2048,                # Optimal für Stabilität (2048-4096 ist sweet spot)
        learning_rate=1e-4,             # ↓↓ DEUTLICH niedriger (war 2.5e-4) - vorsichtiges Lernen!
        gamma=0.98,                     # ↓ Kürzerer Horizont (war 0.99) - fokussiert auf nahe Rewards
        gae_lambda=0.98,                # ↑ Erhöht für bessere Value Estimates (war 0.95)
        ent_coef=0.05,                  # ↑↑ MEHR Exploration (war 0.01) - Agent soll mehr probieren!
        clip_range=0.2,                 # Standard PPO
        vf_coef=1.0,                    # ↑↑ MAXIMAL für Value Function Learning!
        max_grad_norm=0.5,              # Gradient Clipping
        n_epochs=10,                    # ↑↑ Mehr Training aus Daten (war 4)
        verbose=1,
        tensorboard_log="./ppo_mario_tensorboard/",
        device='cuda',
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            normalize_images=False
        )
    )
    
    # GPU-Check
    print(f"\n{'='*60}")
    print(f"GPU Status:")
    print(f"  Model device: {model.device}")
    print(f"  Policy auf GPU: {next(model.policy.parameters()).is_cuda}")
    print(f"  Policy device: {next(model.policy.parameters()).device}")
    print(f"{'='*60}\n")
    
    # Callbacks für Checkpoints und Evaluation
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ // NUM_ENVS,
        save_path='./models/',
        name_prefix='ppo_mario'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best_model/',
        log_path='./logs/',
        eval_freq=5000 // NUM_ENVS,  # ↑ Früher evaluieren (war 10000, jetzt alle ~160k steps)
        deterministic=True,
        render=False,
        n_eval_episodes=5  # 5 Episodes für stabilere Evaluation
    )
    
    # Training starten
    print(f"\n{'='*70}")
    print(f"STARTE TRAINING PPO_3")
    print(f"{'='*70}")
    print(f"Environments: {NUM_ENVS} parallel")
    print(f"Gesamte Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Samples pro Rollout: {NUM_ENVS * N_STEPS:,}")
    print(f"Evaluation alle: {5000 // NUM_ENVS * NUM_ENVS:,} steps")
    print(f"{'='*70}\n")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Finales Model speichern
    model.save("ppo_mario_final")
    print("Training abgeschlossen! Model gespeichert als 'ppo_mario_final'")
    
    # Cleanup
    envs.close()
    eval_env.close()