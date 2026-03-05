import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import numpy as np
import torch

# GPU-Konfiguration für RTX 2070 8GB
torch.set_num_threads(6)
torch.set_num_interop_threads(2)


class SqueezeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
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


class GymCompatibilityWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        obs = self.env.reset()
        return obs, {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info


class ImprovedRewardShaping(gym.Wrapper):
    """
    VERBESSERTES REWARD SHAPING V2
    
    Hauptprobleme des alten Systems:
    1. Agent steckt bei X=315 fest (nur RIGHT+B)
    2. Zu wenig Exploration
    3. Zu harte Strafen für kleine Fehler
    
    Neue Strategie:
    1. Deutlich höhere Belohnung für X-Fortschritt
    2. Weniger Strafen, mehr Anreize
    3. Sanftere Anti-Stuck-Mechanismen
    4. Bessere Checkpoint-Belohnungen
    """
    def __init__(self, env):
        super(ImprovedRewardShaping, self).__init__(env)
        self.max_x = 0
        self.current_x = 0
        self.step_count = 0
        self.last_x_positions = []
        self.stuck_counter = 0
        # Checkpoints
        self.checkpoints = {
            500: False,
            1000: False,
            1500: False,
            2000: False,
            2500: False,
            3000: False,
            3200: False  # Nahe Ziel
        }
        
    def reset(self, **kwargs):
        self.max_x = 0
        self.current_x = 0
        self.step_count = 0
        self.last_x_positions = []
        self.stuck_counter = 0
        self.checkpoints = {k: False for k in self.checkpoints.keys()}
        return self.env.reset()
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.step_count += 1
        
        # Ursprünglicher Reward ignorieren (ist meist 0 oder nutzlos)
        reward = 0.0
        
        if 'x_pos' in info:
            self.current_x = info['x_pos']
            
            # ✅ HAUPTBELOHNUNG: X-Fortschritt (stark erhöht!)
            x_progress = max(info['x_pos'] - self.max_x, 0)
            if x_progress > 0:
                reward += x_progress * 2.0  # ERHÖHT von 1.5 → 2.0
                self.max_x = info['x_pos']
                self.stuck_counter = 0
            else:
                self.stuck_counter += 1
            
            # Track letzten Positionen
            self.last_x_positions.append(self.current_x)
            if len(self.last_x_positions) > 50:  # Längeres Fenster (war 40)
                self.last_x_positions.pop(0)
            
            # ✅ PROGRESSIVE CHECKPOINTS (häufiger und höher belohnt)
            for checkpoint_x, reached in self.checkpoints.items():
                if not reached and self.current_x >= checkpoint_x:
                    # Belohnung steigt mit Distanz
                    bonus = 50 + (checkpoint_x // 500) * 25
                    reward += bonus
                    self.checkpoints[checkpoint_x] = True
                    print(f"🎯 Checkpoint {checkpoint_x} erreicht! Bonus: +{bonus}")
            
            # ⚠️ SANFTE Anti-Stuck Mechanik (weniger hart)
            if len(self.last_x_positions) >= 50:
                x_variance = max(self.last_x_positions) - min(self.last_x_positions)
                if x_variance < 20:  # Erhöht von 10 → 20 (toleranter)
                    reward -= 0.2  # Reduziert von 0.5 → 0.2
            
            # Nur bei SEHR langem Feststecken bestrafen
            if self.stuck_counter > 150:  # Erhöht von 100 → 150
                reward -= 0.5  # Reduziert von 1.0 → 0.5
            
            # ✅ ZEITBONUS: Schneller Fortschritt wird belohnt
            if x_progress > 5:  # Signifikante Bewegung
                reward += 0.2
        
        # ✅ ENDBELOHNUNGEN / STRAFEN
        if done:
            if info.get('flag_get', False):
                # 🏁 LEVEL GESCHAFFT - Riesiger Bonus!
                reward += 1000.0
                # Extra für Geschwindigkeit
                if self.step_count < 300:
                    reward += 500.0
                elif self.step_count < 600:
                    reward += 250.0
                print(f"🏁 LEVEL GESCHAFFT in {self.step_count} Steps!")
            
            elif 'life' in info and info['life'] < 2:
                # 💀 TOD - Moderate Strafe basierend auf Fortschritt
                # Weniger hart wenn Agent schon weit kam
                death_penalty = max(30.0 - (self.max_x / 100), 10.0)
                reward -= death_penalty
            
            elif 'time' in info and info['time'] <= 0:
                # ⏱️ TIMEOUT - Leichte Strafe (Agent hat wenigstens versucht)
                reward -= 20.0
        
        return state, reward, done, info


def make_env(env_id, rank, seed=0):
    """Utility function für parallele Environments"""
    def _init():
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = ImprovedRewardShaping(env)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=4)
        env = SqueezeObservation(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1400)
        env = GymCompatibilityWrapper(env)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        return env
    return _init


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"VERBESSERTE PPO KONFIGURATION V2")
    print(f"{'='*70}")
    print(f"Basierend auf Analyse des failed Run:")
    print(f"  - Agent steckte bei X=315 fest")
    print(f"  - Nutzte nur RIGHT+B (keine Exploration)")
    print(f"  - Clip Fraction = 0 (kein Learning)")
    print(f"\nVerbesserungen:")
    print(f"  ✅ Höhere Learning Rate (5e-5 → 1e-5)")
    print(f"  ✅ Mehr Entropy für Exploration (0.1 → 0.2)")
    print(f"  ✅ Besseres Reward Shaping (2x X-Progress)")
    print(f"  ✅ Sanftere Strafen, häufigere Checkpoints")
    print(f"  ✅ Kleinere Batch Size für mehr Updates")
    print(f"{'='*70}\n")
    
    # ===== KONFIGURATION =====
    NUM_ENVS = 8
    TOTAL_TIMESTEPS = 10_000_000
    SAVE_FREQ = 100_000
    N_STEPS = 2048
    
    env_id = 'SuperMarioBros-1-1-v0'
    
    # Erstelle parallele Environments
    try:
        envs = SubprocVecEnv([make_env(env_id, i) for i in range(NUM_ENVS)])
        print(f"✅ SubprocVecEnv mit {NUM_ENVS} parallelen Prozessen\n")
    except Exception as e:
        print(f"⚠️  SubprocVecEnv fehlgeschlagen, fallback zu DummyVecEnv: {e}\n")
        envs = DummyVecEnv([make_env(env_id, i) for i in range(NUM_ENVS)])
    
    # Evaluation Environment
    eval_env = DummyVecEnv([make_env(env_id, 0)])
    
    # ✅ VERBESSERTER LEARNING RATE SCHEDULE
    def improved_lr_schedule(progress_remaining: float) -> float:
        """
        Höhere initiale LR, langsamerer Decay
        Verhindert zu frühes Einfrieren der Policy
        """
        initial_lr = 1e-4   # ERHÖHT von 3e-5
        final_lr = 5e-6     # ERHÖHT von 1e-6
        return final_lr + (initial_lr - final_lr) * progress_remaining
    
    # ✅ OPTIMIERTES PPO MODEL
    model = PPO(
        policy='CnnPolicy',
        env=envs,
        n_steps=N_STEPS,
        batch_size=512,                      # ↓ REDUZIERT von 1024 → mehr Updates
        learning_rate=improved_lr_schedule,  # ✅ Höhere LR
        gamma=0.99,                          # ↑ ERHÖHT von 0.98 → längerer Horizont
        gae_lambda=0.95,                     # Standard GAE
        ent_coef=0.2,                        # ✅ VERDOPPELT von 0.1 → mehr Exploration
        clip_range=0.2,
        vf_coef=0.5,                         # ↓ REDUZIERT von 1.0 → Policy wichtiger
        max_grad_norm=0.5,
        n_epochs=10,
        verbose=1,
        tensorboard_log="./ppo_mario_tensorboard_v2/",
        device='cuda',
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            normalize_images=False,
            # ✅ Aktivierungsfunktion optimiert
            activation_fn=torch.nn.ReLU,
        )
    )
    
    # GPU-Check
    print(f"\n{'='*60}")
    print(f"HARDWARE-KONFIGURATION:")
    print(f"  CPU: i7 6-Core/12-Thread")
    print(f"  RAM: 16GB")
    print(f"  GPU: RTX 2070 8GB VRAM")
    print(f"\nGPU Status:")
    print(f"  Model device: {model.device}")
    print(f"  Policy auf GPU: {next(model.policy.parameters()).is_cuda}")
    print(f"{'='*60}\n")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ // NUM_ENVS,
        save_path='./models_v2/',
        name_prefix='ppo_mario_v2'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models_v2/best_model/',
        log_path='./logs_v2/',
        eval_freq=5000 // NUM_ENVS,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Training starten
    print(f"\n{'='*70}")
    print(f"STARTE VERBESSERTES TRAINING V2")
    print(f"{'='*70}")
    print(f"Environments: {NUM_ENVS} parallel")
    print(f"Gesamte Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Samples pro Rollout: {NUM_ENVS * N_STEPS:,}")
    print(f"Batch Size: 512 (mehr Updates!)")
    print(f"Learning Rate: 1e-4 → 5e-6 (höher als vorher)")
    print(f"Entropy Coef: 0.2 (mehr Exploration)")
    print(f"Evaluation alle: {5000 // NUM_ENVS * NUM_ENVS:,} steps")
    print(f"{'='*70}\n")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Finales Model speichern
    model.save("ppo_mario_v2_final")
    print("\n" + "="*70)
    print("✅ Training abgeschlossen!")
    print(f"Model gespeichert als 'ppo_mario_v2_final'")
    print("="*70)
    
    # Cleanup
    envs.close()
    eval_env.close()
