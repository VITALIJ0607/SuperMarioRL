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
# Bei Single-GPU keine speziellen CUDA-Flags nötig
torch.set_num_threads(6)  # 6-Core CPU: 6 Threads für Training (optimal für 8 Envs)
torch.set_num_interop_threads(2)  # 2 Threads für Parallelität


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
        self.max_x = 0  # Track höchste erreichte X-Position
        self.current_x = 0  # Aktuelle X-Position
        self.stuck_counter = 0  # Zähler für Feststecken
        self.stuck_threshold = 100  # Steps ohne Fortschritt = feststecken
        self.last_x_positions = []  # Historie der letzten X-Positionen
        self.step_count = 0  # Step Counter für Episode
        # Progressive Meilensteine
        self.checkpoint_1000 = False
        self.checkpoint_2000 = False
        self.checkpoint_3000 = False
        
    def reset(self, **kwargs):
        self.max_x = 0
        self.current_x = 0
        self.stuck_counter = 0
        self.last_x_positions = []
        self.step_count = 0
        self.checkpoint_1000 = False
        self.checkpoint_2000 = False
        self.checkpoint_3000 = False
        return self.env.reset()
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.step_count += 1
        
        # Kombiniere Game-Reward MIT X-Progress (nicht entweder/oder!)
        if 'x_pos' in info:
            self.current_x = info['x_pos']
            self.last_x_positions.append(self.current_x)
            
            # Behalte nur die letzten 40 Positionen (ca. 2 Sekunden bei 20 FPS)
            if len(self.last_x_positions) > 40:
                self.last_x_positions.pop(0)
            
            # Nur NEUEN Fortschritt belohnen (verhindert Hin-und-Her-Laufen)
            x_progress = max(info['x_pos'] - self.max_x, 0)
            
            # Fortschritt-Tracking
            if x_progress > 0:
                self.stuck_counter = 0  # Reset bei Fortschritt
                self.max_x = info['x_pos']
            else:
                self.stuck_counter += 1  # Increment wenn kein Fortschritt
            
            # Basis Reward: Game Reward (stark reduziert) + X-Progress
            reward = reward * 0.05 + x_progress * 1.5
            
            # ANTI-STUCK MECHANISMUS: Bestrafung für Stillstand
            # Prüfe ob sich die X-Position in den letzten 40 Frames kaum bewegt hat
            if len(self.last_x_positions) >= 40:
                x_variance = max(self.last_x_positions) - min(self.last_x_positions)
                if x_variance < 10:  # 10 Pixel (erhöht von 5) - weniger strikt
                    reward -= 0.5  # Kleine kontinuierliche Bestrafung
                    
            # Starke Bestrafung bei längerem Feststecken (adaptiv)
            # Threshold wird strenger im Laufe des Trainings
            adaptive_threshold = self.stuck_threshold
            if self.stuck_counter > adaptive_threshold:
                reward -= 1.0  # -1.0 pro Step nach Threshold
                
            # PROGRESSIVE MEILENSTEINE: Belohne wichtige X-Positionen
            if self.current_x > 1000 and not self.checkpoint_1000:
                reward += 50.0
                self.checkpoint_1000 = True
            elif self.current_x > 2000 and not self.checkpoint_2000:
                reward += 75.0
                self.checkpoint_2000 = True
            elif self.current_x > 3000 and not self.checkpoint_3000:
                reward += 100.0
                self.checkpoint_3000 = True
                
            # ZEIT-EFFIZIENZ: Belohne schnellen Fortschritt leicht
            if x_progress > 0 and self.step_count < 1000:
                reward += 0.1  # Bonus für frühen Fortschritt
        
        # Unterscheide zwischen Tod und Timeout
        if done:
            if info.get('flag_get', False):
                # Level geschafft - GROSSER Bonus
                reward += 500.0
                # Extra Bonus für schnelle Completion
                if self.step_count < 400:
                    reward += 200.0  # Sehr schnell!
                elif self.step_count < 800:
                    reward += 100.0  # Schnell
            elif 'time' in info and info['time'] <= 0:
                # TIMEOUT - Starke Bestrafung (Agent hat das Level nicht geschafft)
                reward -= 100.0
            else:
                # Tod (von Gegner, Grube etc.) - Moderate Bestrafung
                reward -= 50.0
                # Extra Bestrafung wenn früh gestorben (< 500 Steps) - abgeschwächt
                if self.step_count < 500:
                    reward -= 10.0  # Reduziert von 20.0 → weniger hart
            
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
        
        # TimeLimit auf 1400 Steps - gibt mehr Spielraum für Erkundung
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1400)
        
        env = GymCompatibilityWrapper(env)  # NACH gym-Wrappern, konvertiert zu neuem API
        # Seed wird jetzt beim reset() gesetzt (gym >= 0.26)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        return env
    return _init


if __name__ == "__main__":
    # ===== HARDWARE-SPEZIFISCHE KONFIGURATION =====
    # CPU: i7 6-Core, RAM: 16GB, GPU: RTX 2070 8GB VRAM
    # OPTIMIERT FÜR LOW-SPEC HARDWARE
    # =============================================
    NUM_ENVS = 8  # 8 Envs für 6-Core + Hyperthreading (optimal für 12 logische Threads)
    TOTAL_TIMESTEPS = 10_000_000  # Reduziert für schnellere Iteration
    SAVE_FREQ = 100_000
    EPISODES = 5000
    N_STEPS = 2048  # 8*2048 = 16384 steps per rollout
    
    # Erstelle parallele Environments
    env_id = 'SuperMarioBros-1-1-v0'  # Spezifisches Level 1-1 (einfachstes Level)
    
    # SubprocVecEnv für echtes Multiprocessing (nutzt Hyperthreading)
    try:
        envs = SubprocVecEnv([make_env(env_id, i) for i in range(NUM_ENVS)])
        print(f"✅ SubprocVecEnv mit {NUM_ENVS} parallelen Prozessen (nutzt 6C/12T optimal)")
    except Exception as e:
        print(f"⚠️  SubprocVecEnv fehlgeschlagen, fallback zu DummyVecEnv: {e}")
        envs = DummyVecEnv([make_env(env_id, i) for i in range(NUM_ENVS)])
    
    # Erstelle Evaluation Environment
    eval_env = DummyVecEnv([make_env(env_id, 0)])
    
    # Learning Rate Schedule: Linear Decay
    # Start bei 3e-5, endet bei 1e-6 (100x kleiner)
    # Verhindert katastrophisches Vergessen in späten Phasen
    def linear_schedule(progress_remaining: float) -> float:
        """
        Learning rate schedule: Linear decay von initial_lr zu final_lr
        
        Args:
            progress_remaining: Float zwischen 1.0 (Start) und 0.0 (Ende)
        
        Returns:
            Current learning rate
        """
        initial_lr = 3e-5  # Start: Schnelles Lernen
        final_lr = 1e-6    # Ende: Feintuning
        return final_lr + (initial_lr - final_lr) * progress_remaining
    
    # Erstelle PPO Model mit optimierten Hyperparametern für Low-Spec
    # Basis: PPO_7 Config (funktioniert nachweislich gut!)
    model = PPO(
        policy='CnnPolicy',
        env=envs,
        n_steps=N_STEPS,                # 2048 steps × 8 envs = 16384 samples
        batch_size=1024,                # ↓ Reduziert für 8GB VRAM (war 2048)
        learning_rate=linear_schedule,  # 🆕 LR Decay: 3e-5 → 1e-6 (verhindert Vergessen)
        gamma=0.98,                     # Kürzerer Horizont - fokussiert auf nahe Rewards
        gae_lambda=0.98,                # Erhöht für bessere Value Estimates
        ent_coef=0.1,                   # ↑ Erhöht für mehr Exploration (war 0.05)
        clip_range=0.2,                 # Standard PPO
        vf_coef=1.0,                    # MAXIMAL für Value Function Learning
        max_grad_norm=0.5,              # Gradient Clipping
        n_epochs=10,                    # Mehr Training aus Daten
        verbose=1,
        tensorboard_log="./ppo_mario_tensorboard/",
        device='cuda',
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Gleiche Netzwerkgröße wie High-Spec
            normalize_images=False
        )
    )
    
    # GPU-Check
    print(f"\n{'='*60}")
    print(f"HARDWARE-KONFIGURATION:")
    print(f"  CPU: i7 6-Core/12-Thread (8 parallele Environments)")
    print(f"  RAM: 16GB")
    print(f"  GPU: RTX 2070 8GB VRAM")
    print(f"\nGPU Status:")
    print(f"  Model device: {model.device}")
    print(f"  Policy auf GPU: {next(model.policy.parameters()).is_cuda}")
    print(f"  Policy device: {next(model.policy.parameters()).device}")
    print(f"{'='*60}\n")
    
    # Callbacks für Checkpoints und Evaluation
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ // NUM_ENVS,
        save_path='./models/',
        name_prefix='ppo_mario_lowspec'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best_model/',
        log_path='./logs/',
        eval_freq=5000 // NUM_ENVS,  # Evaluiere regelmäßig
        deterministic=True,
        render=False,
        n_eval_episodes=5  # 5 Episodes für stabilere Evaluation
    )
    
    # Training starten
    print(f"\n{'='*70}")
    print(f"STARTE TRAINING - LOW-SPEC CONFIG (PPO_7 Basis + LR Decay)")
    print(f"{'='*70}")
    print(f"Environments: {NUM_ENVS} parallel")
    print(f"Gesamte Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Samples pro Rollout: {NUM_ENVS * N_STEPS:,}")
    print(f"Batch Size: 1024 (optimiert für 8GB VRAM)")
    print(f"Learning Rate: 3e-5 → 1e-6 (Linear Decay)")
    print(f"Evaluation alle: {5000 // NUM_ENVS * NUM_ENVS:,} steps")
    print(f"\n⚠️  HINWEIS: Training dauert ~4× länger als auf High-Spec Hardware")
    print(f"   Erwarte ~250k Steps/Stunde (vs. ~1M auf 40-Core)")
    print(f"{'='*70}\n")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Finales Model speichern
    model.save("ppo_mario_lowspec_final")
    print("Training abgeschlossen! Model gespeichert als 'ppo_mario_lowspec_final'")
    
    # Cleanup
    envs.close()
    eval_env.close()
