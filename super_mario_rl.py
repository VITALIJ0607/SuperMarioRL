import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import numpy as np


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
        self.current_x = 0
        self.max_x = 0
        
    def reset(self, **kwargs):
        self.current_x = 0
        self.max_x = 0
        # Gibt nur obs zurück (alte gym API)
        return self.env.reset()
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        # Belohnung für Fortschritt in x-Richtung
        if 'x_pos' in info:
            reward += (info['x_pos'] - self.current_x) * 0.1
            self.current_x = info['x_pos']
            
            # Bonus für neuen Rekord
            if info['x_pos'] > self.max_x:
                self.max_x = info['x_pos']
                reward += 1.0
        
        # Kleiner Zeit-Penalty um schnelleres Vorankommen zu fördern
        reward -= 0.001
        
        # Death penalty
        if done and info.get('flag_get', False) == False:
            reward -= 5.0
            
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
        env = GymCompatibilityWrapper(env)  # NACH gym-Wrappern, konvertiert zu neuem API
        # Seed wird jetzt beim reset() gesetzt (gym >= 0.26)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        return env
    return _init


if __name__ == "__main__":
    # Konfiguration für optimale GPU-Nutzung
    NUM_ENVS = 16  # Erhöht für RTX 4090 + 32GB RAM
    TOTAL_TIMESTEPS = 20_000_000  # Trainingsschritte (erhöht für bessere Ergebnisse)
    SAVE_FREQ = 100_000  # Checkpoint-Frequenz
    EPISODES = 5000  # Anzahl der Schritte für Test-Episode
    N_STEPS = 512  # Balance: Genug Daten für GPU, nicht zu lange Rollouts
    
    # Erstelle parallele Environments
    # DummyVecEnv statt SubprocVecEnv wegen numpy-Kompatibilitätsproblemen
    env_id = 'SuperMarioBros-1-1-v0'  # Spezifisches Level 1-1 (einfachstes Level)
    envs = DummyVecEnv([make_env(env_id, i) for i in range(NUM_ENVS)])
    
    # Erstelle Evaluation Environment
    eval_env = DummyVecEnv([make_env(env_id, 0)])
    
    # PPO Model mit optimaler GPU-Auslastung
    model = PPO(
        policy='CnnPolicy',
        env=envs,
        n_steps=N_STEPS,                # 512 steps - Balance zwischen GPU-Nutzung und Stabilität
        batch_size=1024,                # Groß genug für GPU, aber 8 Batches statt 4
        learning_rate=2.5e-4,           # Learning rate
        gamma=0.99,                     # Discount factor
        gae_lambda=0.95,                # GAE parameter
        ent_coef=0.01,                  # Entropy coefficient (reduziert für Stabilität)
        clip_range=0.2,                 # PPO clip range
        n_epochs=10,                    # 10 Epochen - Kompromiss zwischen GPU-Last und Overfitting
        verbose=1,                      # Ausgabe während Training
        tensorboard_log="./ppo_mario_tensorboard/",
        device='cuda',                  # Explizit GPU nutzen (RTX 4090)
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])  # Größeres Netzwerk, aber nicht übertrieben
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
        eval_freq=10000 // NUM_ENVS,
        deterministic=True,
        render=False
    )
    
    # Training starten
    print(f"Starte Training mit {NUM_ENVS} parallelen Environments...")
    print(f"Gesamte Timesteps: {TOTAL_TIMESTEPS:,}")
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
    
    # Optional: Teste das trainierte Model
    print("\nTeste das trainierte Model...")
    test_env = DummyVecEnv([make_env(env_id, 0)])
    obs = test_env.reset()
    
    for _ in range(EPISODES):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
        test_env.render()
        
        if dones[0]:
            obs = test_env.reset()
    
    test_env.close()