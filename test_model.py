import gym
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from super_mario_rl import RewardShapingWrapper, SqueezeObservation, GymCompatibilityWrapper
import numpy as np
import sys


def make_test_env(env_id):
    """
    Erstellt ein einzelnes Environment zum Testen (nicht vektorisiert)
    """
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RewardShapingWrapper(env)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    env = SqueezeObservation(env)
    env = GymCompatibilityWrapper(env)
    return env


def test_model(model_path, num_episodes=5, render=True):
    """
    Testet ein trainiertes Modell
    
    Args:
        model_path: Pfad zum gespeicherten Modell (ohne .zip Endung)
        num_episodes: Anzahl der Test-Episoden
        render: Ob das Spiel angezeigt werden soll
    """
    print(f"Lade Modell von: {model_path}")
    
    try:
        # Modell laden
        model = PPO.load(model_path)
        print("✓ Modell erfolgreich geladen")
    except FileNotFoundError:
        print(f"✗ Fehler: Modell nicht gefunden unter {model_path}")
        print("\nVerfügbare Modelle:")
        print("  - ./ppo_mario_final")
        print("  - ./models/best_model/best_model")
        print("  - ./models/ppo_mario_<steps>_steps")
        sys.exit(1)
    
    # Environment erstellen (einzelnes Env für besseres Rendering)
    env_id = 'SuperMarioBros-1-1-v0'
    env = make_test_env(env_id)
    
    print(f"\nStarte Test mit {num_episodes} Episoden...")
    print("=" * 50)
    
    # Statistiken
    total_rewards = []
    max_x_positions = []
    flags_collected = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        max_x = 0
        
        done = False
        while not done:
            # Vorhersage mit deterministic=True für bessere Performance
            # Füge Batch-Dimension hinzu für das Modell
            obs_batch = np.expand_dims(obs, axis=0)
            action, _states = model.predict(obs_batch, deterministic=True)
            
            # Entferne Batch-Dimension von der Action
            if isinstance(action, np.ndarray):
                action = action[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # X-Position tracken
            if 'x_pos' in info:
                max_x = max(max_x, info['x_pos'])
            
            # Flag-Erfolg checken
            if done and info.get('flag_get', False):
                flags_collected += 1
            
            # Rendering
            if render:
                env.render()
        
        total_rewards.append(episode_reward)
        max_x_positions.append(max_x)
        
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Max X-Position: {max_x}")
        print(f"  Flag erreicht: {'✓' if info.get('flag_get', False) else '✗'}")
        print("-" * 50)
    
    # Zusammenfassung
    print("\n" + "=" * 50)
    print("ZUSAMMENFASSUNG")
    print("=" * 50)
    print(f"Durchschnittlicher Reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"Durchschnittliche Max X-Position: {sum(max_x_positions) / len(max_x_positions):.0f}")
    print(f"Flags erreicht: {flags_collected}/{num_episodes} ({flags_collected/num_episodes*100:.1f}%)")
    print(f"Best X-Position: {max(max_x_positions)}")
    
    env.close()
    print("\nTest abgeschlossen!")


if __name__ == "__main__":
    # Standardmäßig das finale Modell testen
    # Du kannst auch andere Modelle testen:
    # - "./models/best_model/best_model"
    # - "./models/ppo_mario_100000_steps"
    
    MODEL_PATH = "./models_v2/best_model/best_model"
    NUM_EPISODES = 5
    RENDER = True
    
    test_model(MODEL_PATH, NUM_EPISODES, RENDER)
