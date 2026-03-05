import gym
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import numpy as np
import sys
import time


# Wrapper importieren aus main script
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


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RewardShapingWrapper, self).__init__(env)
        self.max_x = 0
        self.current_x = 0
        self.stuck_counter = 0
        self.stuck_threshold = 100
        self.last_x_positions = []
        self.step_count = 0
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
        
        if 'x_pos' in info:
            self.current_x = info['x_pos']
            self.last_x_positions.append(self.current_x)
            
            if len(self.last_x_positions) > 40:
                self.last_x_positions.pop(0)
            
            x_progress = max(info['x_pos'] - self.max_x, 0)
            
            if x_progress > 0:
                self.stuck_counter = 0
                self.max_x = info['x_pos']
            else:
                self.stuck_counter += 1
            
            # Belohnung für X-Fortschritt
            reward += x_progress * 0.5
            
            # Progressive Checkpoint Belohnungen
            if not self.checkpoint_1000 and info['x_pos'] >= 1000:
                reward += 100.0
                self.checkpoint_1000 = True
            if not self.checkpoint_2000 and info['x_pos'] >= 2000:
                reward += 150.0
                self.checkpoint_2000 = True
            if not self.checkpoint_3000 and info['x_pos'] >= 3000:
                reward += 200.0
                self.checkpoint_3000 = True
            
            # Steckenbleiben bestrafen
            if self.stuck_counter > self.stuck_threshold:
                reward -= 0.5
            
            # Tod bestrafen
            if done and info.get('life', 2) < 2:
                reward -= 50.0
                if self.step_count < 500:
                    reward -= 10.0
            
        return state, reward, done, info


def make_test_env(env_id):
    """Erstellt ein Test-Environment"""
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RewardShapingWrapper(env)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    env = SqueezeObservation(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1400)
    env = GymCompatibilityWrapper(env)
    return env


def analyze_model(model_path, num_episodes=10, render=False):
    """
    Analysiert das trainierte Modell im Detail
    
    Args:
        model_path: Pfad zum Modell
        num_episodes: Anzahl Test-Episoden
        render: Spiel anzeigen (langsam)
    """
    print(f"\n{'='*70}")
    print(f"BEST MODEL ANALYSE")
    print(f"{'='*70}")
    print(f"Modell: {model_path}")
    print(f"Test-Episoden: {num_episodes}")
    print(f"{'='*70}\n")
    
    try:
        model = PPO.load(model_path)
        print("✅ Modell erfolgreich geladen\n")
    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}")
        sys.exit(1)
    
    env = make_test_env('SuperMarioBros-1-1-v0')
    
    # Statistiken sammeln
    episode_stats = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        raw_reward = 0
        steps = 0
        max_x = 0
        deaths = 0
        flag_reached = False
        
        action_counts = {i: 0 for i in range(7)}  # SIMPLE_MOVEMENT hat 7 Actions
        
        done = False
        start_time = time.time()
        
        while not done:
            obs_batch = np.expand_dims(obs, axis=0)
            action, _states = model.predict(obs_batch, deterministic=True)
            
            if isinstance(action, np.ndarray):
                action = action[0]
            
            action_counts[action] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            if 'x_pos' in info:
                max_x = max(max_x, info['x_pos'])
            
            if done:
                if info.get('flag_get', False):
                    flag_reached = True
                if info.get('life', 2) < 2:
                    deaths += 1
            
            steps += 1
            
            if render:
                env.render()
        
        duration = time.time() - start_time
        
        # Statistik speichern
        stats = {
            'episode': episode + 1,
            'reward': episode_reward,
            'steps': steps,
            'max_x': max_x,
            'flag': flag_reached,
            'died': deaths > 0,
            'duration': duration,
            'actions': action_counts
        }
        episode_stats.append(stats)
        
        # Progress
        status = "🏁 FLAG!" if flag_reached else ("💀 DIED" if deaths > 0 else "⏱️ TIMEOUT")
        print(f"Episode {episode + 1:2d}/{num_episodes}: {status} | Reward: {episode_reward:7.1f} | Steps: {steps:4d} | Max X: {max_x:4d}")
    
    env.close()
    
    # Zusammenfassung
    print(f"\n{'='*70}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*70}")
    
    rewards = [s['reward'] for s in episode_stats]
    steps_list = [s['steps'] for s in episode_stats]
    max_xs = [s['max_x'] for s in episode_stats]
    flags = sum(1 for s in episode_stats if s['flag'])
    deaths = sum(1 for s in episode_stats if s['died'])
    timeouts = num_episodes - flags - deaths
    
    print(f"\n📊 Performance:")
    print(f"  Mean Reward:  {np.mean(rewards):7.1f} ± {np.std(rewards):6.1f}")
    print(f"  Mean Steps:   {np.mean(steps_list):7.1f} ± {np.std(steps_list):6.1f}")
    print(f"  Mean Max X:   {np.mean(max_xs):7.1f} ± {np.std(max_xs):6.1f}")
    
    print(f"\n🎯 Erfolgsrate:")
    print(f"  Flags erreicht: {flags}/{num_episodes} ({flags/num_episodes*100:.1f}%)")
    print(f"  Gestorben:      {deaths}/{num_episodes} ({deaths/num_episodes*100:.1f}%)")
    print(f"  Timeouts:       {timeouts}/{num_episodes} ({timeouts/num_episodes*100:.1f}%)")
    
    # Action Distribution
    print(f"\n🎮 Action Distribution (Durchschnitt über alle Episodes):")
    action_names = ['NOOP', 'RIGHT', 'RIGHT+A', 'RIGHT+B', 'RIGHT+A+B', 'A', 'LEFT']
    total_actions = sum(episode_stats[0]['actions'].values())
    
    all_actions = {i: 0 for i in range(7)}
    for stats in episode_stats:
        for action, count in stats['actions'].items():
            all_actions[action] += count
    
    total = sum(all_actions.values())
    for action, name in enumerate(action_names):
        count = all_actions[action]
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {action} {name:12s}: {pct:5.1f}% {bar}")
    
    # Diagnose
    print(f"\n🔍 DIAGNOSE:")
    avg_x = np.mean(max_xs)
    if flags > 0:
        print(f"  ✅ Agent schafft das Level! ({flags}/{num_episodes} mal)")
    elif avg_x < 500:
        print(f"  ❌ Agent kommt nicht weit (avg X: {avg_x:.0f})")
        print(f"     → Stirbt sehr früh oder bewegt sich kaum")
    elif avg_x < 1500:
        print(f"  ⚠️  Agent kommt zur Mitte (avg X: {avg_x:.0f})")
        print(f"     → Stirbt bei ersten Hindernissen")
    elif avg_x < 3000:
        print(f"  ⚠️  Agent kommt weit (avg X: {avg_x:.0f})")
        print(f"     → Schafft es fast, aber stirbt vor dem Ziel")
    else:
        print(f"  ✅ Agent erreicht fast immer das Ziel (avg X: {avg_x:.0f})")
    
    if np.mean(steps_list) < 150:
        print(f"  ⚠️  Sehr kurze Episodes ({np.mean(steps_list):.0f} Steps)")
        print(f"     → Agent stirbt vermutlich früh")
    
    print(f"\n{'='*70}\n")
    
    return episode_stats


if __name__ == "__main__":
    # Test Best Model
    model_path = "./models/best_model/best_model"
    
    # Ausführliche Analyse ohne Rendering (schneller)
    stats = analyze_model(model_path, num_episodes=10, render=False)
    
    # Optional: Ein paar Episodes mit Rendering zeigen
    # print("\nZeige 3 Episodes mit Rendering...")
    # analyze_model(model_path, num_episodes=3, render=True)
