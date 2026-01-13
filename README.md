# Super Mario Bros Reinforcement Learning mit PPO

Dieses Projekt trainiert einen KI-Agenten, Super Mario Bros mit **Proximal Policy Optimization (PPO)** von Stable-Baselines3 zu spielen.

## 🎮 Überblick

Das Setup verwendet moderne RL-Best-Practices:

- **PPO-Algorithmus** für robustes und stabiles Training
- **Parallele Environments** für schnelleres Lernen
- **CNN-Policy** für visuelle Pixel-Eingaben
- **Custom Reward Shaping** für bessere Lernfortschritte
- **Automatische Checkpoints & Evaluation**

## 📋 Voraussetzungen

### Benötigte Pakete

```bash
# Conda Environment aktivieren
conda activate super_mario_rl

# Stable-Baselines3 installieren (Hauptframework)
pip install stable-baselines3[extra]

# Optional: Tensorboard für Visualisierung
pip install tensorboard

# Bereits installiert sollten sein:
# - gym-super-mario-bros
# - nes-py
# - gym
```

### Systemanforderungen

- **CPU**: Mindestens 4 Kerne (optimal: 8+)
- **RAM**: Mindestens 8 GB
- **GPU**: Optional aber empfohlen (CUDA-fähige NVIDIA GPU)
- **Speicher**: ~5 GB für Models und Logs

## 🏗️ Architektur

### Environment Pipeline

```
SuperMarioBros-v0
    ↓
JoypadSpace (7 Aktionen statt 256)
    ↓
RewardShapingWrapper (Custom Rewards)
    ↓
GrayScaleObservation (RGB → Graustufen)
    ↓
ResizeObservation (240x256 → 84x84)
    ↓
FrameStack (4 Frames für Bewegung)
    ↓
PPO Agent
```

### Wrapper-Details

#### 1. **JoypadSpace**

- Reduziert Action-Space von 256 auf 7 sinnvolle Aktionen
- Verwendet `SIMPLE_MOVEMENT`: [NOOP, rechts, rechts+A, rechts+B, rechts+A+B, A, links]

#### 2. **RewardShapingWrapper** (Custom)

- ✅ **Fortschritts-Belohnung**: +0.1 × Δx (für jeden Pixel nach rechts)
- ✅ **Meilenstein-Bonus**: +1.0 bei neuem x-Rekord
- ⏱️ **Zeit-Penalty**: -0.01 pro Schritt (motiviert schnelles Spielen)
- ☠️ **Death Penalty**: -10.0 bei Tod

#### 3. **GrayScaleObservation**

- Konvertiert RGB zu Graustufen (reduziert Dimensionalität)

#### 4. **ResizeObservation**

- Skaliert auf 84×84 Pixel (Standard für Atari-Games)

#### 5. **FrameStack**

- Stapelt 4 aufeinanderfolgende Frames
- Ermöglicht dem Agenten Bewegungsrichtung zu erkennen

## 🧠 PPO-Konfiguration

### Hyperparameter (optimierte Defaults)

| Parameter       | Wert   | Beschreibung                               |
| --------------- | ------ | ------------------------------------------ |
| `n_steps`       | 256    | Schritte pro Environment vor Policy-Update |
| `batch_size`    | 256    | Mini-Batch-Größe für Training              |
| `learning_rate` | 2.5e-4 | Lernrate (Adam Optimizer)                  |
| `gamma`         | 0.99   | Discount Factor für zukünftige Rewards     |
| `gae_lambda`    | 0.95   | GAE (Generalized Advantage Estimation)     |
| `ent_coef`      | 0.01   | Entropy Coefficient (Exploration)          |
| `clip_range`    | 0.2    | PPO Clipping-Parameter                     |
| `n_epochs`      | 4      | Gradient-Updates pro Rollout               |

### Policy

- **CnnPolicy**: Convolutional Neural Network für Pixel-Inputs
- Architektur: 3 Conv-Layer + 2 Fully-Connected Layers
- Automatische Feature-Extraktion aus Frames

### Parallelisierung

- **SubprocVecEnv**: 8 parallele Environments (separate Prozesse)
- Effektive Batch-Größe: 256 steps × 8 envs = 2048 samples pro Update
- Linearer Speedup durch Multiprocessing

## 🚀 Training starten

### Standard-Training (10M Timesteps)

```bash
python super_mario_rl.py
```

Das Training:

- Läuft mit 8 parallelen Environments
- Erstellt alle 100.000 Steps einen Checkpoint
- Speichert das beste Model basierend auf Evaluation
- Loggt zu TensorBoard

### Training-Output

```
models/               # Checkpoints alle 100k Steps
  ├── ppo_mario_100000_steps.zip
  ├── ppo_mario_200000_steps.zip
  └── best_model/     # Bestes Model nach Evaluation
logs/                 # Evaluation Logs
ppo_mario_tensorboard/ # TensorBoard Logs
ppo_mario_final.zip   # Finales trainiertes Model
```

## 📊 Monitoring mit TensorBoard

```bash
# TensorBoard starten
tensorboard --logdir=./ppo_mario_tensorboard

# Im Browser öffnen: http://localhost:6006
```

### Wichtige Metriken

- **rollout/ep_rew_mean**: Durchschnittliche Episode-Belohnung
- **rollout/ep_len_mean**: Durchschnittliche Episode-Länge
- **train/entropy_loss**: Exploration (sollte langsam fallen)
- **train/policy_loss**: Policy-Optimierung
- **train/value_loss**: Value-Function-Fehler

## 🎯 Training-Erwartungen

### Phase 1: Exploration (0-1M Steps)

- Agent lernt Grundbewegungen
- Viel zufällige Aktionen
- Stirbt häufig

### Phase 2: Lokale Optima (1M-3M Steps)

- Schafft erste Hindernisse
- Lernt Sprungmechanik
- Erreicht erste Pipes/Gegner

### Phase 3: Level-Progress (3M-7M Steps)

- Erreicht zunehmend weiter rechts
- Vermeidet Gegner gezielt
- Nutzt Power-Ups

### Phase 4: Level-Completion (7M+ Steps)

- Schafft komplette Level
- Optimiert Geschwindigkeit
- Hohe Erfolgsrate

## 🔧 Anpassungen & Tuning

### Mehr Exploration?

```python
ent_coef=0.05,  # Erhöhen von 0.01 auf 0.05
```

### Training beschleunigen?

```python
NUM_ENVS = 16,  # Mehr parallele Envs (mehr RAM/CPU nötig)
```

### Andere Level?

```python
env_id = 'SuperMarioBros-1-2-v0'  # World 1-2
env_id = 'SuperMarioBros-2-1-v0'  # World 2-1
```

### Curriculum Learning?

Starte mit einfachen Levels, wechsle später zu schwierigeren:

```python
# Erst World 1-1 trainieren
# Dann Model laden und auf 1-2 weitertrainieren
model = PPO.load("ppo_mario_final", env=new_env)
model.learn(total_timesteps=5_000_000)
```

## 🎮 Trainiertes Model testen

### Model laden und ausführen

```python
from stable_baselines3 import PPO

# Model laden
model = PPO.load("ppo_mario_final")

# Environment erstellen
test_env = DummyVecEnv([make_env('SuperMarioBros-v0', 0)])

# Spielen
obs = test_env.reset()
for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = test_env.step(action)
    test_env.render()
    if dones:
        obs = test_env.reset()
```

### Mit verschiedenen Models experimentieren

```python
# Checkpoint laden
model = PPO.load("models/ppo_mario_500000_steps")

# Bestes Model laden
model = PPO.load("models/best_model/best_model")
```

## 🆘 Troubleshooting

### "GPU not found" / Training langsam

```bash
# CPU-only Installation:
pip install stable-baselines3

# Mit GPU-Support:
pip install stable-baselines3[extra]
# Stelle sicher dass PyTorch mit CUDA installiert ist:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "Out of Memory"

```python
# Reduziere parallele Environments
NUM_ENVS = 4  # statt 8

# Oder reduziere n_steps
n_steps=128  # statt 256
```

### Agent lernt nicht / bleibt stecken

```python
# Erhöhe Exploration
ent_coef=0.05  # statt 0.01

# Passe Reward Shaping an
reward += (info['x_pos'] - self.current_x) * 0.5  # statt 0.1

# Erhöhe Learning Rate
learning_rate=5e-4  # statt 2.5e-4
```

### "Cannot find best model"

Das ist normal am Anfang. Das beste Model wird erst nach der ersten Evaluation gespeichert (nach ~10k Steps).

## 📚 Weiterführende Optimierungen

### 1. Recurrent PPO (LSTM)

Für besseres Timing und Memory:

```python
from sb3_contrib import RecurrentPPO

model = RecurrentPPO('CnnLstmPolicy', env, ...)
```

### 2. Intrinsic Motivation (RND)

Bei sehr sparse Rewards:

```python
# Random Network Distillation für Exploration
# Erfordert Custom Implementation oder sb3-contrib
```

### 3. Curriculum Learning

```python
# Start mit einfachen Levels
# Graduell schwierigere Levels einführen
# Nutze pretrained Models als Basis
```

### 4. Action Masking

```python
# Verhindere ungültige Aktionen in bestimmten Situationen
# Z.B. kein "links" wenn am Level-Start
```

## 📈 Performance-Benchmarks

Auf typischer Hardware (8-Core CPU, RTX 3060):

- **Training-Speed**: ~10-15k Steps/Sekunde (8 Environments)
- **Episode**: ~300-400 Steps (untrainiert), ~1500+ Steps (trainiert)
- **Zeit bis erste Level-Completion**: 3-6 Stunden (5-8M Steps)

## 🔗 Ressourcen

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Gym Super Mario Bros](https://github.com/Kautenja/gym-super-mario-bros)

## 📝 Lizenz

Dieses Projekt verwendet:

- `gym-super-mario-bros` (Lizenz beachten)
- `stable-baselines3` (MIT License)
- Nintendo's Super Mario Bros (nur für Forschung/Bildung)

---

**Viel Erfolg beim Training! 🎮🚀**
