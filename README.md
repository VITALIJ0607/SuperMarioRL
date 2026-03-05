# Super Mario Bros Reinforcement Learning mit PPO

Dieses Projekt trainiert einen KI-Agenten, Super Mario Bros mit **Proximal Policy Optimization (PPO)** von Stable-Baselines3 zu spielen.

## 🎮 Überblick

Das Setup verwendet moderne RL-Best-Practices mit **Low-Spec Optimierungen**:

- **PPO-Algorithmus** für robustes und stabiles Training
- **Parallele Environments** für schnelleres Lernen (8 Envs für 6-Core CPUs)
- **CNN-Policy** für visuelle Pixel-Eingaben
- **Advanced Reward Shaping** mit progressiven Checkpoints und Anti-Stuck-Mechanismus
- **Linear Learning Rate Decay** zur Vermeidung katastrophischen Vergessens
- **Automatische Checkpoints & Evaluation**
- **Optimiert für Low-Spec Hardware** (6-Core CPU, 8GB GPU VRAM)

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

**Optimiert für Low-Spec Hardware:**

- **CPU**: Mindestens 6 Kerne (i7 oder besser, nutzt Hyperthreading)
- **RAM**: Mindestens 16 GB
- **GPU**: RTX 2070 8GB VRAM oder besser (CUDA-fähige NVIDIA GPU)
- **Speicher**: ~5 GB für Models und Logs

**Hinweis:** Training ist ~4× langsamer als auf High-Spec Hardware (40-Core, 96GB GPU), aber vollständig funktionsfähig. Erwarte ~250k Steps/Stunde.

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

#### 2. **RewardShapingWrapper** (Custom - Advanced)

**Basis-Belohnungen:**

- ✅ **X-Progress Reward**: +1.5 × Δx (nur für NEUE Fortschritte, verhindert Hin-und-Her)
- 🎮 **Game Reward**: Original × 0.05 (stark reduziert, X-Progress dominiert)

**Progressive Meilensteine:**

- 🏁 **Checkpoint 1000**: +50.0 beim ersten Erreichen von x=1000
- 🏁 **Checkpoint 2000**: +75.0 beim ersten Erreichen von x=2000
- 🏁 **Checkpoint 3000**: +100.0 beim ersten Erreichen von x=3000

**Anti-Stuck-Mechanismus:**

- 🚫 **Stillstand-Bestrafung**: -0.5 wenn X-Variance < 10 Pixel in 40 Frames
- ⏸️ **Feststeck-Penalty**: -1.0 pro Step nach 100 Steps ohne Fortschritt

**Zeit-Effizienz:**

- ⚡ **Früher Fortschritt**: +0.1 Bonus für X-Progress in ersten 1000 Steps

**Episode-Ende:**

- 🎉 **Level Completion**: +500.0 (+ bis zu +200.0 für schnelle Completion)
- ⏱️ **Timeout**: -100.0 (Level nicht geschafft)
- ☠️ **Tod**: -50.0 (-10.0 extra wenn früh gestorben < 500 Steps)

#### 3. **GrayScaleObservation**

- Konvertiert RGB zu Graustufen (reduziert Dimensionalität)

#### 4. **ResizeObservation**

- Skaliert auf 84×84 Pixel (Standard für Atari-Games)

#### 5. **FrameStack**

- Stapelt 4 aufeinanderfolgende Frames
- Ermöglicht dem Agenten Bewegungsrichtung zu erkennen

## 🧠 PPO-Konfiguration

### Hyperparameter (PPO_7 Basis + Low-Spec Optimierungen)

| Parameter       | Wert                 | Beschreibung                                   |
| --------------- | -------------------- | ---------------------------------------------- |
| `n_steps`       | 2048                 | Schritte pro Environment vor Policy-Update     |
| `batch_size`    | 1024                 | Mini-Batch-Größe (reduziert für 8GB VRAM)      |
| `learning_rate` | 3e-5 → 1e-6 (Linear) | **Learning Rate Decay** (verhindert Vergessen) |
| `gamma`         | 0.98                 | Kürzerer Horizont für nahe Rewards             |
| `gae_lambda`    | 0.98                 | Erhöht für bessere Value Estimates             |
| `ent_coef`      | 0.1                  | Hohe Exploration (war 0.01)                    |
| `clip_range`    | 0.2                  | Standard PPO Clipping                          |
| `vf_coef`       | 1.0                  | Maximal für Value Function Learning            |
| `n_epochs`      | 10                   | Mehr Training aus Daten (war 4)                |
| `max_grad_norm` | 0.5                  | Gradient Clipping                              |

### Learning Rate Schedule 🆕

**Linear Decay für Stabilität:**

```python
def linear_schedule(progress_remaining: float) -> float:
    initial_lr = 3e-5  # Start: Schnelles Lernen
    final_lr = 1e-6    # Ende: Feintuning
    return final_lr + (initial_lr - final_lr) * progress_remaining
```

- **Startet bei 3e-5**: Ermöglicht schnelles Lernen zu Beginn
- **Endet bei 1e-6**: Feines Tuning ohne katastrophisches Vergessen
- **Verhindert**: Dass der Agent in späten Phasen bereits Gelerntes vergisst

### Policy

- **CnnPolicy**: Convolutional Neural Network für Pixel-Inputs
- Architektur: [256, 256] für Policy und Value Function
- Automatische Feature-Extraktion aus Frames
- Optimiert für 8GB VRAM

### Parallelisierung (Low-Spec Optimiert)

- **SubprocVecEnv**: 8 parallele Environments (nutzt 6-Core + Hyperthreading optimal)
- Effektive Batch-Größe: 2048 steps × 8 envs = 16384 samples pro Rollout
- Episode TimeLimit: 1400 Steps (gibt mehr Spielraum für Erkundung)
- Torch Threads: 6 (Training) + 2 (Parallelität)

## 🚀 Training starten

### Low-Spec Training (10M Timesteps)

```bash
python super_mario_rl_low_spec.py
```

Das Training:

- Läuft mit 8 parallelen Environments (optimal für 6-Core CPU)
- Erstellt alle 100.000 Steps einen Checkpoint
- Speichert das beste Model basierend auf Evaluation
- Loggt zu TensorBoard
- **Erwarte ~250k Steps/Stunde** (vs. ~1M auf High-Spec Hardware)
- **Gesamt-Trainingszeit**: ~40 Stunden für 10M Steps

### Training-Output

```
models/                      # Checkpoints alle 100k Steps
  ├── ppo_mario_lowspec_100000_steps.zip
  ├── ppo_mario_lowspec_200000_steps.zip
  └── best_model/            # Bestes Model nach Evaluation
logs/                        # Evaluation Logs
ppo_mario_tensorboard/       # TensorBoard Logs
ppo_mario_lowspec_final.zip  # Finales trainiertes Model
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

## 🎯 Training-Erwartungen (mit Advanced Reward Shaping)

### Phase 1: Exploration & Basic Movement (0-1M Steps)

- Agent lernt Grundbewegungen (rechts = gut)
- Progressive Checkpoints motivieren Vorwärtsbewegung
- Anti-Stuck-Mechanismus verhindert Stillstand
- Erreicht erste Checkpoints (x=1000)

### Phase 2: Checkpoint Mastery (1M-3M Steps)

- Erreicht regelmäßig x=1000-2000
- Lernt Sprungmechanik für Hindernisse
- Vermeidet einfache Gegner
- Zeit-Effizienz-Boni motivieren schnelleren Fortschritt

### Phase 3: Advanced Navigation (3M-7M Steps)

- Erreicht Checkpoint 3000+ regelmäßig
- Vermeidet Gegner gezielt
- Lernt komplexe Sprungkombinationen
- Weniger Tode durch bessere Planung

### Phase 4: Level-Completion (7M+ Steps)

- Schafft komplette Level (x=3161 = Flag)
- Optimiert für schnelle Completion (< 400-800 Steps)
- Learning Rate Decay stabilisiert Gelerntes
- Hohe Erfolgsrate (> 70%)

## 🔧 Anpassungen & Tuning

### Mehr Exploration?

```python
ent_coef=0.15,  # Erhöhen von 0.1 auf 0.15 (bereits hoch!)
```

### Training beschleunigen? (Erfordert mehr Hardware!)

```python
NUM_ENVS = 12,  # Mehr parallele Envs (braucht 8+ Core CPU)
batch_size=2048,  # Größere Batches (braucht 16GB+ VRAM)
```

⚠️ **Hinweis**: Low-Spec Config ist bereits für 6-Core/8GB VRAM optimiert!

### Learning Rate anpassen?

```python
# Schnelleres Lernen (riskanter)
initial_lr = 5e-5  # statt 3e-5

# Noch stabileres Lernen (langsamer)
initial_lr = 1e-5  # statt 3e-5
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
model = PPO.load("ppo_mario_lowspec_final")

# Environment erstellen
test_env = DummyVecEnv([make_env('SuperMarioBros-1-1-v0', 0)])

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
model = PPO.load("models/ppo_mario_lowspec_500000_steps")

# Bestes Model laden (basierend auf Evaluation)
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

### "Out of Memory" (GPU VRAM)

```python
# Bereits auf 1024 reduziert, aber wenn weiterhin Probleme:
batch_size=512  # statt 1024

# Oder reduziere parallele Environments
NUM_ENVS = 4  # statt 8 (halbiert VRAM-Bedarf)

# Oder reduziere n_steps
n_steps=1024  # statt 2048
```

### Agent lernt nicht / bleibt stecken

**Hinweis**: Anti-Stuck-Mechanismus ist bereits implementiert!

```python
# Verstärke Anti-Stuck Penalties
if x_variance < 10:
    reward -= 1.0  # statt 0.5

# Reduziere Stuck Threshold (schneller bestrafen)
self.stuck_threshold = 50  # statt 100

# Erhöhe X-Progress Multiplikator
reward = reward * 0.05 + x_progress * 2.0  # statt 1.5
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

**Low-Spec Hardware (6-Core i7, RTX 2070 8GB):**

- **Training-Speed**: ~250k Steps/Stunde (8 Environments)
- **Episode-Länge**: ~300-500 Steps (untrainiert), ~400-1400 Steps (trainiert)
- **TimeLimit**: 1400 Steps (gibt Agent mehr Zeit für Exploration)
- **Zeit bis erste Level-Completion**: ~8-12 Stunden (2-3M Steps mit Advanced Reward Shaping)
- **Gesamt-Trainingszeit**: ~40 Stunden für 10M Steps
- **VRAM-Nutzung**: ~6-7GB bei Batch Size 1024

**Vergleich zu High-Spec Hardware (40-Core, 2x 96GB GPU):**

- **~4× langsamer**: 250k vs. 1M Steps/Stunde
- **Aber**: Gleiche Qualität durch optimierte Hyperparameter!
- **Learning Rate Decay**: Kompensiert längere Trainingszeit

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
# SuperMarioRL
