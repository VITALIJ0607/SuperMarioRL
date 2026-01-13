# Implementierungsschritte: Super Mario RL mit PPO

## Zusammenfassung der durchgeführten Änderungen

### 1. **Algorithmus-Umstellung**

- ❌ **Vorher**: Einfaches Random-Agent-Testskript
- ✅ **Nachher**: Vollständiges PPO-Training mit Stable-Baselines3

### 2. **Implementierte Features**

#### Core-Komponenten

- **PPO-Algorithmus** von Stable-Baselines3
- **CnnPolicy** für Pixel-basierte Beobachtungen
- **DummyVecEnv** mit 8 parallelen Environments
- **Automatische Checkpoints** alle 100k Steps
- **Evaluation Callback** mit Best-Model-Speicherung
- **TensorBoard Logging** für Monitoring

#### Environment-Wrapper-Pipeline

```
SuperMarioBros-v0
    ↓
JoypadSpace (7 Aktionen)
    ↓
RewardShapingWrapper (Custom Rewards)
    ↓
GrayScaleObservation (RGB → Graustufen)
    ↓
ResizeObservation (240x256 → 84x84)
    ↓
FrameStack (4 Frames)
    ↓
SqueezeObservation (Shape-Korrektur)
    ↓
GymCompatibilityWrapper (API-Brücke)
```

### 3. **Custom Wrapper Implementierungen**

#### **GymCompatibilityWrapper**

**Problem**: Alte gym-super-mario-bros nutzt alte gym API, Stable-Baselines3 erwartet neue gymnasium API

**Lösung**:

```python
class GymCompatibilityWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        # Entfernt seed/options Parameter
        # Konvertiert: obs → (obs, info)

    def step(self, action):
        # Konvertiert: (obs, reward, done, info)
        # zu: (obs, reward, terminated, truncated, info)
```

#### **SqueezeObservation**

**Problem**: GrayScaleObservation + FrameStack erzeugt Shape (4,84,84,1) statt (4,84,84)

**Lösung**:

```python
class SqueezeObservation(gym.ObservationWrapper):
    # Entfernt überflüssige Dimension
    # Passt observation_space entsprechend an
```

#### **RewardShapingWrapper**

**Motivations-Features**:

- ✅ +0.1 × Δx für Fortschritt nach rechts
- ✅ +1.0 Bonus für neuen x-Position-Rekord
- ⏱️ -0.01 Zeit-Penalty pro Step
- ☠️ -10.0 Death Penalty

### 4. **Behobene Probleme**

#### Problem 1: SubprocVecEnv Multiprocessing-Crash

**Fehler**: `ConnectionResetError: [Errno 104] Connection reset by peer`

**Ursache**: numpy/nes_py Inkompatibilität in Subprozessen

**Lösung**: Wechsel zu **DummyVecEnv** (sequentiell, aber stabil)

---

#### Problem 2: seed() Parameter nicht unterstützt

**Fehler**: `TypeError: JoypadSpace.reset() got an unexpected keyword argument 'seed'`

**Ursache**: Alte gym Environments kennen seed/options Parameter nicht

**Lösung**: GymCompatibilityWrapper filtert Parameter raus

---

#### Problem 3: Reset Return Value Mismatch

**Fehler**: `ValueError: too many values to unpack (expected 2)`

**Ursache**:

- Alte API: `reset()` → `obs`
- Neue API: `reset()` → `(obs, info)`

**Lösung**: GymCompatibilityWrapper gibt `(obs, {})` zurück

---

#### Problem 4: Step Return Value Mismatch

**Fehler**: `ValueError: not enough values to unpack (expected 5, got 4)`

**Ursache**:

- Alte API: `step()` → `(obs, reward, done, info)` (4 Werte)
- Neue API: `step()` → `(obs, reward, terminated, truncated, info)` (5 Werte)

**Lösung**: GymCompatibilityWrapper konvertiert: `done → (terminated, False)`

---

#### Problem 5: cv2 (OpenCV) fehlt

**Fehler**: `ModuleNotFoundError: No module named 'cv2'`

**Ursache**: GrayScaleObservation benötigt OpenCV

**Lösung**:

```bash
pip install opencv-python-headless
pip install --force-reinstall numpy==1.23.5  # Wegen Konflikt
```

---

#### Problem 6: Observation Shape Mismatch

**Fehler**: `ValueError: could not broadcast input array from shape (4,84,84,1) into shape (4,84,84)`

**Ursache**: GrayScaleObservation fügt Dimension hinzu → (84,84,1)

**Lösung**: SqueezeObservation nach FrameStack platziert

---

#### Problem 7: numpy Version Konflikte

**Fehler**: `OverflowError: Python integer 1024 out of bounds for uint8`

**Ursache**: nes_py ist nicht kompatibel mit numpy 2.x

**Lösung**:

```bash
pip install --force-reinstall numpy==1.23.5
```

### 5. **Optimierte PPO-Hyperparameter**

| Parameter       | Wert   | Zweck                            |
| --------------- | ------ | -------------------------------- |
| `n_steps`       | 256    | Steps pro Environment vor Update |
| `batch_size`    | 256    | Mini-Batch für Training          |
| `learning_rate` | 2.5e-4 | Adam Learning Rate               |
| `gamma`         | 0.99   | Discount Factor                  |
| `gae_lambda`    | 0.95   | GAE Parameter                    |
| `ent_coef`      | 0.01   | Exploration (Entropy)            |
| `clip_range`    | 0.2    | PPO Clipping                     |
| `n_epochs`      | 4      | Gradient Updates pro Rollout     |

### 6. **Environment Setup**

#### Neue Environment-Struktur

```bash
conda create -n super_mario_rl python=3.10
conda activate super_mario_rl

# Installierte Pakete
numpy==1.23.5              # Kompatibel mit nes_py
gym==0.25.2                # Basis
gymnasium==0.28.1          # Für SB3
shimmy==0.2.1              # gym↔gymnasium Bridge
stable-baselines3==2.0.0   # RL Framework
torch                      # Mit CUDA Support
tensorboard                # Monitoring
opencv-python-headless     # Für GrayScale
gym-super-mario-bros==7.4.0
nes-py==8.2.1
```

### 7. **Warum diese Lösungen?**

#### DummyVecEnv statt SubprocVecEnv

- ✅ Vermeidet Multiprocessing-Probleme mit nes_py
- ✅ Einfacher zu debuggen
- ⚠️ Langsamer, aber für 8 Envs akzeptabel

#### Wrapper-Reihenfolge kritisch

```
Falsch: GymCompatibility ZUERST
→ GrayScale kriegt falsches Format

Richtig: GymCompatibility ZULETZT
→ Erst alle gym-Wrapper, dann API-Konvertierung
```

#### numpy 1.23.5 statt 2.x

- nes_py verwendet veraltete numpy APIs
- opencv-python-headless funktioniert trotzdem (ignoriere pip warning)

### 8. **Finaler Code-Umfang**

- **~160 Zeilen** Python
- **3 Custom Wrapper** (GymCompatibility, SqueezeObservation, RewardShaping)
- **Vollständiges Training-Pipeline** mit Callbacks
- **Automatische Model-Speicherung**
- **TensorBoard Integration**

### 9. **Training starten**

```bash
conda activate super_mario_rl
python super_mario_rl.py
```

**Erwartete Ausgabe**:

- Environment-Initialisierung (8x)
- PPO Model Creation
- Logging to tensorboard/
- Rollout-Progress (Steps, Episodes, Rewards)
- Automatic Checkpoints

### 10. **Nächste Schritte**

#### Monitoring

```bash
tensorboard --logdir=./ppo_mario_tensorboard
```

#### Training anpassen

- Mehr Exploration: `ent_coef=0.05`
- Mehr Environments: `NUM_ENVS=16`
- Andere Levels: `env_id='SuperMarioBros-1-2-v0'`

#### Fortgeschrittene Optimierungen

- Intrinsic Motivation (RND) für sparse Rewards
- Curriculum Learning (einfache → schwere Levels)
- RecurrentPPO mit LSTM für besseres Timing

---

## Lessons Learned

1. **API-Kompatibilität ist kritisch**: gym → gymnasium Migration betrifft viele Projekte
2. **Wrapper-Reihenfolge wichtig**: Observation-Transformation vor API-Konvertierung
3. **numpy Versionen**: Legacy-Code (nes_py) braucht alte Versionen
4. **Multiprocessing fragil**: Bei Problemen → DummyVecEnv als Fallback
5. **Shape Debugging**: print() nach jedem Wrapper hilft enorm

---

**Status**: ✅ **Voll funktionsfähig und trainingsbereit!**
