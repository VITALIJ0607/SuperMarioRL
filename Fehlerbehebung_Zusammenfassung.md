# Super Mario RL - Fehlerbehebung Zusammenfassung

## Projekt

Super Mario Bros Reinforcement Learning Umgebung

## Aufgetretene Probleme und Lösungen

### 1. Python 3.14 Inkompatibilität

**Problem:**

```
OverflowError: Python integer 1024 out of bounds for uint8
```

**Ursache:**

- Python 3.14 hat strengere Typ-Checks eingeführt
- Die Bibliothek `nes_py` ist noch nicht mit Python 3.14 kompatibel
- Fehler trat beim Laden der ROM-Daten auf

**Lösung:**
Python auf Version 3.11 downgraden:

```bash
conda install python=3.11 -y
```

---

### 2. NumPy 2.x Inkompatibilität

**Problem:**

```
OverflowError: Python integer 1024 out of bounds for uint8
```

(Gleicher Fehler blieb nach Python-Downgrade bestehen)

**Ursache:**

- NumPy 2.x (Version 2.1.3) hat strengere Typ-Validierung
- `nes_py` versucht Wert 1024 in uint8 zu schreiben (nur 0-255 erlaubt)
- Die Bibliothek ist nicht mit NumPy 2.x kompatibel

**Lösung:**
NumPy auf Version 1.x downgraden:

```bash
conda install -n super_mario_rl "numpy<2.0" -y
```

Installierte Version: NumPy 1.26.4

---

### 3. Gym API-Versionsinkompatibilität

**Problem:**

```
ValueError: not enough values to unpack (expected 5, got 4)
```

**Ursache:**

- Gym Version 0.26.2 war installiert
- Ab Gym 0.26+ gibt `env.step()` 5 Werte zurück: `(observation, reward, terminated, truncated, info)`
- `gym-super-mario-bros` nutzt die alte API mit 4 Werten: `(observation, reward, done, info)`
- Wrapper-Konflikt zwischen alter und neuer API

**Lösung:**
Gym auf kompatible Version 0.23.1 downgraden:

```bash
python -m pip uninstall gym -y
python -m pip install gym==0.23.1
```

---

## Funktionierende Umgebungskonfiguration

### Python-Pakete

| Paket                | Version |
| -------------------- | ------- |
| Python               | 3.11.14 |
| NumPy                | 1.26.4  |
| gym                  | 0.23.1  |
| nes-py               | 8.2.1   |
| gym-super-mario-bros | 7.4.0   |

### Conda Environment

```bash
conda create -n super_mario_rl python=3.11 -y
conda activate super_mario_rl
conda install "numpy<2.0" -y
pip install nes-py gym-super-mario-bros gym==0.23.1
```

---

## Code

```python
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()
```

---

## Wichtige Hinweise

1. **NumPy-Version kritisch:** NumPy muss < 2.0 sein
2. **Gym-Version kritisch:** Gym muss <= 0.25.2 sein (0.23.1 empfohlen)
3. **Python-Version:** Python 3.11 oder 3.12 verwenden, NICHT 3.14
4. **Alte API:** Der Code verwendet die alte Gym-API mit 4 Rückgabewerten

---

## Ausführen des Projekts

```bash
conda activate super_mario_rl
python super_mario_rl.py
```

---

Datum: 12. Januar 2026
