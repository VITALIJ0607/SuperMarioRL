# Training Verbesserungen - Super Mario RL

**Datum:** 13. Januar 2026  
**Problem:** Nach 10 Millionen Trainingsschritten konnte der Agent kein Level zu Ende spielen

---

## 🔍 Problemanalyse

Nach dem ersten Training mit 10 Millionen Timesteps hat das Modell nicht gelernt, ein Level zu beenden. Mögliche Ursachen:

### 1. **Zu kurze Trainingszeit**

- 10M Timesteps mit 8 parallelen Environments = ~1,25M Schritte pro Environment
- Für Super Mario Bros ist das oft zu wenig
- Das Spiel ist komplex: Springen, Timing, Gegner ausweichen, Lücken überwinden

### 2. **Zu hoher Zeit-Penalty**

```python
reward -= 0.01  # ALT
```

- Bei Episode mit 400 Schritten → **-4.0 Gesamtstrafe**
- Macht positive Rewards zunichte
- Agent wird "bestraft fürs Spielen"

### 3. **Zu niedriger Exploration-Wert**

```python
ent_coef=0.01  # ALT
```

- Zu wenig Exploration
- Agent probiert nicht genug neue Strategien aus
- Bleibt in lokalen Optima stecken

### 4. **n_steps zu klein**

```python
n_steps=256  # ALT
```

- Ein Level braucht ~400 Schritte bis zum Ende
- Agent "sieht" in seiner Lernphase selten das Level-Ende
- Kann langfristige Strategien nicht lernen

### 5. **Zu hohe Death-Penalty**

```python
reward -= 10.0  # ALT
```

- Sehr starke Bestrafung für Tod
- Agent wird zu vorsichtig, traut sich nichts

### 6. **Generisches Environment**

```python
env_id = 'SuperMarioBros-v0'  # ALT - zufällige Level
```

- Trainiert auf zufälligen Leveln
- Schwerer zu lernen als auf einem konstanten Level

---

## ✅ Implementierte Verbesserungen

### 1. **Trainingsdauer verdoppelt**

```python
# ALT
TOTAL_TIMESTEPS = 10_000_000

# NEU
TOTAL_TIMESTEPS = 20_000_000  # +100% mehr Training
```

**Begründung:** Mehr Zeit zum Lernen, Agent sieht mehr verschiedene Situationen

---

### 2. **Zeit-Penalty drastisch reduziert**

```python
# ALT
reward -= 0.01

# NEU
reward -= 0.001  # 10x schwächer
```

**Begründung:**

- Bei 400 Schritten nur noch -0.4 statt -4.0
- Positive Rewards (Fortschritt) überwiegen wieder
- Agent wird nicht mehr fürs Spielen bestraft

---

### 3. **Death-Penalty halbiert**

```python
# ALT
reward -= 10.0

# NEU
reward -= 5.0
```

**Begründung:**

- Weniger Angst vor Fehlern
- Agent traut sich mehr zu (Springen über Lücken)
- Immer noch genug Strafe, um Vorsicht zu fördern

---

### 4. **Exploration erhöht**

```python
# ALT
ent_coef=0.01

# NEU
ent_coef=0.05  # 5x mehr Exploration
```

**Begründung:**

- Agent probiert mehr verschiedene Aktionen aus
- Entdeckt neue Strategien
- Verhindert frühes Festfahren in lokalen Optima

---

### 5. **n_steps verdoppelt**

```python
# ALT
n_steps=256

# NEU
n_steps=512
```

**Begründung:**

- Agent sammelt längere Sequenzen
- Kann besser langfristige Zusammenhänge lernen
- Sieht öfter komplette Episoden bis zum Ende

---

### 6. **Spezifisches Level für Training**

```python
# ALT
env_id = 'SuperMarioBros-v0'  # Zufällige Level

# NEU
env_id = 'SuperMarioBros-1-1-v0'  # Immer Level 1-1
```

**Begründung:**

- Level 1-1 ist das einfachste Level
- Konsistentes Training auf demselben Level
- Agent kann Level besser "auswendig" lernen
- Schnellerer Lernfortschritt

---

## 📊 Erwartete Verbesserungen

### Vor den Änderungen:

- ❌ Schafft kein Level
- ❌ Bleibt oft in frühen Bereichen stecken
- ❌ Zu vorsichtig oder zu reckless
- ❌ Wenig Exploration

### Nach den Änderungen:

- ✅ Sollte Level 1-1 nach 10-15M Steps schaffen
- ✅ Bessere Balance zwischen Vorsicht und Mut
- ✅ Mehr Exploration neuer Strategien
- ✅ Längerfristige Planung durch größere n_steps
- ✅ Stabilere Reward-Signale

---

## 🚀 Nächste Schritte

1. **Neues Training starten:**

   ```bash
   python super_mario_rl.py
   ```

2. **TensorBoard überwachen:**

   ```bash
   tensorboard --logdir=ppo_mario_tensorboard --host=0.0.0.0 --port=6006
   ```

   Achte auf:

   - **Rollout/ep_rew_mean**: Sollte steigen
   - **Rollout/ep_len_mean**: Episode-Länge sollte steigen
   - **Train/entropy_loss**: Sollte nicht zu schnell gegen 0 gehen

3. **Checkpoints prüfen:**

   - Beste Modelle werden in `./models/best_model/` gespeichert
   - Checkpoints alle 100k Steps in `./models/`

4. **Testen während Training:**
   ```bash
   python test_model.py
   ```

---

## 🔧 Weitere Optimierungsmöglichkeiten

Falls das Training immer noch nicht funktioniert:

### A) **Reward Shaping weiter anpassen**

```python
# Stärkere Belohnung für Fortschritt
reward += (info['x_pos'] - self.current_x) * 0.2  # Statt 0.1

# Bonus für schnellen Fortschritt
if info['x_pos'] > self.max_x:
    reward += 2.0  # Statt 1.0
```

### B) **Learning Rate anpassen**

```python
learning_rate=1e-4,  # Niedriger = stabiler aber langsamer
# oder
learning_rate=5e-4,  # Höher = schneller aber instabiler
```

### C) **Mehr parallele Environments**

```python
NUM_ENVS = 16  # Statt 8
```

→ Mehr Datenpunkte pro Update, aber mehr RAM/CPU-Bedarf

### D) **Curriculum Learning**

1. Trainiere auf Level 1-1 bis Erfolg
2. Dann auf Level 1-2
3. Dann auf Level 1-3
4. Schließlich auf zufälligen Leveln

### E) **Andere PPO-Parameter testen**

```python
gamma=0.995,         # Höherer Discount = langfristigeres Denken
clip_range=0.1,      # Konservativer
n_epochs=8,          # Mehr Epochen pro Update
```

---

## 📈 Erfolgsmetriken

Das Training ist erfolgreich, wenn:

1. **TensorBoard zeigt:**

   - Steigenden Mean Episode Reward (> 500)
   - Steigende Episode Length (> 300 Schritte)
   - x_pos erreicht > 3000 (Level-Ende)

2. **Test zeigt:**

   - Agent schafft Level 1-1 in >50% der Versuche
   - Springt gezielt über Lücken
   - Weicht Gegnern aus oder springt drauf

3. **Modell kann:**
   - Erste Pipe erreichen (~x=400)
   - Treppe runter (x~600)
   - Über Lücken springen (x~1500)
   - Level-Ende erreichen (x~3200)

---

## 📝 Dokumentation der Parameter-Änderungen

| Parameter         | Alt   | Neu    | Faktor      | Begründung         |
| ----------------- | ----- | ------ | ----------- | ------------------ |
| `TOTAL_TIMESTEPS` | 10M   | 20M    | 2x          | Mehr Lernzeit      |
| `Zeit-Penalty`    | -0.01 | -0.001 | 10x weniger | Weniger Bestrafung |
| `Death-Penalty`   | -10.0 | -5.0   | 2x weniger  | Mutiger spielen    |
| `ent_coef`        | 0.01  | 0.05   | 5x          | Mehr Exploration   |
| `n_steps`         | 256   | 512    | 2x          | Längere Sequenzen  |
| `env_id`          | v0    | 1-1-v0 | -           | Spezifisches Level |

---

## ⚠️ Bekannte Probleme

1. **Exit Code 139** beim Testen

   - Segmentation Fault, oft durch `render()` verursacht
   - Lösung: Render nur bei Bedarf oder headless mode

2. **Training bricht ab**

   - Speicherprobleme bei vielen Environments
   - Lösung: NUM_ENVS reduzieren oder RAM upgraden

3. **GPU nicht genutzt**
   - Prüfe mit `nvidia-smi`
   - Stable Baselines3 nutzt automatisch GPU wenn verfügbar

---

**Erstellt am:** 13. Januar 2026  
**Nächstes Review:** Nach 20M Timesteps Training
