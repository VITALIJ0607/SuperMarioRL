# PPO Run Analyse - Super Mario RL

**Datum**: 17. Januar 2026  
**Training**: 2.225.000 Timesteps (~22% des Ziels)

## 🔍 Hauptproblem identifiziert

### Agent-Verhalten beim Best Model (200K Steps):
- ❌ **Erreicht nur X-Position 315** (Ziel ist bei ~3200)
- ❌ **Nutzt NUR eine Action: RIGHT+B** (100% der Zeit!)
- ❌ **Deterministisches Fehlverhalten**: Stirbt nach exakt 106 Steps
- ❌ **Keine Exploration mehr**: Clip Fraction = 0.0

### Performance-Metriken:
```
Mean Reward:  409.5 ± 0.0
Mean Steps:   106.0 ± 0.0  
Mean Max X:   315.0 ± 0.0

Erfolgsrate:
  🏁 Flags erreicht: 0/10 (0%)
  💀 Gestorben:      0/10 (0%)
  ⏱️ Timeouts:       10/10 (100%)
```

## 📊 Training Progression

| Phase | Timesteps | Mean Reward | Episode Length | Status |
|-------|-----------|-------------|----------------|--------|
| Start | 5K | -954.3 | 1400 | Time limit |
| **Peak** | **200K** | **+435.2** | **106** | **Best** |
| 25% | 555K | +434.8 | 120 | Plateau |
| 50% | 1.1M | +434.8 | 120 | Plateau |
| 75% | 1.67M | +434.8 | 120 | Plateau |
| Ende | 2.22M | +430.9 | 112 | Keine Verbesserung |

**Kritisch**: Letzten 20 Evals Mean Reward: **-843.0** (Verschlechterung!)

## 🐛 Identifizierte Probleme

### 1. Learning Rate zu konservativ
- Aktuell: Linear Decay 3e-5 → 1e-6
- Problem: Policy "friert ein" nach 200K Steps
- Clip Fraction = 0.0 → Keine signifikanten Updates mehr

### 2. Zu wenig Exploration
- `ent_coef=0.1` reicht nicht
- Agent exploitiert lokales Optimum (RIGHT+B spam)
- Keine Diversität im Action Space

### 3. Reward Shaping Probleme
- X-Progress Belohnung zu niedrig (1.5x)
- Zu harte Strafen für Steckenbleiben
- Anti-Stuck-Mechanismus zu aggressiv
- Agent lernt "sicher sterben" statt "Risiko eingehen"

### 4. Batch Size zu groß
- 1024 → zu wenig Updates pro Rollout
- Policy ändert sich zu langsam

## ✅ Implementierte Verbesserungen

### Neue Version: super_mario_rl_improved.py

#### 1. Höhere Learning Rate
```python
initial_lr = 1e-4   # ERHÖHT von 3e-5
final_lr = 5e-6     # ERHÖHT von 1e-6
```

#### 2. Mehr Exploration
```python
ent_coef=0.2  # VERDOPPELT von 0.1
```

#### 3. Verbessertes Reward Shaping
```python
# Höhere X-Progress Belohnung
reward += x_progress * 2.0  # ERHÖHT von 1.5

# Häufigere Checkpoints mit steigenden Boni
checkpoints = {500, 1000, 1500, 2000, 2500, 3000, 3200}
bonus = 50 + (checkpoint_x // 500) * 25

# Sanftere Anti-Stuck Strafen
- Toleranz erhöht: 10 → 20 pixels
- Strafe reduziert: 0.5 → 0.2
- Threshold erhöht: 100 → 150 steps

# Moderate Tod-Strafen
death_penalty = max(30.0 - (max_x / 100), 10.0)
```

#### 4. Kleinere Batch Size
```python
batch_size=512  # REDUZIERT von 1024
```

#### 5. Optimierte PPO-Parameter
```python
gamma=0.99          # ERHÖHT von 0.98 (längerer Horizont)
vf_coef=0.5         # REDUZIERT von 1.0 (Policy wichtiger)
gae_lambda=0.95     # Standard GAE
```

## 📈 Erwartete Verbesserungen

1. **Mehr Exploration**: Agent probiert verschiedene Actions
2. **Schnelleres Learning**: Höhere LR ermöglicht größere Updates
3. **Bessere Anreize**: Stärkere Belohnung für Fortschritt
4. **Weniger "Angst"**: Sanftere Strafen → Agent wagt mehr
5. **Mehr Updates**: Kleinere Batch Size → schnellere Anpassung

## 🎯 Nächste Schritte

1. ✅ **Analyse durchgeführt**: Problem identifiziert
2. ✅ **Verbesserungen implementiert**: Neue Version erstellt
3. ⏳ **Training starten**: `python super_mario_rl_improved.py`
4. ⏳ **Monitoring**: TensorBoard + regelmäßige Evaluations
5. ⏳ **Falls erfolgreich**: Curriculum Learning für mehrere Levels

## 📝 Monitoring Checklist

Bei neuem Training beobachten:
- [ ] Clip Fraction > 0 (Learning findet statt)
- [ ] Action Distribution diverser (nicht nur RIGHT+B)
- [ ] Max X-Position steigt über 315
- [ ] Entropy Loss bleibt hoch (Exploration)
- [ ] Mean Reward steigt kontinuierlich
- [ ] Checkpoints werden erreicht

## 🔗 Dateien

- **Analyse Script**: `analyze_best_model.py`
- **Alte Version**: `super_mario_rl_low_spec.py`
- **Neue Version**: `super_mario_rl_improved.py`
- **Logs Alt**: `logs/evaluations.npz`
- **TensorBoard Alt**: `ppo_mario_tensorboard/PPO_1/`
- **TensorBoard Neu**: `ppo_mario_tensorboard_v2/`
- **Models Neu**: `models_v2/`
