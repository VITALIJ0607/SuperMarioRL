# Reward Shaping Optimierungen (Version 2)

**Datum:** 15. Januar 2026  
**Basierend auf:** PPO_5 Training-Analyse

## 🎯 Ziele der Optimierung

1. ✅ Timeout-Bestrafung implementieren
2. ✅ Feststecken erkennen und vermeiden
3. ✅ Schnelle Level-Completions fördern
4. ✅ Frühen Tod stärker bestrafen

---

## 📋 Implementierte Änderungen

### **1. Anti-Stuck Mechanismus**

#### **Tracking-System:**

- `stuck_counter`: Zählt Steps ohne X-Fortschritt
- `last_x_positions`: Speichert letzte 40 X-Positionen (~2 Sekunden bei 20 FPS)
- `stuck_threshold`: 100 Steps ohne Fortschritt = Feststecken

#### **Bestrafungen:**

```python
# Kontinuierliche Bestrafung bei wenig Bewegung
if x_variance < 5 pixels in 40 frames:
    reward -= 0.5

# Starke Bestrafung bei längerem Feststecken
if stuck_counter > 100:
    reward -= 1.0 per step
```

### **2. Timeout-Bestrafung**

**Unterscheidung bei Episode-Ende:**

```python
if flag_get:
    reward += 500.0  # Level geschafft
elif time <= 0:
    reward -= 100.0  # TIMEOUT - Neu!
else:
    reward -= 50.0   # Tod
```

**Zusätzlich:**

- Extra -20.0 bei frühem Tod (< 500 Steps)
- Agent lernt, länger zu überleben

### **3. Geschwindigkeits-Incentives**

**Belohnung für schnelle Completions:**

```python
if flag_get:
    if steps < 400:
        reward += 200.0  # Sehr schnell!
    elif steps < 800:
        reward += 100.0  # Schnell
```

**Früher Fortschritt:**

```python
if x_progress > 0 and steps < 1000:
    reward += 0.1  # Leichter Bonus
```

### **4. Angepasste Basis-Rewards**

**Vorher:**

```python
reward = game_reward * 0.1 + x_progress * 1.0
```

**Nachher:**

```python
reward = game_reward * 0.05 + x_progress * 1.5
```

**Rationale:**

- Game-Reward noch weiter reduziert (0.1 → 0.05)
- X-Progress stärker gewichtet (1.0 → 1.5)
- Fokus auf Vorwärtsbewegung

---

## 📊 Erwartete Verbesserungen

### **Problem-Lösung Matrix:**

| Problem             | Alte Metrik | Lösung                 | Erwartung |
| ------------------- | ----------- | ---------------------- | --------- |
| **Timeouts**        | 38.4%       | -100 Bestrafung        | < 20%     |
| **Feststecken**     | Häufig      | Stuck-Detection        | Selten    |
| **Completion-Rate** | 4.6%        | Speed-Bonus            | > 10%     |
| **Episode-Länge**   | 614 avg     | Zeit-Effizienz         | < 500 avg |
| **Instabilität**    | σ = 118     | Konsistente Bestrafung | σ < 80    |

---

## 🔧 Neue Tracking-Variablen

```python
self.max_x = 0              # Höchste X-Position
self.current_x = 0          # Aktuelle X-Position
self.stuck_counter = 0      # Steps ohne Fortschritt
self.stuck_threshold = 100  # Limit für Feststecken
self.last_x_positions = []  # Historie (40 Frames)
self.step_count = 0         # Episode Step Counter
```

---

## 🧪 Empfohlene Test-Strategie

### **Phase 1: Kurzer Test (500k Steps)**

- Prüfen ob Anti-Stuck funktioniert
- Timeout-Rate beobachten
- Vergleich mit PPO_5 Baseline

### **Phase 2: Mittlerer Test (2M Steps)**

- Completion-Rate messen
- Episode-Längen-Verteilung
- Reward-Stabilität

### **Phase 3: Voller Test (5-10M Steps)**

- Langzeit-Performance
- Generalisierung prüfen
- Vergleich mit besten PPO_5 Ergebnissen

---

## 📈 Zu überwachende Metriken

### **Haupt-KPIs:**

1. **Timeout-Rate:** Sollte signifikant sinken
2. **Completion-Rate:** Sollte steigen (> 10%)
3. **Durchschnittliche Episode-Länge:** Sollte konsistent sein (~400-600)
4. **Reward-Stabilität:** Standardabweichung sollte sinken

### **Sekundär-Metriken:**

5. **Max-X-Position:** Durchschnitt sollte steigen
6. **Stuck-Events:** Häufigkeit sollte sinken
7. **Frühe Tode (< 500 Steps):** Sollte reduziert werden

---

## ⚙️ Hyperparameter zum Tuning

Wenn die Ergebnisse nicht zufriedenstellend sind:

```python
# Anti-Stuck
stuck_threshold = 100      # → 80 oder 120 testen
stuck_penalty = -1.0       # → -0.5 oder -2.0 testen

# Stillstand-Detection
x_variance_threshold = 5   # → 3 oder 10 testen
stillstand_penalty = -0.5  # → -0.3 oder -1.0 testen

# Timeout
timeout_penalty = -100.0   # → -75 oder -150 testen

# Speed Bonus
fast_completion_time = 400 # → 300 oder 500 testen
```

---

## 🚀 Nächste Schritte

1. **Training starten** mit neuer Reward-Struktur
2. **TensorBoard überwachen** für erste 500k Steps
3. **Evaluations-Daten vergleichen** mit PPO_5
4. **Bei Bedarf Hyperparameter anpassen**
5. **Längeres Training (5M+ Steps)** wenn vielversprechend

---

## 💡 Design-Philosophie

**Prinzipien:**

- ✅ Fortschritt belohnen (X-Progress)
- ✅ Stillstand bestrafen (Stuck-Detection)
- ✅ Effizienz fördern (Speed-Bonus)
- ✅ Zeitverschwendung bestrafen (Timeout-Penalty)
- ✅ Konsistenz über Spikes (Glatte Belohnungen)

**Vermeiden:**

- ❌ Zu aggressive Bestrafungen (Verhindert Lernen)
- ❌ Zu viele kleine Rewards (Rauschen)
- ❌ Widersprüchliche Signale
- ❌ Überoptimierung auf einen Aspekt

---

## 📝 Code-Änderungen

**Geänderte Datei:** `super_mario_rl_low_spec.py`  
**Klasse:** `RewardShapingWrapper`  
**Zeilen:** ~57-140

**Wichtigste Änderungen:**

1. Erweiterte `__init__()` mit Tracking-Variablen
2. Erweiterte `reset()` für sauberes Zurücksetzen
3. Vollständig überarbeitete `step()` Logik
4. Neue Bestrafungsmechanismen
5. Geschwindigkeits-Incentives

---

**Version:** 2.0  
**Letzte Aktualisierung:** 15. Januar 2026, 23:40
