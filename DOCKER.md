# Docker Setup für Super Mario RL

## Schnellstart

### Option 1: Docker Compose (Empfohlen)

Training + TensorBoard zusammen starten:

```bash
docker-compose up
```

TensorBoard ist dann verfügbar unter: http://localhost:6006

### Option 2: Nur Docker

**Build:**

```bash
docker build -t super-mario-rl .
```

**Training starten:**

```bash
docker run -v $(pwd)/models:/app/models \
           -v $(pwd)/logs:/app/logs \
           -v $(pwd)/ppo_mario_tensorboard:/app/ppo_mario_tensorboard \
           super-mario-rl
```

**TensorBoard separat:**

```bash
docker run -p 6006:6006 \
           -v $(pwd)/ppo_mario_tensorboard:/logs \
           tensorflow/tensorflow \
           tensorboard --logdir=/logs --host=0.0.0.0
```

## GPU Support (Optional)

### Voraussetzungen

- NVIDIA GPU
- nvidia-docker installiert

### Docker Compose mit GPU

Uncomment die GPU Sektion in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Docker Run mit GPU

```bash
docker run --gpus all \
           -v $(pwd)/models:/app/models \
           -v $(pwd)/logs:/app/logs \
           -v $(pwd)/ppo_mario_tensorboard:/app/ppo_mario_tensorboard \
           super-mario-rl
```

## Volumes

Folgende Verzeichnisse werden gemountet:

- **models/** - Trainierte Model-Checkpoints
- **logs/** - Evaluation Logs
- **ppo_mario_tensorboard/** - TensorBoard Logs

## Entwicklung

### Interaktive Shell

```bash
docker run -it --entrypoint /bin/bash super-mario-rl
```

### Code-Änderungen live testen

```bash
docker run -v $(pwd):/app super-mario-rl
```

## Troubleshooting

### "Permission denied" auf Volumes

```bash
sudo chown -R $USER:$USER models logs ppo_mario_tensorboard
```

### Training läuft nicht

```bash
# Check Container Logs
docker-compose logs -f training

# Oder direkt
docker logs super_mario_rl_training
```

### GPU nicht erkannt

```bash
# Test ob nvidia-docker funktioniert
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Produktion

### Training im Hintergrund

```bash
docker-compose up -d
```

### Logs verfolgen

```bash
docker-compose logs -f training
```

### Stoppen

```bash
docker-compose down
```

### Cleanup

```bash
# Container & Images entfernen
docker-compose down --rmi all

# Volumes behalten, nur Container löschen
docker-compose down
```

## Image Größe optimieren

Das aktuelle Image ist ~2-3 GB. Für Produktion:

1. Multi-stage Build verwenden
2. Alpine statt Slim (komplexer wegen Dependencies)
3. Nur benötigte torch Komponenten installieren

## CI/CD Integration

### GitHub Actions Beispiel

```yaml
- name: Build Docker Image
  run: docker build -t super-mario-rl .

- name: Run Training
  run: docker run --rm super-mario-rl
```
