# HOST SOUND MIXER

```bash
sudo amixer -c 1 sset Capture 100% cap
Simple mixer control 'Capture',0
  Capabilities: cvolume cswitch
  Capture channels: Front Left - Front Right
  Limits: Capture 0 - 63
  Front Left: Capture 63 [100%] [30.00dB] [on]
  Front Right: Capture 63 [100%] [30.00dB] [on]

sudo amixer -c 1 sset Mic 100% mute
Simple mixer control 'Mic',0
  Capabilities: pvolume pswitch
  Playback channels: Front Left - Front Right
  Limits: Playback 0 - 31
  Mono:
  Front Left: Playback 31 [100%] [12.00dB] [off]
  Front Right: Playback 31 [100%] [12.00dB] [off]

```

## TRAINED DATA on DOCKER

### BACKUP

- DOCKER COMPOSE

```bash
docker compose run --rm snd-monitor  tar -czf - -C /app/data . > ./data/backup/data_volume_backup.tar.gz
```

- SSH + DOCKER IMAGE alpine

```bash
ssh user@remote "docker run --rm -v t-mon_snd_monitor_data:/app/data alpine tar -czf - -C /app/data ." > ./data/backup/data_volume_backup.tar.gz
```

### RESTORE

- Linux/macOS

```bash
cat ./data/backup/data_volume_backup.tar.gz | ssh user@remote "docker run --rm -i -v t-mon_snd_monitor_data:/app/data alpine tar -xzf - -C /app/data"
```

- Windows cmd

    - DOCKER COMPOSE

    ```bash
    type .\data\backup\data_volume_backup.tar.gz | docker compose run --rm  snd-monitor tar -xzf - -C /app/data
    ```

- Windows PowerShell

- DOCKER COMPOSE

   ```bash
   Get-Content ./data/backup/data_volume_backup.tar.gz -AsByteStream | docker compose run --rm  snd-monitor  tar -xzf - -C /app/data
   ```

- DOCKER IMAGE alpine

   ```bash
   Get-Content ./data/backup/data_volume_backup.tar.gz -AsByteStream | docker run --rm -i -v t-mon_snd_monitor_data:/app/data alpine tar -xzf - -C /app/data
   ```
