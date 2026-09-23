# HOST SOUND MIXER

```bash
sudo amixer -c 1 sset Capture 80% cap
Simple mixer control 'Capture',0
  Capabilities: cvolume cswitch
  Capture channels: Front Left - Front Right
  Limits: Capture 0 - 63
  Front Left: Capture 50 [79%] [20.25dB] [on]
  Front Right: Capture 50 [79%] [20.25dB] [on]

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

.env

```dotenv
WAITER=1
```

WAITER=1

### Train

```bash
docker compose exec -it snd-monitor bash
root@f83591478a80:/app# cd src/SoundMonitor/
root@f83591478a80:/app/src/SoundMonitor# python main.py train --prune
Available input devices:
  [1] HD-Audio Generic: ALC3227 Analog (hw:1,0)  (max in=2, default rate=44100)

2026-09-21 23:21:24 [INFO] SoundMonitor.train: 
============================================================
TRAINING MODE: auto
Each recording lasts 8 seconds.
Will try sample rates: [44100, 48000, 16000]
============================================================
2026-09-21 23:21:24 [INFO] SoundMonitor.train: 
============================================================
  Compressor ON. Total templates: (30) with delay 10 seconds
 Pruned total: 30 previous templates for label: on


>>> Prepare 'on [0]' state, then press Enter to start recording…
2026-09-21 23:26:05 [INFO] SoundMonitor.audio_device:   Audio opened at 44100 Hz
2026-09-21 23:26:05 [INFO] SoundMonitor.audio_device:   Recording 8.0s @ 44100 Hz …
2026-09-21 23:26:16 [INFO] SoundMonitor.audio_device:   Recording done
2026-09-21 23:26:17 [INFO] SoundMonitor.templates:   Saved template → /app/data/templates/template_on_000
.npz  (sr=44100)
2026-09-21 23:26:17 [INFO] SoundMonitor.train:   Top peaks (Hz) from 108: 97, 94, 100, 92, 102
2026-09-21 23:26:17 [INFO] SoundMonitor.train: Sleeping 10 seconds ...

...

============================================================
  Compressor OFF. Total templates: (5) with delay 30 seconds
 Pruned total: 5 previous templates for label: off

>>> Prepare 'off [0]' state, then press Enter to start recording…
2026-09-21 23:35:35 [INFO] SoundMonitor.audio_device:   Audio opened at 44100 Hz
2026-09-21 23:35:35 [INFO] SoundMonitor.audio_device:   Audio opened at 44100 Hz
2026-09-21 23:35:35 [INFO] SoundMonitor.audio_device:   Recording 8.0s @ 44100 Hz …
2026-09-21 23:35:45 [INFO] SoundMonitor.audio_device:   Recording done
2026-09-21 23:35:45 [INFO] SoundMonitor.templates:   Saved template → /app/data/templates/template_off_000.npz  (sr=44100)
2026-09-21 23:35:45 [INFO] SoundMonitor.train:   Top peaks (Hz) from 108: 92, 226, 89, 94, 285
2026-09-21 23:35:45 [INFO] SoundMonitor.train: Sleeping 30 seconds ...
...
2026-09-21 23:41:02 [INFO] SoundMonitor.calibrate: ============================================================
2026-09-21 23:41:02 [INFO] SoundMonitor.calibrate: AUTOMATIC THRESHOLD CALIBRATION
2026-09-21 23:41:02 [INFO] SoundMonitor.calibrate: ============================================================
2026-09-21 23:41:02 [INFO] SoundMonitor.calibrate:   Spectral Distance ON ↔ OFF : 0.9973
2026-09-21 23:41:02 [INFO] SoundMonitor.calibrate:   Intra-class Distance       : 0.9836
2026-09-21 23:41:02 [INFO] SoundMonitor.calibrate:   Safety Margin              : 0.05
2026-09-21 23:41:02 [INFO] SoundMonitor.calibrate:   Confidence Gate (min_score): 0.550
2026-09-21 23:41:02 [INFO] SoundMonitor.calibrate:   Hysteresis ON / OFF        : 0.70 / 0.20
2026-09-21 23:41:02 [INFO] SoundMonitor.calibrate:   Template Sample Rate       : 44100 Hz
2026-09-21 23:41:02 [INFO] SoundMonitor.calibrate:   Saved to                   : /app/data/templates/threshold.json
2026-09-21 23:41:02 [INFO] SoundMonitor.calibrate:   ✓  Good separation between ON and OFF templates.
2026-09-21 23:41:02 [INFO] SoundMonitor.calibrate: ============================================================
```

### Test detection

```bash
docker compose exec -it snd-monitor bash
root@f83591478a80:/app# cd src/SoundMonitor/
root@f83591478a80:/app/src/SoundMonitor# python main.py --loglevel=DEBUG detect --plot 

```

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
