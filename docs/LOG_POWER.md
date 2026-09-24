# LOGS DEBUG

## Compressor ON

```bash
SETUP AUDIO MIXER...
Simple mixer control 'Mic Boost',0
  Capabilities: volume
  Playback channels: Front Left - Front Right
  Capture channels: Front Left - Front Right
  Limits: 0 - 3
  Front Left: 0 [0%] [0.00dB]
  Front Right: 0 [0%] [0.00dB]
Simple mixer control 'Internal Mic Boost',0
  Capabilities: volume
  Playback channels: Front Left - Front Right
  Capture channels: Front Left - Front Right
  Limits: 0 - 3
  Front Left: 0 [0%] [0.00dB]
  Front Right: 0 [0%] [0.00dB]
Simple mixer control 'Capture',0
  Capabilities: cvolume cswitch
  Capture channels: Front Left - Front Right
  Limits: Capture 0 - 63
  Front Left: Capture 50 [79%] [20.25dB] [on]
  Front Right: Capture 50 [79%] [20.25dB] [on]
Simple mixer control 'Mic',0
  Capabilities: pvolume pswitch
  Playback channels: Front Left - Front Right
  Limits: Playback 0 - 31
  Mono:
  Front Left: Playback 31 [100%] [12.00dB] [off]
  Front Right: Playback 31 [100%] [12.00dB] [off]


RUNNING detect --loglevel=DEBUG --plot...
2026-09-24 01:00:39 [INFO] SoundMonitor.run_detect: ============================================================
2026-09-24 01:00:39 [INFO] SoundMonitor.run_detect: LIVE ASYNC DETECTION
2026-09-24 01:00:39 [INFO] SoundMonitor.run_detect:   Window       : 3.0s
2026-09-24 01:00:39 [INFO] SoundMonitor.run_detect:   SQLite DB    : /app/data/snd_data.db
2026-09-24 01:00:39 [INFO] SoundMonitor.run_detect:   Will try rates: [44100, 48000, 16000]
2026-09-24 01:00:39 [INFO] SoundMonitor.run_detect:   Ctrl+C to stop
2026-09-24 01:00:39 [INFO] SoundMonitor.run_detect: ============================================================
2026-09-24 01:00:39 [INFO] SoundMonitor.audio_device: 
Available input devices:
  [1] HD-Audio Generic: ALC3227 Analog (hw:1,0)  (max in=2, default rate=44100)

2026-09-24 01:00:39 [INFO] SoundMonitor.run_detect: 
2026-09-24 01:00:39 [INFO] SoundMonitor.db_writer_worker: init_db done
2026-09-24 01:00:39 [INFO] SoundMonitor.db_writer_worker: [DB] Worker is ready
2026-09-24 01:00:39 [INFO] SoundMonitor.db_writer_worker: [DB] CLEANUP initialized every 86400 seconds.
2026-09-24 01:00:39 [INFO] SoundMonitor.async_mic:   Microphone stream started (callback, 44100 Hz)
2026-09-24 01:00:39 [DEBUG] SoundMonitor.state_filter: scored_weighted_density, plot=True. (Ctrl+C to stop)

2026-09-24 01:00:39 [WARNING] SoundMonitor.detector: Recording failed: audio device is short 0 < min_samples=132300
2026-09-24 01:00:40 [WARNING] SoundMonitor.detector: Recording failed: audio device is short 5120 < min_samples=132300
...
2026-09-24 01:00:45 [WARNING] SoundMonitor.detector: Recording failed: audio device is short 132096 < min_samples=132300

2026-09-24 01:00:45 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:00:45 [DEBUG] SoundMonitor.power_predictor: Power: -59.97dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:00:45 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:00:45 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.96, state=1, weighted_density=0.96. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:00:45 [INFO] SoundMonitor.detector: STATE: ON , SCORE: 0.9590, TRUSTED STATE: ON 

2026-09-24 01:00:46 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:00:46 [DEBUG] SoundMonitor.power_predictor: Power: -60.55dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:00:46 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:00:46 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.91, state=1, weighted_density=0.92. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:00:46 [INFO] SoundMonitor.detector: STATE: ON , SCORE: 0.9055, TRUSTED STATE: ON 

2026-09-24 01:00:48 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:00:48 [DEBUG] SoundMonitor.power_predictor: Power: -60.56dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:00:48 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:00:48 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.90, state=1, weighted_density=0.91. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:00:48 [INFO] SoundMonitor.detector: STATE: ON , SCORE: 0.9043, TRUSTED STATE: ON 

2026-09-24 01:00:49 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:00:49 [DEBUG] SoundMonitor.power_predictor: Power: -59.65dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:00:49 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:00:49 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.99, state=1, weighted_density=0.94. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:00:49 [INFO] SoundMonitor.detector: STATE: ON , SCORE: 0.9884, TRUSTED STATE: ON 
```

## Compressor ON -> OFF

```bash
2026-09-24 01:06:21 [INFO] SoundMonitor.detector: STATE: ON , SCORE: 0.8234, TRUSTED STATE: ON 

2026-09-24 01:06:22 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:22 [DEBUG] SoundMonitor.power_predictor: Power: -68.54dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:22 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:22 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.17, state=1, weighted_density=0.91. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:06:22 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.1710, TRUSTED STATE: ON 

2026-09-24 01:06:24 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:24 [DEBUG] SoundMonitor.power_predictor: Power: -71.82dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:24 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:24 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.00, state=1, weighted_density=0.85. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:06:24 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.0000, TRUSTED STATE: ON 

2026-09-24 01:06:25 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:25 [DEBUG] SoundMonitor.power_predictor: Power: -72.59dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:25 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:25 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.00, state=1, weighted_density=0.79. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:06:25 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.0000, TRUSTED STATE: ON 

2026-09-24 01:06:26 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:26 [DEBUG] SoundMonitor.power_predictor: Power: -73.82dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:26 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:26 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.00, state=1, weighted_density=0.73. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:06:26 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.0000, TRUSTED STATE: ON 

2026-09-24 01:06:27 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:27 [DEBUG] SoundMonitor.power_predictor: Power: -73.13dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:27 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:27 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.00, state=1, weighted_density=0.68. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:06:27 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.0000, TRUSTED STATE: ON 

2026-09-24 01:06:29 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:29 [DEBUG] SoundMonitor.power_predictor: Power: -73.78dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:29 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:29 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.00, state=1, weighted_density=0.62. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:06:29 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.0000, TRUSTED STATE: ON 

2026-09-24 01:06:30 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:30 [DEBUG] SoundMonitor.power_predictor: Power: -74.32dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:30 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:30 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.00, state=1, weighted_density=0.57. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:06:30 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.0000, TRUSTED STATE: ON 

2026-09-24 01:06:31 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:31 [DEBUG] SoundMonitor.power_predictor: Power: -72.95dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:31 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:31 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.00, state=1, weighted_density=0.53. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:06:31 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.0000, TRUSTED STATE: ON 

2026-09-24 01:06:33 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:33 [DEBUG] SoundMonitor.power_predictor: Power: -73.09dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:33 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:33 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.00, state=1, weighted_density=0.48. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:06:33 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.0000, TRUSTED STATE: ON 

2026-09-24 01:06:34 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:34 [DEBUG] SoundMonitor.power_predictor: Power: -73.19dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:34 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:34 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.00, state=1, weighted_density=0.44. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:06:34 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.0000, TRUSTED STATE: ON 

2026-09-24 01:06:35 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:35 [DEBUG] SoundMonitor.power_predictor: Power: -72.02dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:35 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:35 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.00, state=1, weighted_density=0.39. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:06:35 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.0000, TRUSTED STATE: ON 

2026-09-24 01:06:37 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:37 [DEBUG] SoundMonitor.power_predictor: Power: -72.75dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:37 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:37 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.00, state=1, weighted_density=0.35. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:06:37 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.0000, TRUSTED STATE: ON 

2026-09-24 01:06:38 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:38 [DEBUG] SoundMonitor.power_predictor: Power: -72.55dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:38 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:38 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.00, state=1, weighted_density=0.32. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:06:38 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.0000, TRUSTED STATE: ON 
```

## Compressor OFF

```bash
2026-09-24 01:06:39 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:39 [DEBUG] SoundMonitor.power_predictor: Power: -72.54dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:39 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:39 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.00, state=0, weighted_density=0.28. ON/OFF: [ 0.50 | 0.30 ]

2026-09-24 01:06:39 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.0000, TRUSTED STATE: OFF <<<
```

## Add save record to Query for store to DataBase

```bash
2026-09-24 01:06:39 [DEBUG] SoundMonitor.detector: Added to result_queue: 1
2026-09-24 01:06:39 [DEBUG] SoundMonitor.db_writer_worker: [DB] len(batch)=1
```

## Save Live Picture

```bash
2026-09-24 01:06:40 [INFO] SoundMonitor.plot_psd: Saved figure to /app/data/live.png
```

## Saved delayed records from Query for to DataBase

```bash
2026-09-24 01:06:41 [DEBUG] SoundMonitor.analizer: compute_psd welch len(psd)=8193 len(freqs)=8193 nperseg=16384
2026-09-24 01:06:41 [DEBUG] SoundMonitor.power_predictor: Power: -72.39dB, Range for ON-OFF: [ -59.52 | -64.00 | -70.40 ]dB
2026-09-24 01:06:41 [DEBUG] SoundMonitor.detector: success=True, Analyzed frequencies (5): 88.8, 91.5, 94.2, 96.9, 99.6 Hz
2026-09-24 01:06:41 [DEBUG] SoundMonitor.state_filter: SCORED_WEIGHTED_DENSITY : score=0.00, state=0, weighted_density=0.25. ON/OFF: [ 0.50 | 0.30 ]
2026-09-24 01:06:41 [INFO] SoundMonitor.detector: STATE: OFF, SCORE: 0.0000, TRUSTED STATE: OFF
2026-09-24 01:06:41 [DEBUG] SoundMonitor.db_writer_worker: [DB] Saved 1 events.
````
