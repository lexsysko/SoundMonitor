#!/bin/sh

# Check for 'free-threading' in sys.version or hasattr sys._is_gil_enabled
if python -c "import sys; exit(0 if 'free-threading' in sys.version else 1)"; then
  echo "Free-threaded build detected. Disabling GIL..."
  export PYTHON_GIL=0
fi


if [ "${SET_MIXER:-}" = "1" ]; then
 echo "SETUP AUDIO MIXER..."
 amixer -c 1 sset 'Mic Boost' 0%
 amixer -c 1 sset 'Internal Mic Boost' 0%
 amixer -c 1 sset 'Capture' 80% cap
 amixer -c 1 sset Mic 100% mute
fi


if [ "${WAITER:-}" = "1" ]; then
 echo "RUNNING TERMINAL INFINITE WAITER..."
 exec tail -f /dev/null
 exit
fi

if [ -n "${THRESHOLD:-}" ]; then
  THRESHOLD=" --threshold ${THRESHOLD}"
fi

if [ "${PLOT:-}" = "1" ]; then
  PLOT=" --plot"
fi

echo "\n\nRUNNING detect --loglevel=${LOGLEVEL:-WARNING}${THRESHOLD:-}${PLOT:-}..."
exec python /app/src/SoundMonitor/main.py --loglevel=${LOGLEVEL:-WARNING} detect${THRESHOLD:-}${PLOT:-}