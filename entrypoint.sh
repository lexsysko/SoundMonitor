#!/bin/sh

# Check for 'free-threading' in sys.version or hasattr sys._is_gil_enabled
if python -c "import sys; exit(0 if 'free-threading' in sys.version else 1)"; then
  echo "Free-threaded build detected. Disabling GIL..."
  export PYTHON_GIL=0
fi


if [ "${SET_MIXER:-}" = "1" ]; then
 echo "SETUP AUDIO MIXER..."
 amixer -c ${DEVICE:-1} sset 'Mic Boost' ${MIC_BOOST:-0}
 amixer -c ${DEVICE:-1} sset 'Internal Mic Boost' ${INTERNAL_MIC_BOOST:-0}
 amixer -c ${DEVICE:-1} sset 'Capture' ${CAPTURE_VOLUME:-90}% cap
 amixer -c ${DEVICE:-1} sset Mic 100% mute
 amixer -c ${DEVICE:-1} sset 'Speaker' ${SPEAKER_VOLUME:-40}%
fi


if [ "${WAITER:-}" = "1" ]; then
 echo "RUNNING TERMINAL INFINITE WAITER..."
 exec tail -f /dev/null
 exit
fi

if [ -n "${THRESHOLD:-}" ]; then
  THRESHOLD=" --threshold ${THRESHOLD}"
fi

if [ -n "${DEVICE:-}" ]; then
  DEVICE=" --device ${DEVICE}"
fi

if [ -n "${DETECT_WINDOW:-}" ]; then
  DETECT_WINDOW=" --window ${DETECT_WINDOW}"
fi

if [ -n "${LOGLEVEL:-WARNING}" ]; then
  LOGLEVEL=" --loglevel ${LOGLEVEL}"
fi

if [ "${PLOT:-}" = "1" ]; then
  PLOT=" --plot"
fi

if [ "${BEEP:-}" = "1" ]; then
  BEEP=" --beep"
fi

echo "\n\nRUNNING detect ${LOGLEVEL}${THRESHOLD:-}${PLOT:-}${DEVICE:-}${BEEP:-}${DETECT_WINDOW:-} ..."
exec python /app/src/SoundMonitor/main.py ${LOGLEVEL} detect${THRESHOLD:-}${PLOT:-}${DEVICE:-}${BEEP:-}${DETECT_WINDOW:-}