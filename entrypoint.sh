#!/bin/sh

if [ -n "${WAITER:-}" ]; then
 echo "RUNNING TERMINAL INFINITE WAITER..."
 exec tail -f /dev/null
 exit
fi

exec python -u /app/src/SoundMonitor/main.py