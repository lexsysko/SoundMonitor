import asyncio

import logging
import signal

logger = logging.getLogger(__name__)


def setup_signal_handlers(
    shutdown_event: asyncio.Event,
    loop=None,
):
    """
    Cross-platform signal handler setup.
    Works on both Linux/macOS (Docker) and Windows natively.
    """

    def handle_signal(sig, frame=None):
        logger.info(f"\n[System] Received signal {sig}. Triggering graceful shutdown...")
        # Check if loop is running and thread-safely set the shutdown flag/event
        if loop and loop.is_running():
            loop.call_soon_threadsafe(shutdown_event.set)
        else:
            shutdown_event.set()

    # Standard signal bindings compatible with Windows and Linux
    signal.signal(signal.SIGINT, handle_signal)  # Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal)  # Docker stop / termination
