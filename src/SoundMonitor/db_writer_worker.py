import time

import datetime

from contextlib import asynccontextmanager

import logging

import asyncio
from pathlib import Path

import aiosqlite

from SoundMonitor.settings import DB_FILE, BATCH_FLUSH_DB_TIMEOUT, CLEANUP_TIMEOUT, CLEANUP_PERIOD_DAYS

logger = logging.getLogger(__name__)


INSERT_EVENTS_DATA_SQL = """ 
                    INSERT INTO events (timestamp, state, d_on, d_off)
                    VALUES (?, ?, ?, ?)
                    """


@asynccontextmanager
async def get_db_connection(db_path: Path | None = None):
    """Centralized database connection provider with optimized PRAGMAs."""
    path = db_path or DB_FILE
    async with aiosqlite.connect(path) as db:
        # Standardize performance & concurrency settings across all connections
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA busy_timeout=5000;")
        await db.execute("PRAGMA synchronous=NORMAL;")
        yield db


async def init_db(db_path: Path | None = None) -> None:
    async with get_db_connection(db_path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL    NOT NULL,
                state       INTEGER NOT NULL,
                d_on        REAL    NOT NULL,
                d_off       REAL    NOT NULL
            )
            """
        )
        await db.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA busy_timeout=5000;")
        await db.execute("PRAGMA synchronous=NORMAL;")
        await db.commit()


async def db_writer_worker(
    queue: asyncio.Queue,
    shutdown_event: asyncio.Event,
    db_path: Path | None = None,
) -> None:
    await init_db(db_path)
    logger.info(f"  SQLite writer ready")
    batch = []

    async with get_db_connection(db_path) as db:

        async def flush_batch():
            if not batch:
                return
            await db.executemany(INSERT_EVENTS_DATA_SQL, batch)
            await db.commit()
            logger.debug(f"[DB] Saved {len(batch)} events.")
            batch.clear()

        try:
            while not (shutdown_event.is_set() and queue.empty()):
                try:
                    get_task = asyncio.create_task(queue.get())
                    shutdown_task = asyncio.create_task(shutdown_event.wait())

                    done, pending = await asyncio.wait(
                        {get_task, shutdown_task},
                        timeout=BATCH_FLUSH_DB_TIMEOUT,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Clean up tasks that did not complete
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    if get_task in done:
                        item = get_task.result()
                        batch.append(item)
                        queue.task_done()
                    elif shutdown_task in done:
                        logger.info("[DB] Shutdown event received, stopping worker...")
                    else:
                        # Timed out without getting an item or shutdown signal
                        await flush_batch()

                    if len(batch) >= 10:
                        await flush_batch()

                except asyncio.TimeoutError:
                    await flush_batch()

        finally:
            # Drain residual items from queue during app shutdown
            while not queue.empty():
                try:
                    batch.append(queue.get_nowait())
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break

            await flush_batch()
            logger.info("[DB] Writer worker shut down cleanly.")

    logger.info("  SQLite writer stopped.")


async def async_cleanup(cutoff_timestamp: float, db_path: Path | None = None) -> int:
    """Deletes records older than cutoff_timestamp and returns the deleted row count."""
    async with get_db_connection(db_path) as db:
        cursor = await db.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_timestamp,))
        await db.commit()
        return cursor.rowcount


async def db_cleanup_worker(shutdown_event: asyncio.Event, db_path: Path, cleanup_timeout=None):
    cleanup_timeout = cleanup_timeout or CLEANUP_TIMEOUT
    cleanup_period = datetime.timedelta(days=CLEANUP_PERIOD_DAYS).total_seconds()

    if not cleanup_period:
        logger.info("[DB] DB WORKER FOR CLEANUP IS DISABLED")
        return
    logger.info(f"[DB] CLEANUP initialized every {cleanup_timeout} seconds.")

    while not shutdown_event.is_set():
        try:
            cutoff_timestamp = time.time() - cleanup_period
            deleted_count = await async_cleanup(cutoff_timestamp, db_path)

            if deleted_count:
                logger.info(f"[DB] Cleanup finished. Deleted {deleted_count} old sensor records.")

        except Exception as e:
            logger.error(f"[DB] Error during database cleanup: {e}", exc_info=True)

        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=cleanup_timeout)
        except asyncio.TimeoutError:
            ...

    logger.info("[DB] Cleanup worker shut down cleanly.")
