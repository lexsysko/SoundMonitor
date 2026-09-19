import logging

import asyncio
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


async def init_db(db_path: Path) -> None:
    async with aiosqlite.connect(db_path) as db:
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
    db_path: Path,
) -> None:
    await init_db(db_path)
    logger.info(f"  SQLite writer ready → {db_path}")

    async with aiosqlite.connect(db_path) as db:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    queue.task_done()
                    break

                await db.execute(
                    """
                    INSERT INTO events (timestamp, state, d_on, d_off)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        item.timestamp,
                        int(item.state),
                        item.d_on,
                        item.d_off,
                    ),
                )
                await db.commit()
                logger.debug(f"[db worker] saved data: {item}")
            except Exception as e:
                logger.error(f"[db worker] error: {e}")
            finally:
                try:
                    queue.task_done()
                except ValueError:
                    ...

    logger.info("  SQLite writer stopped.")
