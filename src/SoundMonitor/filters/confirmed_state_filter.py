"""
Second-stage filter to sit on top of SoundMonitor.state_filter.StateFilter.

StateFilter already smooths frame-level noise into a state (density /
streak / weighted-density). This module adds event-level protection
against two separate problems:

1. A false transition getting accepted just because enough time passed
   since the last real one, silently eating a later real transition:

       :00  real transition, confirmed
       :11  noise -> filter thinks it flipped again
       :13  noise clears -> filter flips back
       :20  the device genuinely transitions for real

   Fixed by requiring a candidate to persist UNBROKEN for
   `confirm_duration` before it's even eligible - if it reverts to the
   confirmed state first, it's discarded with no cooldown ever applied.

2. Using the wrong clock for duration math. `min_interval` and
   `confirm_duration` are measured on a MONOTONIC clock, which never
   jumps backward or forward (NTP corrections, leap seconds, an admin
   fixing the system clock can all move wall time - monotonic time
   never moves). The event's stored timestamp uses `time.time()`
   (wall-clock, NTP-synced, already Unix epoch seconds - the same
   representation your DB worker uses, so no datetime/timezone
   conversion layer is needed anywhere in this module). The two clocks
   are read together, at the instant a candidate starts, so the epoch
   timestamp stays correctly paired with the monotonic instant it
   corresponds to.

Deliberately decoupled from StateFilter (composition, not inheritance):

    raw_smoothed = state_filter.update(frame, score)
    commit = confirmed_filter.update(raw_smoothed)
    if commit is not None:
        db.insert(state=commit.state, event_at=commit.event_at,
                   recorded_at=commit.committed_at)  # both plain Unix ts (float)
"""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StateCommit:
    state: int
    event_at: float  # Unix ts (wall-clock): true onset for "change", now() for "baseline"/"heartbeat"
    committed_at: float  # Unix ts (wall-clock) the filter confirmed/emitted it - audit/latency only
    kind: str = "change"  # "change" | "baseline" | "heartbeat"


class ConfirmedStateFilter:
    def __init__(
        self,
        confirm_duration: float = 120.0,
        min_interval: float = 600.0,
        initial_state: int = 0,
        mono_clock=time.monotonic,
        wall_clock=time.time,
    ):
        """
        :param confirm_duration: seconds (monotonic) a candidate must
            persist, unbroken, before it's eligible for commit. Set
            this above the longest false blip you actually observe.
        :param min_interval: minimum seconds (monotonic) between
            COMMITTED transitions - your physical "can't cycle faster
            than this" constraint.
        :param mono_clock: monotonic time source, used for ALL interval
            math. Injectable for tests.
        :param wall_clock: wall-clock (NTP, Unix epoch seconds) time
            source, used ONLY to label events for storage - never for
            duration comparisons. Injectable for tests.
        """
        self.confirm_duration = confirm_duration
        self.min_interval = min_interval
        self._mono = mono_clock
        self._wall = wall_clock

        self.confirmed_state = initial_state
        self._confirmed_at_mono: float | None = None
        self._last_write_at_mono: float | None = None

        self._pending_state: int | None = None
        self._pending_since_mono: float | None = None
        self._pending_since_wall: float | None = None

    def update(self, smoothed_state: int) -> StateCommit | None:
        """
        Feed in StateFilter's (already frame-smoothed) state. Returns a
        StateCommit ONLY on the tick a transition is actually confirmed
        (write this row) - None every other tick, including while a
        candidate is pending or was just discarded as noise.
        """
        mono_now = self._mono()

        # logger.debug(
        #     f"{smoothed_state=} {self.confirmed_state=}  {self._pending_state=} {self._pending_since_mono=} {self._pending_since_wall=} "
        # )

        if smoothed_state == self.confirmed_state:
            # Matches what's already confirmed -> any pending candidate
            # for the opposite state was noise. Discard it, no cooldown
            # applied - nothing was ever committed.
            if self._pending_state is not None:
                logger.debug(
                    f"discarding pending candidate {self._pending_state} "
                    f"(reverted to confirmed state {self.confirmed_state})"
                )
            self._pending_state = None
            self._pending_since_mono = None
            self._pending_since_wall = None
            # logger.debug(f"noting changed:  smoothed_state == confirmed_state")

            return None

        if self._pending_state != smoothed_state:
            # New candidate - capture BOTH clocks together, right now,
            # so the epoch-time label stays correctly paired with the
            # monotonic instant it corresponds to.
            self._pending_state = smoothed_state
            self._pending_since_mono = mono_now
            self._pending_since_wall = self._wall()
            logger.debug(f"new candidate {smoothed_state}")
            return None

        # Same candidate persisting - check both gates, monotonic only.
        held_for = mono_now - self._pending_since_mono
        since_last_change = float("inf") if self._confirmed_at_mono is None else mono_now - self._confirmed_at_mono

        logger.debug(f"{held_for=} {since_last_change=}  {self.min_interval=} {self.confirm_duration=}")

        if held_for >= self.confirm_duration and since_last_change >= self.min_interval:
            event_at = self._pending_since_wall  # true onset, Unix ts
            committed_at = self._wall()
            logger.debug(
                f"COMMIT {self.confirmed_state}->{smoothed_state}: "
                f"event_at={event_at:.0f} "
                f"(held {held_for:.0f}s, {since_last_change:.0f}s since last change)"
            )
            self.confirmed_state = smoothed_state
            self._confirmed_at_mono = mono_now
            self._pending_state = None
            self._pending_since_mono = None
            self._pending_since_wall = None
            self._last_write_at_mono = mono_now
            return StateCommit(state=smoothed_state, event_at=event_at, committed_at=committed_at, kind="change")

        return None

    def initial_commit(self) -> StateCommit:
        """
        Call once at startup to seed the DB with a baseline row. update()
        only ever returns a commit when the state actually transitions,
        so a device that stays in its `initial_state` forever would
        otherwise never get a single row written. kind="baseline" marks
        it as a snapshot, not a transition.
        """
        now = self._wall()
        self._confirmed_at_mono = self._mono()
        self._last_write_at_mono = self._confirmed_at_mono
        return StateCommit(state=self.confirmed_state, event_at=now, committed_at=now, kind="baseline")

    def heartbeat(self, min_gap: float = 60.0) -> StateCommit | None:
        """
        Periodic confirmation write of the current TRUSTED state - not a
        new transition. Call this on your own timer (e.g. every 2 min).

        Returns None:
          - while a candidate transition is pending (state isn't fully
            trusted yet - same rule as update()), or
          - if a commit (change/baseline/heartbeat) was already emitted
            less than `min_gap` seconds ago, to avoid a near-duplicate
            row when a real change happens to land close to a heartbeat
            tick.

        Timestamps both event_at and committed_at as "now": a heartbeat
        isn't reporting a new onset, just "state X still holds as of
        this moment" - so it's always chronologically at or after the
        last change's event_at, never conflicting with it.
        """
        if self._pending_state is not None:
            return None

        mono_now = self._mono()
        if self._last_write_at_mono is not None and (mono_now - self._last_write_at_mono) < min_gap:
            return None

        now = self._wall()
        self._last_write_at_mono = mono_now
        return StateCommit(state=self.confirmed_state, event_at=now, committed_at=now, kind="heartbeat")


if __name__ == "__main__":
    # Fake clock pair that advances in lockstep, so the demo never
    # touches the real wall or monotonic clock but the pairing is exact.
    class FakeClocks:
        def __init__(self, start_wall: float):
            self._mono_t = 0.0
            self._wall_t = start_wall

        def mono(self):
            return self._mono_t

        def wall(self):
            return self._wall_t

        def advance(self, seconds):
            self._mono_t += seconds
            self._wall_t += seconds

    start = 1758729600.0  # arbitrary Unix ts, e.g. 2026-09-24 18:00:00 UTC
    clocks = FakeClocks(start)
    f = ConfirmedStateFilter(
        confirm_duration=120,
        min_interval=600,
        initial_state=0,
        mono_clock=clocks.mono,
        wall_clock=clocks.wall,
    )

    # Seed the DB once at startup - not a "change", just a baseline.
    baseline = f.initial_commit()
    print(f"t=:00  -> WRITE ({baseline.kind}) state={baseline.state}")

    # (minutes to advance by, raw smoothed state)
    events = [(0, 0), (11, 1), (2, 0), (7, 1), (2, 1), (2, 1), (6, 1)]
    # -> t=:00 :11 :13 :20 :22 :24 :30

    t = 0
    for dt_minutes, raw in events:
        t += dt_minutes
        if dt_minutes:
            clocks.advance(dt_minutes * 60)
        commit = f.update(raw)
        if commit is not None:
            print(
                f"t=:{t:02d}  raw_in={raw}  -> WRITE ({commit.kind}) state={commit.state} "
                f"event_at={commit.event_at:.0f} committed_at={commit.committed_at:.0f}"
            )
        else:
            print(f"t=:{t:02d}  raw_in={raw}  -> (no write)")

        # A heartbeat tick lands at :24 (2 min after the :22 change commit) -
        # min_gap=60s means it's NOT suppressed (120s have passed), so it
        # writes normally, reasserting the already-confirmed state.
        if t == 24:
            hb = f.heartbeat(min_gap=60)
            print(f"t=:{t:02d}  (heartbeat tick) -> {'WRITE ' + hb.kind if hb else '(suppressed)'}")
