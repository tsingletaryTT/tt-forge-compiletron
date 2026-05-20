# lib/expedition/hud.py
#
# Per-chip score state and status file IPC for Expedition Mode.
#
# ChipState holds all mutable state for a single chip's run.
# ChipHUD wraps ChipState and provides methods to record events,
# then serialise state to a simple key=value status file that
# the shared status-strip display reads via EXPEDITION_STATUS_DIR.
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from lib.expedition.scorer import ScoreResult


# Environment variable that controls where status files are written.
# Defaults to /tmp so it always works without explicit configuration.
_STATUS_DIR_ENV = "EXPEDITION_STATUS_DIR"
_DEFAULT_STATUS_DIR = "/tmp"

# Points deducted from the running total on each compilation failure.
_FAILURE_PENALTY = 10


@dataclass
class ChipState:
    """All mutable state for a single chip's Expedition run.

    Attributes:
        chip_id:        Zero-based chip index.
        run_number:     Current expedition run number (written to status file so
                        a resumed per-model worker can distinguish same-run state
                        from a stale file left by a previous run).
        pts:            Cumulative score (successes add, failures deduct).
        streak:         Current consecutive-success streak.
        best_streak:    Highest streak seen so far in this run.
        successes:      Total successful compilations.
        failures:       Total failed compilations.
        current_model:  Model identifier currently being compiled.
        current_index:  Zero-based position in the model queue.
        total_models:   Total number of models in this Expedition.
        done:           True once the chip has finished all its models.
    """
    chip_id: int
    run_number: int = 0
    pts: int = 0
    streak: int = 0
    best_streak: int = 0
    successes: int = 0
    failures: int = 0
    current_model: str = ""
    current_index: int = 0
    total_models: int = 0
    done: bool = False


class ChipHUD:
    """Head-Up Display for one chip in an Expedition run.

    Tracks score, streak, and model progress, then serialises state to a
    ``expedition_chip_<id>.status`` file whenever ``write_status()`` is called.

    The status files are read by the shared status-strip display pane; using
    flat key=value text means the reader has no library dependency and the
    format is trivially grep-able for debugging.

    Args:
        chip_id:       Zero-based index identifying which chip this HUD tracks.
        total_models:  Total number of models scheduled for this Expedition run.
    """

    def __init__(self, chip_id: int, total_models: int, run_number: int = 0) -> None:
        self._state = ChipState(chip_id=chip_id, total_models=total_models,
                                run_number=run_number)
        # When the TUI dispatches one model per subprocess invocation, each new
        # worker must carry forward the pts/successes/failures/streak accumulated
        # by earlier models on the same chip.  We do this by reading the existing
        # status file if (and only if) its run_number matches — which guarantees
        # we're resuming within the same run rather than inheriting a stale file
        # from the previous expedition.
        if run_number > 0:
            self._resume_accumulated_state()

    # ------------------------------------------------------------------
    # Public read-only access
    # ------------------------------------------------------------------

    @property
    def state(self) -> ChipState:
        """Return the current ChipState (read-only by convention)."""
        return self._state

    # ------------------------------------------------------------------
    # State mutators
    # ------------------------------------------------------------------

    def set_current(self, model_id: str, index: int) -> None:
        """Record which model is about to be compiled.

        Args:
            model_id: Human-readable model identifier (e.g. ``"bert/qa"``).
            index:    Zero-based position in the model queue.
        """
        self._state.current_model = model_id
        self._state.current_index = index

    def record_success(self, model_id: str, score: ScoreResult) -> None:
        """Apply a successful compilation result to the running totals.

        Adds ``score.pts`` to the chip's cumulative points, increments the
        success counter, and extends the current streak (updating best_streak
        if the new value is higher).

        **Call contract:** call either ``record_success`` OR ``record_failure``
        for each compile attempt — never both. Do NOT pass a failure-path
        ScoreResult (pts=-10) to this method; that would double-count the
        penalty since ``record_failure`` deducts independently.

        Args:
            model_id: Identifier of the model that compiled successfully.
            score:    ScoreResult produced by the scorer for this compilation.
        """
        self._state.pts += score.pts
        self._state.successes += 1
        self._state.streak += 1
        self._state.best_streak = max(self._state.best_streak, self._state.streak)

    def record_failure(self, model_id: str) -> None:
        """Apply a failed compilation to the running totals.

        Deducts ``_FAILURE_PENALTY`` points, increments the failure counter,
        and resets the current streak to zero (best_streak is preserved).

        Args:
            model_id: Identifier of the model that failed to compile.
        """
        self._state.pts -= _FAILURE_PENALTY
        self._state.failures += 1
        self._state.streak = 0

    def mark_done(self) -> None:
        """Signal that this chip has finished all its models for the run."""
        self._state.done = True

    # ------------------------------------------------------------------
    # Status file I/O
    # ------------------------------------------------------------------

    def _resume_accumulated_state(self) -> None:
        """Load accumulated run state from the existing status file, if any.

        Only carries forward state when the status file's run number matches
        the current run_number, ensuring stale files from previous expeditions
        are ignored.  Silently does nothing on any I/O or parse error.
        """
        status_dir = os.environ.get(_STATUS_DIR_ENV, _DEFAULT_STATUS_DIR)
        path = Path(status_dir) / f"expedition_chip_{self._state.chip_id}.status"
        try:
            data: dict[str, str] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    data[k.strip()] = v.strip()
            if data.get("run") != str(self._state.run_number):
                return  # stale file from a different run — start fresh
            self._state.pts         = int(data.get("pts",         0))
            self._state.successes   = int(data.get("successes",   0))
            self._state.failures    = int(data.get("failures",    0))
            self._state.streak      = int(data.get("streak",      0))
            self._state.best_streak = int(data.get("best_streak", 0))
            # Carry forward the run total so per-model dispatch doesn't reset it
            # to 1 on every subprocess invocation.  Only override when the file
            # has a larger total — protects against a stale 1 being restored.
            stored_total = int(data.get("total", 0))
            if stored_total > self._state.total_models:
                self._state.total_models = stored_total
        except Exception:
            pass  # missing file, parse error, etc. — fresh state is fine

    def write_status(self) -> None:
        """Serialise current state to a flat key=value status file.

        The output path is::

            <EXPEDITION_STATUS_DIR>/expedition_chip_<chip_id>.status

        where ``EXPEDITION_STATUS_DIR`` defaults to ``/tmp``.  Each line
        has the form ``key=value`` with no spaces around ``=``.

        This file is polled by the shared status-strip display pane, so
        write_status() should be called after every state-changing event.
        """
        status_dir = os.environ.get(_STATUS_DIR_ENV, _DEFAULT_STATUS_DIR)
        path = Path(status_dir) / f"expedition_chip_{self._state.chip_id}.status"
        s = self._state
        lines = [
            f"chip_id={s.chip_id}",
            f"run={s.run_number}",
            f"current={s.current_index}",
            f"total={s.total_models}",
            f"successes={s.successes}",
            f"failures={s.failures}",
            f"pts={s.pts}",
            f"streak={s.streak}",
            f"best_streak={s.best_streak}",
            f"model={s.current_model}",
            f"done={1 if s.done else 0}",
        ]
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(path)
