# lib/expedition/run_state.py
#
# In-memory model layer for one Expedition run.
#
# RunState is the single source of truth for the TUI during a run.  It lives in
# RunScreen (the controller) and is updated as worker subprocesses emit events.
# WaveFinaleScreen and SummaryScreen receive a reference and call its accessor
# methods — they never touch the filesystem.
#
# Workers still write CSV files to /tmp as durable artifacts, but those files
# are read only once per model by the controller (_absorb_csv_row in
# expedition_tui.py) and immediately stored here.  After that the CSV is not
# consulted again by the TUI.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResult:
    """One compile/infer attempt for a single model on a single chip.

    Attributes:
        chip_id:      Zero-based chip index that ran this model.
        model_id:     Model identifier (e.g. "gpt2/pytorch").
        status:       "success" or "failed".
        pts:          Points awarded (positive for success, negative for failure).
        compile_time: Wall-clock compile time in seconds (0.0 on failure).
        artifact:     Human-readable inference output summary, or "".
        first_ever:   True if this is the first successful compile of this model.
        first_voice:  True if the model produced decoded text output this run.
        error:        Error string on failure (may be multi-line), or "".
        rarity:       Rarity tier: "legendary" | "rare" | "uncommon" | "common".
        streak:       Consecutive-success streak count at the time of this result.
        is_sq:        True for side quest (bonus) models; False for main queue.
    """

    chip_id: int
    model_id: str
    status: str
    pts: int
    compile_time: float
    artifact: str
    first_ever: bool
    first_voice: bool
    error: str
    rarity: str
    streak: int
    is_sq: bool = False

    @classmethod
    def from_csv_row(
        cls,
        row: dict[str, str],
        chip_id: int,
        rarity: str,
        streak: int,
        is_sq: bool = False,
    ) -> "ModelResult":
        """Build a ModelResult from a CSV row dict (as produced by DictReader).

        Args:
            row:    Dict from csv.DictReader with the standard _CSV_FIELDNAMES keys.
            chip_id:  Chip that ran this model.
            rarity: Rarity detected from stdout events (not in the CSV).
            streak: Streak at completion time (maintained by ChipRun, not CSV).
            is_sq:  True for side quest models.
        """
        return cls(
            chip_id=chip_id,
            model_id=row.get("model", ""),
            status=row.get("status", "failed"),
            pts=int(row.get("pts") or 0),
            compile_time=float(row.get("compile_time") or 0.0),
            artifact=row.get("artifact", ""),
            first_ever=row.get("first_ever") == "True",
            first_voice=row.get("first_voice") == "True",
            error=row.get("error", ""),
            rarity=rarity,
            streak=streak,
            is_sq=is_sq,
        )


@dataclass
class ChipRun:
    """Mutable state for a single chip during an Expedition run.

    Attributes:
        chip_id:       Zero-based chip index.
        current_model: model_id of the model being compiled right now.
        rarity:        Rarity of the model currently loading, set from stdout
                       before SUCCESS/FAILED is emitted.  Reset to "common"
                       after each result is recorded.
        streak:        Current consecutive-success streak (mutated by RunState.add_result).
        best_streak:   Highest streak seen so far this run.
        results:       Ordered list of all ModelResults for this chip.
    """

    chip_id: int
    current_model: str = ""
    rarity: str = "common"
    streak: int = 0
    best_streak: int = 0
    results: list[ModelResult] = field(default_factory=list)

    @property
    def pts(self) -> int:
        """Cumulative points across all results (main queue + side quests)."""
        return sum(r.pts for r in self.results)

    @property
    def successes(self) -> list[ModelResult]:
        """Main-queue successes only."""
        return [r for r in self.results if r.status == "success" and not r.is_sq]

    @property
    def failures(self) -> list[ModelResult]:
        """Main-queue failures only."""
        return [r for r in self.results if r.status == "failed" and not r.is_sq]

    @property
    def sq_successes(self) -> list[ModelResult]:
        """Side-quest successes."""
        return [r for r in self.results if r.status == "success" and r.is_sq]

    @property
    def sq_failures(self) -> list[ModelResult]:
        """Side-quest failures."""
        return [r for r in self.results if r.status == "failed" and r.is_sq]


class RunState:
    """Authoritative in-memory state for one Expedition run.

    Owned by RunScreen (the controller).  Updated as worker subprocesses
    complete.  WaveFinaleScreen and SummaryScreen receive a reference to this
    object and call read-only accessor methods — they perform no file I/O.

    Args:
        num_chips:  Number of chips in this run (initialises one ChipRun each).
        run_number: Run counter for this Expedition session.
    """

    def __init__(self, num_chips: int, run_number: int) -> None:
        self.num_chips  = num_chips
        self.run_number = run_number
        self._chips: dict[int, ChipRun] = {
            i: ChipRun(chip_id=i) for i in range(num_chips)
        }

    # ------------------------------------------------------------------
    # Controller write-path (called by RunScreen)
    # ------------------------------------------------------------------

    def chip(self, chip_id: int) -> ChipRun:
        """Return the ChipRun for chip_id, creating one if needed."""
        return self._chips.setdefault(chip_id, ChipRun(chip_id=chip_id))

    def set_current(self, chip_id: int, model_id: str) -> None:
        """Record which model is about to start compiling on chip_id.

        Called by RunScreen immediately before launching the worker subprocess,
        so that _parse_for_events can read current_model without a status-file
        read when SUCCESS/FAILED arrives.
        """
        self.chip(chip_id).current_model = model_id

    def set_rarity(self, chip_id: int, rarity: str) -> None:
        """Record rarity detected from the RARITY: stdout event.

        Called by _parse_for_events when the worker reveals the rarity label.
        Reset to "common" after each result is absorbed.
        """
        self.chip(chip_id).rarity = rarity

    def add_result(self, result: ModelResult) -> None:
        """Append a completed ModelResult and update streak/best_streak.

        Args:
            result: Completed ModelResult from _absorb_csv_row.
        """
        c = self.chip(result.chip_id)
        if result.status == "success":
            c.streak      += 1
            c.best_streak  = max(c.best_streak, c.streak)
        else:
            c.streak = 0
        c.results.append(result)

    # ------------------------------------------------------------------
    # View read-path (called by WaveFinaleScreen and SummaryScreen)
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Aggregate stats dict consumed by WaveFinaleScreen._load_stats.

        Replaces the CSV-reading loop that previously lived in _load_stats.

        Returns a dict with keys:
            attempted, compiled, failed, pts, new_models, max_streak, best_rarity
        """
        rarity_rank = {"legendary": 3, "rare": 2, "uncommon": 1, "common": 0}
        total_s = total_f = total_pts = max_streak = new_models = 0
        best_rank  = 0
        best_rarity = "COMPLETE"
        for c in self._chips.values():
            for r in c.results:
                total_pts += r.pts
                if r.status == "success":
                    total_s   += 1
                    if r.first_ever:
                        new_models += 1
                    max_streak = max(max_streak, r.streak)
                    rank = rarity_rank.get(r.rarity, 0)
                    if rank > best_rank:
                        best_rank   = rank
                        best_rarity = r.rarity.upper() + " FINALE"
                else:
                    total_f += 1
        return {
            "attempted":  total_s + total_f,
            "compiled":   total_s,
            "failed":     total_f,
            "points":     total_pts,   # key name matches _wave_frame_tui expectation
            "new_models": new_models,
            "streak":     max_streak,  # key name matches _wave_frame_tui expectation
            "rarity":     best_rarity, # key name matches _wave_frame_tui expectation
        }

    def get_chip_results(self) -> list[dict]:
        """List of per-chip summary dicts for SummaryScreen._populate.

        Returns the same shape that SummaryScreen previously built by reading
        CSV files directly.  Each dict has keys:
            chip_id, pts, successes, failures, first_evers, first_voice, times

        The inner lists contain dicts in the CSV-row format produced by
        _result_to_row so that SummaryScreen display logic is unchanged.

        Sorted descending by pts so the leaderboard is already ordered.
        """
        out: list[dict] = []
        for c in self._chips.values():
            successes = c.successes
            failures  = c.failures
            if not successes and not failures:
                continue
            out.append({
                "chip_id":    c.chip_id,
                "pts":        c.pts,
                "successes":  [self._result_to_row(r) for r in successes],
                "failures":   [self._result_to_row(r) for r in failures],
                "first_evers":[self._result_to_row(r) for r in successes if r.first_ever],
                "first_voice":[self._result_to_row(r) for r in successes if r.first_voice],
                "times":      [r.compile_time for r in successes if r.compile_time],
            })
        out.sort(key=lambda x: -x["pts"])
        return out

    def get_sq_results(self) -> dict[int, dict]:
        """Side quest summary dicts keyed by chip_id for SummaryScreen._populate.

        Returns the same shape previously built from _sq.csv files.  Each value
        has keys: pts, successes, failures.
        """
        out: dict[int, dict] = {}
        for c in self._chips.values():
            sq_s = c.sq_successes
            sq_f = c.sq_failures
            if not sq_s and not sq_f:
                continue
            out[c.chip_id] = {
                "pts":       sum(r.pts for r in sq_s + sq_f),
                "successes": [self._result_to_row(r) for r in sq_s],
                "failures":  [self._result_to_row(r) for r in sq_f],
            }
        return out

    @staticmethod
    def _result_to_row(r: ModelResult) -> dict:
        """Convert a ModelResult to the CSV-row dict shape SummaryScreen expects.

        Keeps the display layer decoupled from the ModelResult dataclass — the
        SummaryScreen code continues to use r.get("model"), r.get("first_ever"),
        etc. without any changes.
        """
        return {
            "model":        r.model_id,
            "status":       r.status,
            "pts":          str(r.pts),
            "compile_time": str(r.compile_time),
            "artifact":     r.artifact,
            "first_ever":   "True" if r.first_ever else "False",
            "first_voice":  "True" if r.first_voice else "False",
            "error":        r.error,
            "rarity":       r.rarity,
            "streak":       str(r.streak),
        }
