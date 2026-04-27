from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Bestiary:
    """Persistent store for all Expedition Mode compilation history.

    Loaded from and saved to a JSON file. Three top-level sections:
      compiled    — keyed by model_id, one entry per successfully compiled model
      failed      — keyed by model_id, retry-tracking only (no scoring penalty beyond initial -10)
      chip_totals — keyed by str(chip_index), cumulative all-time scores

    The file is NOT auto-saved on mutation; callers must invoke save() explicitly
    (typically at the end of a run or after each batch of mutations to avoid data
    loss on crash).
    """

    def __init__(self, path: Path | str = "data/bestiary.json") -> None:
        self.path = Path(path)
        self._data: dict[str, Any] = self._load()

    # ── public accessors ──────────────────────────────────────────────────────

    @property
    def compiled(self) -> dict[str, Any]:
        """All successfully compiled models, keyed by model_id."""
        return self._data["compiled"]

    @property
    def failed(self) -> dict[str, Any]:
        """All models that failed to compile, keyed by model_id."""
        return self._data["failed"]

    @property
    def chip_totals(self) -> dict[str, Any]:
        """Cumulative per-chip scores, keyed by str(chip_index)."""
        return self._data["chip_totals"]

    def is_compiled(self, model_id: str) -> bool:
        """Return True if this model has at least one successful compilation recorded."""
        return model_id in self._data["compiled"]

    # ── mutation ──────────────────────────────────────────────────────────────

    def record_success(
        self,
        model_id: str,
        chip: int,
        run: int,
        time_s: float,
        task: str,
        source: str,
        rarity: str,
        hf_downloads: int | None,
        hf_created_at: str | None,
        artifact: str,
    ) -> None:
        """Record a successful compilation.

        On first success, creates the compiled entry with all metadata. On
        subsequent calls, increments counters and updates best_time_s if the
        new run was faster. The artifact field is always overwritten with the
        most recent decoded inference output.

        Args:
            model_id:      HuggingFace model identifier (e.g. "openai/whisper-large-v3").
            chip:          Zero-based index of the Tenstorrent chip that ran this model.
            run:           Sequential run number within the current expedition session.
            time_s:        Wall-clock compilation + inference time in seconds.
            task:          HuggingFace pipeline task string (e.g. "automatic_speech_recognition").
            source:        Data origin: "huggingface", "local", etc.
            rarity:        Rarity tier from scorer.py: "common", "uncommon", "rare", "legendary".
            hf_downloads:  Monthly downloads from HuggingFace model card (None if unavailable).
            hf_created_at: ISO-8601 creation timestamp from HuggingFace (None if unavailable).
            artifact:      Decoded inference output string — the model's "voice" in the bestiary.
        """
        now = datetime.now(timezone.utc).isoformat()
        if model_id not in self._data["compiled"]:
            # First-time entry: capture all immutable metadata from this run.
            self._data["compiled"][model_id] = {
                "first_compiled": now,
                "first_chip": chip,
                "run": run,
                "best_time_s": time_s,
                "attempts": 0,
                "successes": 0,
                "source": source,
                "task": task,
                "rarity": rarity,
                "hf_downloads": hf_downloads,
                "hf_created_at": hf_created_at,
                "artifact": artifact,
            }
        entry = self._data["compiled"][model_id]
        entry["attempts"] += 1
        entry["successes"] += 1
        # Track the fastest compilation time across all chips/runs.
        if time_s < entry["best_time_s"]:
            entry["best_time_s"] = time_s
        # Always update artifact so the bestiary reflects the most recent output.
        entry["artifact"] = artifact

    def record_failure(self, model_id: str, run: int, error: str) -> None:
        """Track a failed compilation attempt for retry-interest purposes.

        Failures are stored separately from compiled models. They are NOT used
        for score penalties (the -10 penalty is applied by the caller at runtime),
        but they inform future run decisions (e.g. skip models that always fail).

        Args:
            model_id: HuggingFace model identifier.
            run:      Sequential run number within the current expedition session.
            error:    String representation of the exception or error message.
        """
        if model_id not in self._data["failed"]:
            self._data["failed"][model_id] = {
                "run_first_failed": run,
                "attempts": 0,
                "last_error": "",
            }
        entry = self._data["failed"][model_id]
        entry["attempts"] += 1
        entry["last_error"] = error

    def add_chip_points(
        self, chip: int, pts: int, first_ever: bool, streak: int
    ) -> None:
        """Accumulate points and statistics into the all-time chip leaderboard.

        Called after each scored compilation event. Chip totals persist across
        sessions and accumulate indefinitely, enabling long-term rivalry tracking.

        Args:
            chip:       Zero-based chip index.
            pts:        Points earned in this event (may be negative for failures).
            first_ever: True if this was the first-ever compilation of the model
                        across all runs (triggers "first blood" bonus tracking).
            streak:     Current success streak length for this chip. Updates
                        best_streak if it exceeds the stored record.
        """
        key = str(chip)
        if key not in self._data["chip_totals"]:
            self._data["chip_totals"][key] = {
                "pts": 0,
                "first_evers": 0,
                "best_streak": 0,
            }
        totals = self._data["chip_totals"][key]
        totals["pts"] += pts
        if first_ever:
            totals["first_evers"] += 1
        if streak > totals["best_streak"]:
            totals["best_streak"] = streak

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Write current state to disk, creating parent directories as needed.

        The file is written atomically via write_text, which replaces the old
        file in a single OS call. Not crash-safe for very large files, but
        acceptable for the typical bestiary size (<10 MB).
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

    # ── private ───────────────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        """Load state from disk, or return a fresh empty state on any error.

        Handles:
          - Missing file (first run): returns empty dicts.
          - Corrupt/truncated JSON: falls back to empty state rather than crashing.
          - OS read errors (permissions, etc.): same fallback.
        """
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {"compiled": {}, "failed": {}, "chip_totals": {}}
