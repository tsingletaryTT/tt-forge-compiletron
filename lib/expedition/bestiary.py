from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sanitize_model_id(model_id: str) -> str:
    """Convert a model ID to a safe filename (slashes and spaces → underscores)."""
    return re.sub(r"[\s/]+", "_", model_id).strip("_")


class Bestiary:
    """Persistent store for all Expedition Mode compilation history.

    Loaded from and saved to a JSON file. Three top-level sections:
      compiled    — keyed by model_id, one entry per successfully compiled model
      failed      — keyed by model_id, retry-tracking only (no scoring penalty beyond initial -10)
      chip_totals — keyed by str(chip_index), cumulative all-time scores

    Note: `compiled` and `failed` are NOT mutually exclusive. A model that
    fails first and later succeeds appears in both dicts. Use is_compiled()
    to determine whether any successful compilation exists.

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
                "successes": 0,
                "source": source,
                "task": task,
                "rarity": rarity,
                "hf_downloads": hf_downloads,
                "hf_created_at": hf_created_at,
                "artifact": artifact,
            }
        entry = self._data["compiled"][model_id]
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
        """Write current state to disk via an atomic rename.

        Writes to a sibling .tmp file first, then renames into place, so a
        crash during save never leaves a truncated or corrupt bestiary.json.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def save_artifact(
        self,
        model_id: str,
        task: str,
        compiled_at: str,
        chip: int,
        run: int,
        artifact_text: str,
        artifacts_dir: Path | str = "data/artifacts",
    ) -> Path:
        """Write a plain-text artifact file for a model compilation result.

        First line is a metadata header; remaining lines are the artifact content.
        Both successful compilations and failures can use this method — the caller
        passes the inference output string or error message as artifact_text.
        Returns the path written.

        Args:
            model_id:      HuggingFace model identifier (e.g. "openai/whisper-large-v3").
            task:          HuggingFace pipeline task string.
            compiled_at:   ISO-8601 timestamp of the compilation event.
            chip:          Zero-based index of the Tenstorrent chip used.
            run:           Sequential run number within the current expedition session.
            artifact_text: The decoded inference output or error message to persist.
            artifacts_dir: Directory in which to write the .txt file (created if absent).
        """
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        filename = _sanitize_model_id(model_id) + ".txt"
        path = artifacts_dir / filename
        header = f"{model_id} · {task} · {compiled_at} · chip-{chip} · run-{run}"
        path.write_text(f"{header}\n{artifact_text}", encoding="utf-8")
        return path

    def load_artifact(
        self,
        model_id: str,
        artifacts_dir: Path | str = "data/artifacts",
    ) -> str | None:
        """Read and return the full artifact file content, or None if not found.

        Args:
            model_id:      HuggingFace model identifier used when the artifact was saved.
            artifacts_dir: Directory to search for the .txt file.
        """
        path = Path(artifacts_dir) / (_sanitize_model_id(model_id) + ".txt")
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

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
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for key in ("compiled", "failed", "chip_totals"):
                    data.setdefault(key, {})
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"compiled": {}, "failed": {}, "chip_totals": {}}
