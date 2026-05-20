from __future__ import annotations
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sanitize_model_id(model_id: str) -> str:
    """Convert a model ID to a safe filename (slashes and spaces → underscores)."""
    return re.sub(r"[\s/]+", "_", model_id).strip("_")


# ── Error classification ───────────────────────────────────────────────────────
# Ordered rules: first match wins.  Each entry is
#   (substring_to_find, category_key, display_label, action_hint)
# Substrings are matched case-insensitively against the full error string.
# The ordering matters: put more specific / less ambiguous patterns first.

_ERROR_RULES: list[tuple[str, str, str, str]] = [
    # Forge tracing limitation — the most common fixable failure
    ("Tracer cannot infer type",
     "tracer_output_type",
     "Tracer output type",
     "fix: set return_dict=False on the model before tracing"),
    # Models requiring trust_remote_code — "code" may be cut off in older stored errors
    ("contains custom",
     "custom_code",
     "Custom model code",
     "fix: add trust_remote_code=True to AutoModel.from_pretrained()"),
    ("trust_remote_code",
     "custom_code",
     "Custom model code",
     "fix: add trust_remote_code=True to AutoModel.from_pretrained()"),
    # Unsupported architecture in the installed transformers version
    # "but Transfo" matches even if "rmers" was cut by the old 80-char truncation
    ("but Transfo",
     "unsupported_arch",
     "Unsupported architecture",
     "need: upgrade transformers or add forge ops for this arch"),
    # model_type absent from config.json — most common catch-all
    ("Unrecognized model",
     "unknown_arch",
     "Unknown architecture",
     "filter: check for model_type in config before queuing"),
    ("model_type",
     "unknown_arch",
     "Unknown architecture",
     "filter: check for model_type in config before queuing"),
    # Quantised weights in formats we can't compile
    ("quantization",
     "quantized_format",
     "Quantized model format",
     "filter: extend format blocklist in hf_discover.py"),
    ("torchao",
     "quantized_format",
     "Quantized model format",
     "filter: extend format blocklist in hf_discover.py"),
    ("bitsandbytes",
     "quantized_format",
     "Quantized model format",
     "filter: extend format blocklist in hf_discover.py"),
    # Our loader builder couldn't find a path for this pipeline tag
    ("Cannot build dynamic loader",
     "loader_missing",
     "No loader available",
     "fix: extend build_dynamic_loader() in hf_discover.py"),
    # Required optional package not installed (e.g. mamba-ssm, flash-attn).
    # Matches the HuggingFace transformers pattern "X is required by the Y model".
    ("is required by the",
     "missing_dependency",
     "Missing optional dependency",
     "fix: pip install the required package or exclude these model types"),
    # Raw ModuleNotFoundError / ImportError for packages that aren't in the venv
    # (e.g. FlagEmbedding, flash-attn, apex).  Distinct from the above because
    # these come from the model code itself, not from transformers' guards.
    ("modulenot",
     "missing_dependency",
     "Missing optional dependency",
     "fix: pip install the missing package or blacklist this model family"),
    # XLA/JAX runtime crash — usually PJRT device init or compile failure.
    # Error code 13 = INTERNAL, code 8 = RESOURCE_EXHAUSTED, etc.
    ("xlaruntimeerror",
     "xla_runtime_error",
     "XLA runtime error",
     "bug: check XLA/PJRT init; may need tt-xla venv rebuild or chip reset"),
    ("jaxruntimeerror",
     "xla_runtime_error",
     "XLA runtime error",
     "bug: check XLA/PJRT init; may need tt-xla venv rebuild or chip reset"),
    # Forge missing op — model uses a PyTorch operator forge hasn't implemented
    ("are not implemented",
     "forge_missing_op",
     "Forge missing operator",
     "need: file a forge op request or filter this model_type"),
    ("not implemented",
     "forge_missing_op",
     "Forge missing operator",
     "need: file a forge op request or filter this model_type"),
    # Forge internal errors — file upstream bugs
    ("Internal Writer Error",
     "forge_internal",
     "Forge internal error",
     "bug: report to Tenstorrent with the model ID"),
    ("reconstruction error",
     "forge_internal",
     "Forge internal error",
     "bug: report to Tenstorrent with the model ID"),
    # ONNX loader bug in our code — onnx_tmp_path argument missing
    ("onnx_tmp_path",
     "loader_missing",
     "No loader available",
     "fix: repair onnx loader in build_dynamic_loader()"),
    # Tensor shape/dimension mismatches inside model forward pass — model-specific
    ("shapes cannot be multiplied",
     "shape_mismatch",
     "Tensor shape mismatch",
     "bug: model uses non-standard tensor layout; report with model ID"),
    ("incorrect shape",
     "shape_mismatch",
     "Tensor shape mismatch",
     "bug: model uses non-standard tensor layout; report with model ID"),
    ("size mismatch",
     "shape_mismatch",
     "Tensor shape mismatch",
     "bug: model uses non-standard tensor layout; report with model ID"),
    # JAX/Flax model called with PyTorch-style keyword arguments (pixel_values, etc.)
    # — framework API mismatch in our XLA worker argument construction
    ("unexpected keyword argument",
     "api_mismatch",
     "API keyword argument mismatch",
     "fix: check XLA worker forward() call for this model type"),
    # Custom model class missing expected method/attribute — model-specific bug
    ("has no attribute '_initialize_weights'",
     "model_bug",
     "Custom model class bug",
     "filter: model has broken custom code; exclude this model"),
    # Backend mismatch — paddlepaddle/paddle models not supported
    ("paddlepaddle",
     "unsupported_backend",
     "Unsupported backend",
     "filter: exclude paddle models from discovery"),
    ("requires `pixel_values` arguments",
     "unsupported_backend",
     "Unsupported backend",
     "filter: paddle forward() signature differs from PyTorch; skip paddle models"),
    # Model gone, private, or inaccessible on HuggingFace
    ("Can't load the model",
     "model_access",
     "Model access error",
     "filter: model private or deleted — add freshness check to discovery"),
    ("Repository Not Found",
     "model_access",
     "Model access error",
     "filter: model private or deleted — add freshness check to discovery"),
    # Worker-level timeouts from SIGALRM
    ("TIMEOUT",
     "timeout",
     "Timeout",
     "config: raise --timeout-s or lower --max-model-params"),
    # Out-of-memory
    ("out of memory",
     "oom",
     "Out of memory",
     "config: lower --max-model-params or add chip memory guard"),
    # IRD_LF_CACHE server required (Tenstorrent internal model cache)
    # These models pull weights from an internal S3 via a cache proxy that's
    # only reachable in the Tenstorrent CI environment.
    ("IRD_LF_CACHE",
     "missing_dependency",
     "IRD LF Cache server required",
     "filter: model requires internal Tenstorrent model cache server; exclude from external runs"),
    # Index out of range / out of bounds — usually a model-specific shape bug
    # that won't be fixed by retrying.
    ("out of bounds",
     "shape_mismatch",
     "Index out of bounds",
     "bug: model has a dimension/index mismatch; report with model ID"),
    ("index out of range",
     "shape_mismatch",
     "Index out of range",
     "bug: model has a dimension/index mismatch; report with model ID"),
    # Wrong backend: PyTorch tensor passed to the XLA/JAX worker.  Happens when
    # a model with library="pytorch" is incorrectly routed to XLA via arch-affinity
    # (e.g. "gpt2" model_type).  The router now guards against this, but historical
    # entries should be classified so the router's xla-fatal gate can fire.
    ("interpreting argument",
     "wrong_backend",
     "Wrong backend (PyTorch→XLA)",
     "fix: router now guards library=pytorch from XLA routing; re-run to confirm"),
    # Tensor stride conflict during an op — layout incompatibility between tensors
    ("stride mismatch",                "shape_mismatch",    "Stride mismatch",               "tensor stride conflict during op"),
    # Flax/JAX module used before init() was called — common in XLA worker setup
    ("ScopeCollectionNotFound",        "flax_scope_error",  "Flax scope not found",          "Flax module not initialized; call init() first"),
    # numpy/JAX ndarray expected but torch tensor passed — framework boundary leak
    ("concatenate requires ndarray",   "api_mismatch",      "JAX concat type error",         "numpy/jax ndarray expected, got torch tensor"),
    # Forge has no Resize2d implementation; needs operator decomposition
    ("Resize2d op",                    "forge_missing_op",  "Missing Resize2d op",           "forge does not support Resize2d; needs decomp"),
    # Model output tuple wider than unpacking target expects
    ("too many values to unpack",      "api_mismatch",      "Unpack arity mismatch",         "model output tuple wider than expected"),
    # Encoder-decoder model called without decoder_input_ids — None passed as tensor
    ("must be Tensor, not NoneType",   "api_mismatch",      "None passed as Tensor",         "encoder-decoder input missing decoder_input_ids"),
]

_CATEGORY_OTHER = ("other", "Other", "investigate manually")


def _classify_error(error: str) -> tuple[str, str, str]:
    """Map a raw error string to (category_key, display_label, action_hint).

    Applies ``_ERROR_RULES`` in order, returning the first match.
    Falls back to ``("other", "Other", "investigate manually")`` if nothing matches.
    """
    lower = error.lower()
    for substring, key, label, hint in _ERROR_RULES:
        if substring.lower() in lower:
            return key, label, hint
    return _CATEGORY_OTHER


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

    def is_compiled_by(self, model_id: str, backend: str) -> bool:
        """Return True if this model was successfully compiled by the given backend.

        Checks the ``backends_succeeded`` list first (populated since the backend
        tracking feature was added). Falls back to the ``backend`` field for entries
        written before ``backends_succeeded`` existed, defaulting to "forge" for
        legacy entries that predate the field entirely.

        Args:
            model_id: HuggingFace model identifier.
            backend:  Backend name to test, e.g. "forge" or "xla".
        """
        entry = self._data["compiled"].get(model_id)
        if not entry:
            return False
        return backend in entry.get("backends_succeeded", [entry.get("backend", "forge")])

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
        backend: str = "forge",
        first_voice: str = "",
        compile_s: float = 0.0,
        infer_s: float = 0.0,
        throughput: float = 0.0,
        throughput_unit: str = "",
        mesh_chips: int = 1,
    ) -> None:
        """Record a successful compilation.

        On first success, creates the compiled entry with all metadata. On
        subsequent calls, increments counters and updates best_time_s if the
        new run was faster. The artifact field is always overwritten with the
        most recent decoded inference output.

        Backend tracking: ``backend`` records which compilation backend produced
        this success. On first success, ``backend`` and ``backends_succeeded``
        (a list) are both initialised. On subsequent calls, the backend is appended
        to ``backends_succeeded`` if not already present; the original ``backend``
        field is preserved as the first-ever successful backend.

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
            artifact:      Raw decoded inference output (tensor stats or top-class prediction).
            backend:       Compilation backend used: "forge" (default) or "xla".
            first_voice:   Decoded first-voice text (real sample input → decoded output).
                           Empty string when first voice was not attempted or failed.
            compile_s:     Wall-clock seconds spent in forge.compile() / jax.jit(). Zero
                           means not measured; zero values are ignored for rolling-best.
            infer_s:       Wall-clock seconds for the post-compile inference call. Zero
                           means not measured; zero values are ignored for rolling-best.
            throughput:    Numeric throughput value. Interpretation depends on
                           throughput_unit: "tokens/sec" (higher is better) or
                           "ms/sample" (lower is better). Zero means not measured.
            throughput_unit: Unit string for throughput, e.g. "tokens/sec" or
                           "ms/sample". Empty string when throughput is not measured.
        """
        now = datetime.now(timezone.utc).isoformat()
        if model_id not in self._data["compiled"]:
            # First-time entry: capture all immutable metadata from this run,
            # including which backend first compiled this model successfully.
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
                "first_voice": first_voice,
                "backend": backend,
                "backends_succeeded": [backend],
                "mesh_chips": mesh_chips,
            }
        entry = self._data["compiled"][model_id]
        entry["successes"] += 1
        # Track the fastest compilation time across all chips/runs.
        if time_s < entry["best_time_s"]:
            entry["best_time_s"] = time_s
        # Update split timing and throughput rolling bests.
        # Zero values mean "not measured" and are skipped so legacy callers
        # that omit these params don't overwrite real data with zeros.
        if compile_s > 0.0:
            if "best_compile_s" not in entry or compile_s < entry["best_compile_s"]:
                entry["best_compile_s"] = compile_s
        if infer_s > 0.0:
            if "best_infer_s" not in entry or infer_s < entry["best_infer_s"]:
                entry["best_infer_s"] = infer_s
        if throughput > 0.0:
            # Always keep throughput_unit in sync with the latest non-zero measurement.
            entry["throughput_unit"] = throughput_unit
            if "best_throughput" not in entry:
                entry["best_throughput"] = throughput
            elif throughput_unit == "tokens/sec":
                # Higher throughput is better for token-rate metrics.
                if throughput > entry["best_throughput"]:
                    entry["best_throughput"] = throughput
            else:
                # Lower is better for latency metrics such as ms/sample.
                if throughput < entry["best_throughput"]:
                    entry["best_throughput"] = throughput
        # Track the highest chip count ever used to compile this model.
        if mesh_chips > entry.get("mesh_chips", 1):
            entry["mesh_chips"] = mesh_chips
        # Always update artifact so the bestiary reflects the most recent output.
        entry["artifact"] = artifact
        # Update first_voice if we got a non-empty result (never clobber with "").
        if first_voice:
            entry["first_voice"] = first_voice
        elif "first_voice" not in entry:
            entry["first_voice"] = ""
        # Accumulate the set of backends that have successfully compiled this model.
        # setdefault handles legacy entries written before this field existed,
        # seeding the list from the stored `backend` field (defaulting to "forge").
        entry.setdefault("backends_succeeded", [entry.get("backend", "forge")])
        if backend not in entry["backends_succeeded"]:
            entry["backends_succeeded"].append(backend)

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
                "error_category": "",
            }
        entry = self._data["failed"][model_id]
        entry["attempts"] += 1
        entry["last_error"] = error
        # Always recompute so the category stays in sync if the error changes.
        entry["error_category"] = _classify_error(error)[0]

    def failure_stats(self) -> list[dict]:
        """Aggregate all-time failures by error category, sorted by count desc.

        Handles entries that predate the error_category field by classifying
        them on the fly from their stored last_error string.

        Returns a list of dicts, each with:
          key:    category key (e.g. "tracer_output_type")
          label:  human-readable name (e.g. "Tracer output type")
          hint:   short action suggestion
          count:  number of distinct models in this category
          models: list of model_ids, sorted by attempt count desc
        """
        # Build a map from category_key → list of (model_id, attempts) pairs.
        buckets: dict[str, list[tuple[str, int]]] = {}
        for model_id, entry in self._data["failed"].items():
            # Prefer the stored category; classify retroactively if missing.
            cat = entry.get("error_category") or ""
            if not cat:
                cat = _classify_error(entry.get("last_error", ""))[0]
            buckets.setdefault(cat, []).append(
                (model_id, entry.get("attempts", 1))
            )

        # Build label/hint lookup from _ERROR_RULES + the "other" fallback.
        _meta: dict[str, tuple[str, str]] = {
            key: (label, hint) for _, key, label, hint in _ERROR_RULES
        }
        _meta["other"] = (_CATEGORY_OTHER[1], _CATEGORY_OTHER[2])

        result: list[dict] = []
        for key, pairs in buckets.items():
            label, hint = _meta.get(key, (key, "investigate manually"))
            # Sort models within category by attempt count (most attempts first).
            pairs.sort(key=lambda x: -x[1])
            result.append({
                "key":    key,
                "label":  label,
                "hint":   hint,
                "count":  len(pairs),
                "models": [m for m, _ in pairs],
            })
        # Sort categories by model count descending.
        result.sort(key=lambda x: -x["count"])
        return result

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

    def append_perf_record(self, record: dict) -> None:
        """Append one performance record line to the sibling perf_history.jsonl file.

        The file is stored alongside bestiary.json as data/perf_history.jsonl.
        Each line is a self-contained JSON object — one per model per run.
        The parent directory is created if it does not already exist, so this
        method is safe to call before save() has ever been invoked.

        Args:
            record: Arbitrary dict of performance fields for one run. Callers
                    typically include at minimum: model_id, run, compile_s,
                    infer_s, throughput, throughput_unit.
        """
        perf_path = self.path.parent / "perf_history.jsonl"
        perf_path.parent.mkdir(parents=True, exist_ok=True)
        with perf_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

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

        Also re-runs classification on any failed entry that is still categorised
        as "other" — this upgrades stale entries whenever a new _ERROR_RULES entry
        is added without needing a separate migration script.
        """
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for key in ("compiled", "failed", "chip_totals"):
                    data.setdefault(key, {})
                # Re-classify stale "other" entries so new rules take effect.
                reclassified = 0
                for entry in data["failed"].values():
                    if entry.get("error_category", "other") == "other":
                        err = entry.get("last_error", "")
                        new_cat = _classify_error(err)[0]
                        if new_cat != "other":
                            entry["error_category"] = new_cat
                            reclassified += 1
                # Persist re-classifications immediately so JSON stays in sync
                # with _ERROR_RULES. Write-back is best-effort; failures are silent.
                if reclassified > 0:
                    try:
                        self.path.write_text(
                            json.dumps(data, indent=2, default=str),
                            encoding="utf-8",
                        )
                    except OSError:
                        pass
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"compiled": {}, "failed": {}, "chip_totals": {}}
