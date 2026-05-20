# Ephemeral Cache Cleanup Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `--ephemeral` flag to the expedition that automatically evicts freshly-downloaded model weights after compilation and benchmarking, preserving disk space across long runs while keeping "gold star" models for later inspection.

**Architecture:** A new `cache_janitor` module handles all HF cache introspection and deletion logic. The two worker processes (`expedition_worker.py` and `expedition_worker_xla.py`) call into it after each model result. The orchestrator (`expedition.py`) surfaces the flags to the CLI.

**Tech Stack:** `huggingface_hub` cache scanning API (`scan_cache_dir`, `CachedRepoInfo`), `shutil.rmtree` for deletion, existing `ScoreResult` and `Rarity` types from `scorer.py`.

---

## Flags

Two new CLI flags, both on the expedition launcher (`expedition.py`) and passed through to workers:

| Flag | Default | Meaning |
|---|---|---|
| `--ephemeral` | off | Enable cleanup mode |
| `--evict-failures` | off | Requires `--ephemeral`; also evict weights for failed models |

`--evict-failures` alone (without `--ephemeral`) is a no-op and should log a warning.

---

## Gold Star Criteria

A model result is **gold star** if either condition holds:
- `result.rarity` is `"rare"` or `"legendary"` (as defined in `scorer.py`'s `Rarity` enum)
- `result.is_first_ever` is `True`

Gold star models always have their weights preserved, regardless of flags.

---

## Eviction Decision Table

Evaluated after each model result, only when `--ephemeral` is active:

| Result | Net-new download? | `--evict-failures`? | Action |
|---|---|---|---|
| Success, gold star | yes | — | Preserve; print `★ SAVED` |
| Success, non-gold-star | yes | — | Evict; print `♻ evicted (Xmb freed)` |
| Failure | yes | no | Keep; no output |
| Failure | yes | yes | Evict; print `♻ evicted (Xmb freed)` |
| Any | no (pre-existing) | — | Skip silently |
| Any (static/local loader) | — (no HF entry) | — | Skip silently |

"Pre-existing" means the model's HF cache repo was present in the snapshot taken at worker startup, before any downloads occurred.

---

## New Module: `lib/expedition/cache_janitor.py`

Three public functions, no class required:

### `snapshot_preexisting() -> frozenset[str]`

Scans `~/.cache/huggingface/hub/` via `huggingface_hub.scan_cache_dir()` and returns the set of `repo_id` strings (e.g. `"openai-community/gpt2"`) already present. Called once at worker startup before any model loads.

Returns empty frozenset if the cache dir does not exist or `scan_cache_dir()` raises.

### `is_gold_star(model_id: str, result) -> bool`

Returns `True` if the result is a **successful** compilation (`result.pts > 0`) AND either `result.rarity in ("rare", "legendary")` or `result.is_first_ever`. Failures are never gold star — a first-ever failure has `is_first_ever=True` in the scorer but should not preserve weights. The `result` parameter is the `ScoreResult` dataclass from `scorer.py`. `Rarity` is `str`-enum so direct string comparison works.

### `maybe_evict(model_id: str, result, preexisting: frozenset[str], evict_failures: bool = False) -> tuple[bool, int]`

Orchestrates the eviction decision. Returns `(evicted: bool, bytes_freed: int)`.

Logic:
1. If not in `preexisting`, check HF cache for this `model_id`. If the repo exists in cache AND was not in `preexisting`, it is net-new. If no HF entry at all, return `(False, 0)`.
2. If result is a success: evict unless gold star.
3. If result is a failure: evict only if `evict_failures=True`.
4. Eviction: find all blob files for the repo via `scan_cache_dir()`, sum their sizes, then call `shutil.rmtree()` on the repo's cache directory (`~/.cache/huggingface/hub/models--{org}--{name}/`). Return `(True, bytes_freed)`.

The `model_id` → cache directory mapping: replace `/` with `--` and prefix with `models--`. E.g. `openai-community/gpt2` → `models--openai-community--gpt2`.

---

## Integration: `expedition_worker.py`

1. **New CLI args:** `--ephemeral` (store_true), `--evict-failures` (store_true). Passed through from `expedition.py` subprocess launch.
2. **At startup:** if `--ephemeral`, call `preexisting = cache_janitor.snapshot_preexisting()` and store it.
3. **After each model result** (after bestiary write, before next model load): call `maybe_evict(model_id, result, preexisting, evict_failures)`.
4. **TUI output:** append to the existing result line:
   - Gold star preserve: `  [yellow]★ SAVED[/yellow]` (after the pts display)
   - Evicted: `  [dim]♻ {human_bytes(freed)} freed[/dim]`

---

## Integration: `expedition_worker_xla.py`

Same pattern as above — identical CLI args, same `snapshot_preexisting()` at startup, same `maybe_evict()` call after each result.

---

## Integration: `expedition.py`

1. Add `--ephemeral` and `--evict-failures` to the main expedition argument parser.
2. Pass both flags through in the subprocess args when launching worker processes.
3. If `--evict-failures` is set without `--ephemeral`, print a warning and ignore `--evict-failures`.

---

## Error Handling

- If `scan_cache_dir()` raises at snapshot time: log a warning, set `preexisting = frozenset()`, continue the run (conservative — treats everything as pre-existing, so nothing gets evicted).
- If `shutil.rmtree()` raises during eviction: log the error, count as `(False, 0)`, continue the run. Never abort the expedition over a cleanup failure.
- If the HF cache dir for a model can't be found despite the model being net-new: skip silently (the model may use a custom cache path or in-memory weights).

---

## What Is NOT Evicted

- Tokenizer-only downloads (no model blobs) — the whole repo dir is targeted, so tokenizers inside a model repo are evicted with the weights. Standalone tokenizer repos (rare) are outside scope.
- Forge compiled artifacts (`.ttnn`, flatbuffers) — these live in `/dev/shm` or temp dirs and are already cleaned up by the existing shm-clearing logic.
- Non-HuggingFace caches (e.g. `~/.cache/tt-forge-compiletron-flax/`) — out of scope.

---

## Testing

- Unit tests in `tests/test_cache_janitor.py` using a tmp dir as a fake HF cache.
- Test `snapshot_preexisting()` with a populated fake cache dir.
- Test `is_gold_star()` for all four cases (rare success, legendary success, first-ever, common success).
- Test `maybe_evict()` for each row of the decision table, mocking `shutil.rmtree`.
- Integration: existing `tests/test_expedition_worker.py` (or equivalent) — confirm new flags are accepted without error.
