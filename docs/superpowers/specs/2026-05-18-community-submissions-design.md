# Community Bench Submissions Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let players at home submit verified bench results from their own Tenstorrent hardware, building a cross-hardware performance comparison table on the website.

**Architecture:** GitHub Issue form → GitHub Action validates + opens PR → maintainer reviews and merges → website dynamically fetches and renders community data grouped by use case → model → hardware.

**Trust model:** Maintainer approval required on every PR. No auto-merge.

**Data bar:** Full 5-pass bench run (`--bench-passes 5`). Submitters paste or attach their `perf_history.jsonl` output as JSON.

---

## 1. Community Data Schema

Each accepted submission lands as a `.jsonl` file in `data/community/` named:

```
<submitter>-<hardware_system>-<chips_used>chip-<YYYY-MM-DD>.jsonl
```

Each line is one model run — the existing `perf_history.jsonl` format extended with five new fields:

```jsonl
{"model_id": "alexnet/pytorch", "backend": "forge", "compile_s": 3.12,
 "bench_passes": 5, "infer_p50_s": 0.041, "throughput_p50": 41.2,
 "throughput_unit": "ms/sample", "timestamp": "2026-05-18T10:35:03Z",
 "hardware_system": "N300", "chips_used": 2, "chips_in_system": 2,
 "firmware_version": "80.14.0.0", "backend_version": "0.1.0",
 "tt_kmd_version": "1.29",
 "submitter": "gh-username", "submission_issue": 142}
```

### Required fields (from existing perf_history.jsonl)

| Field | Type | Notes |
|---|---|---|
| `model_id` | string | e.g. `"alexnet/pytorch"` |
| `backend` | string | `"forge"`, `"xla"`, or `"onnx"` |
| `compile_s` | float | wall-clock compile time in seconds |
| `bench_passes` | int | must be ≥ 5 |
| `infer_p50_s` | float | median inference time in seconds |
| `throughput_p50` | float | median throughput value |
| `throughput_unit` | string | `"tokens/sec"` or `"ms/sample"` |
| `timestamp` | string | ISO 8601 |

### New required fields

| Field | Type | Notes |
|---|---|---|
| `hardware_system` | enum | `N150`, `N300`, `QB`, `QB2`, `LoudBox`, `custom` |
| `chips_used` | int | chips active for this run (1–32) |
| `chips_in_system` | int | total chips in the box/cluster; must be ≥ chips_used |
| `firmware_version` | string | from `tt-smi -s` firmware field |
| `backend_version` | string | `pip show tt-forge\|pjrt-plugin-tt\|tt-forge-onnx \| grep Version` |

### Injected by the Action (not submitted by user)

| Field | Type | Notes |
|---|---|---|
| `submitter` | string | GitHub username from event context |
| `submission_issue` | int | issue number — traceability |

### Optional fields

| Field | Type | Notes |
|---|---|---|
| `tt_kmd_version` | string | `modinfo tenstorrent \| grep ^version` |
| `infer_p95_s` | float | p95 inference time |
| `throughput_p95` | float | p95 throughput |

### Validation rules

- `bench_passes ≥ 5`
- `chips_used ≤ chips_in_system`
- `hardware_system` in `{N150, N300, QB, QB2, LoudBox, custom}`
- `throughput_unit` in `{tokens/sec, ms/sample}`
- `firmware_version` and `backend_version` non-empty strings
- All required fields present and correctly typed

---

## 2. Issue Form

**File:** `.github/ISSUE_TEMPLATE/bench-submission.yml`

Label `bench-submission` auto-applied so the Action knows what to process.

**Fields:**

```
Title:  [Bench] <model> · <system> · <N>-chip

Tenstorrent system          (dropdown)
  N150 | N300 | QB | QB2 | LoudBox | custom

Chips used for this run     (dropdown)
  1 | 2 | 4 | 8 | 16 | 32

Total chips in system       (dropdown)
  1 | 2 | 4 | 8 | 16 | 32

Firmware version            (text, required)
  Hint: tt-smi -s | grep firmware

Backend version             (text, required)
  forge  → pip show tt-forge | grep Version
  xla    → pip show pjrt-plugin-tt | grep Version
  onnx   → pip show tt-forge-onnx | grep Version

tt-kmd version              (text, optional)
  Hint: modinfo tenstorrent | grep ^version

Bench JSON                  (textarea, required)
  Paste perf_history.jsonl lines from your --bench-passes 5 run,
  or attach the file (rename to .txt if GitHub rejects .jsonl).
  One JSON object per line.

Notes                       (textarea, optional)
  Anything unusual: cooling, ambient temp, other workloads running, etc.
```

`submitter` and `submission_issue` are never shown in the form — the Action injects them from GitHub event context.

---

## 3. GitHub Action Pipeline

**Files:**
- `.github/workflows/community-submission.yml` — Action definition
- `scripts/validate_submission.py` — validation logic (also runnable locally)

**Trigger:** `issues: [opened, edited]` where issue has label `bench-submission`.

**Flow:**

```
1. Parse issue body
   Extract hardware fields (system, chips_used, chips_in_system,
   firmware_version, backend_version, tt_kmd_version) from form inputs.
   Extract bench JSON lines from textarea or attached file.

2. Validate  (scripts/validate_submission.py)
   Run all validation rules.
   → On failure: comment on issue with specific per-field errors,
     apply label "submission-invalid", exit. No PR created.

3. Enrich
   Inject submitter (from github.event.issue.user.login),
   submission_issue (from github.event.issue.number),
   and version fields into each JSON line.

4. Create PR
   Branch:    community/<submitter>-<issue-number>
   File:      data/community/<submitter>-<system>-<chips>chip-<date>.jsonl
   PR title:  [Community] <submitter> · <system> · <chips>-chip · N models
   PR body:   table of submitted models (model_id, compile_s, throughput_p50)
              + link back to originating issue

5. Comment on issue
   "✓ Validated — PR #<n> opened for maintainer review."
   Apply label "submission-pending".

6. On PR merge
   Apply label "submission-accepted" to original issue.
```

**Authentication:** Uses `GITHUB_TOKEN` only — no extra secrets required for PR creation against the same repo.

**Local validation:**
```bash
python3 scripts/validate_submission.py my_results.jsonl \
  --system QB2 --chips-used 4 --chips-in-system 4 \
  --firmware-version "80.14.0.0" --backend-version "0.1.0"
```

---

## 4. Website Community Section

**File:** `docs/index.html` — new `#community` section below `#perf`.

### Data fetching

Pure JS, no build step. On page load:

1. `GET https://api.github.com/repos/<owner>/<repo>/contents/data/community/` — lists community files
2. For each `.jsonl` file, fetch raw content and parse line-by-line
3. Merge with maintainer baseline from `data/perf_history.jsonl` (fetched same way, maintainer rows labeled "maintainer · QB2 · 4-chip")
4. Falls back silently if API rate-limited (60 req/hr unauthenticated) or directory is empty

### Rendering hierarchy

**Use case → Model → Hardware**

```
Community Benchmarks — Submitted by Players at Home

[N submissions · M hardware platforms · K unique models]  ← summary callout

── CV: Image Classification ──────────────────────────────────────────

  alexnet/pytorch
    QB2    4-chip   2.98s compile   169ms p50   169 ms/smp   maintainer
    N300   2-chip   3.12s compile    41ms p50    41 ms/smp   @someone ↗
    N150   1-chip   3.44s compile    78ms p50    78 ms/smp   @other   ↗

  mobilenetv2/pytorch
    QB2    4-chip   10.45s compile  484ms p50   484 ms/smp   maintainer

── Text Generation ────────────────────────────────────────────────────

  bloom/causal_lm/jax
    QB2    4-chip    3.28s compile   33ms p50  3,856 tok/s   maintainer
    N300   2-chip    4.10s compile   61ms p50  2,100 tok/s   @someone ↗
```

**Sorting:** use cases by name; models within use case by submission count descending; hardware rows within model by throughput (best tok/s or lowest ms/sample first).

**Models with maintainer-only data** are still shown — implicit call-to-action ("be the first to benchmark this on N150").

**Per-row details:**
- `firmware_version` shown as a hover tooltip (keeps table scannable)
- `↗` in the "By" column links to the originating GitHub issue
- `backend_version` shown in tooltip alongside firmware

### Summary callout

At the top of the section, a single line updated dynamically:

```
7 submissions · 3 hardware platforms (QB2, N300, N150) · 12 unique models
```

---

## 5. File Layout

```
.github/
  ISSUE_TEMPLATE/
    bench-submission.yml        ← structured issue form
  workflows/
    community-submission.yml    ← intake + validation + PR creation
scripts/
  validate_submission.py        ← schema validator (used by Action + locally)
data/
  community/                    ← merged submission files (one per submitter run)
    <submitter>-<system>-<chips>chip-<date>.jsonl
docs/
  index.html                    ← #community section added
```

---

## 6. Out of Scope

- Auto-merge (maintainer approval required on every PR)
- Site regeneration on merge (JS fetches live; no build step)
- Leaderboard gamification (points, streaks) for community submissions
- Authenticated API calls (public repo, unauthenticated GitHub API is sufficient)
- Submissions without bench data (compile-only results not accepted)
