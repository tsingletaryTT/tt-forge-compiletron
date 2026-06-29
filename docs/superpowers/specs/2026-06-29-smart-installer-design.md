# Smart Installer Design — `scripts/install.sh`

**Date:** 2026-06-29  
**Status:** Approved  
**Scope:** `scripts/install.sh` — smart front door wrapping the existing `scripts/setup-venvs.sh`

---

## Problem

`scripts/setup-venvs.sh` is a reliable install engine but a poor diagnostic tool:

- `set -euo pipefail` means it stops at the first failure — you never see the full picture
- No check-only mode — running it always installs/upgrades
- No version comparison — can't tell whether an existing install is up-to-date
- Known gotchas (wrong `forge` package, stale shm segments, missing mesh descriptor, triton conflicts) surface as confusing errors rather than named, actionable failures
- No summary — you scroll through dense output to determine what happened

## Approach

**Thin wrapper (A):** `install.sh` is the smart front door — it checks, compares versions, detects gotchas, and prints a structured summary. The actual pip work is delegated to `setup-venvs.sh` unchanged. Two files, each focused. Existing callers of `setup-venvs.sh` (e.g. `record_demo.sh`) are unaffected.

---

## Architecture

```
scripts/
  install.sh        ← new; smart check + summary wrapper
  setup-venvs.sh    ← unchanged; install engine
```

`install.sh` never calls `pip` directly — it calls `setup-venvs.sh --forge` or `setup-venvs.sh --xla` when a step needs fixing.

---

## Steps

Each step has three behaviours:

| Mode | Behaviour |
|---|---|
| `--status` | Check only. Print pass/warn/fail. No installs, no sudo. |
| default | Check first. If failing and fixable, fix it. Re-check. |

```
[1]  Hardware presence    tt-smi detects ≥1 TT chip; print board type + fw bundle version
[2]  Hugepages            /proc/sys/vm/nr_hugepages ≥ 64
                          fix: sudo sysctl -w vm.nr_hugepages=128
[3]  Disk space           warn if < 50 GB free; info on what's eating space
[4]  Forge venv           ~/tt-forge-venv exists; correct TT forge package importable
                          gotcha: PyPI's `forge` package is a Django app — must verify
                          `forge.__version__` contains a Tenstorrent semver, not Django cruft
                          fix: setup-venvs.sh --forge
[5]  Forge version        parse TT PyPI simple index for latest non-dev X.Y.Z release
                          compare against installed; warn if behind
[6]  Forge-fe shim        ~/tt-forge-fe/env/activate exists; sources without error
                          fix hint: re-run setup-venvs.sh --forge (shim written automatically)
[7]  XLA venv             ~/tt-xla/venv exists; pjrt-plugin-tt importable
                          fix: setup-venvs.sh --xla
[8]  XLA version          parse TT PyPI simple index for latest pjrt-plugin-tt X.Y.Z
                          compare against installed; warn if behind; print upgrade command
[9]  Mesh descriptor      p100_mesh_graph_descriptor.textproto at expected path under ~/tt-xla
                          fix: symlink bundled descriptor (same logic as setup-venvs.sh)
[10] tt-forge-models      ~/code/tt-forge-models present
                          warn-only (frontier-only mode works without it)
[11] Stale /dev/shm       sm_segment.tt-quietbox.*.0 files — leftover from crashed runs
                          warn + print fix command; never auto-delete (user decides)
```

---

## Version Detection

Both forge and pjrt-plugin-tt publish to `https://pypi.eng.aws.tenstorrent.com/`.  
The simple index (`/simple/<package>/`) lists all filenames; we parse for the latest non-dev `X.Y.Z`:

```bash
_latest_tt_version() {
    local pkg="$1"
    python3 - "$pkg" <<'EOF'
import sys, urllib.request, re
pkg = sys.argv[1]
url = f"https://pypi.eng.aws.tenstorrent.com/simple/{pkg}/"
try:
    html = urllib.request.urlopen(url, timeout=5).read().decode()
    versions = re.findall(rf'{re.escape(pkg)}-(\d+\.\d+\.\d+)-', html, re.IGNORECASE)
    stable = sorted(set(versions), key=lambda v: list(map(int, v.split('.'))))
    print(stable[-1] if stable else "")
except Exception:
    print("")
EOF
}
```

Falls back to empty string if the index is unreachable — steps that use it degrade to "cannot check, skipping version comparison" rather than failing.

---

## Flags

| Flag | Behaviour |
|---|---|
| `--status` | Check-only; no installs, no sudo. Exit 0 if all pass/warn, 1 if any fail. |
| `--forge` | Scope to forge steps only (1–6). Mirrors setup-venvs.sh --forge. |
| `--xla` | Scope to XLA steps only (1–3, 7–9). Mirrors setup-venvs.sh --xla. |
| `--skip-hardware` | Skip step 1 (hardware detection). For CI or offsite setup. |
| `--help` | Print usage. |

Flags compose: `--status --forge` checks forge steps only without installing.

---

## Output

Each step prints a single line:
```
  ✓  Forge venv          ~/tt-forge-venv  (forge 1.3.0)
  ⚠  Forge version       installed 1.2.0 — latest 1.3.0  (run: scripts/install.sh --forge)
  ✗  Mesh descriptor     not found at ~/tt-xla/third_party/...
```

Summary table at the end:
```
╔══════════════════════════════════════
║  Summary
╠══════════════════════════════════════
  ✓  Hardware            4x Blackhole  fw 19.11.0
  ✓  Hugepages           96
  ⚠  Disk space          49 GB free — getting low
  ✓  Forge venv          1.3.0
  ⚠  Forge version       behind (1.2.0 installed, 1.3.0 available)
  ...
╚══════════════════════════════════════
  1 warning(s). Run: scripts/install.sh to fix.
```

No right-side border characters (per project convention).

---

## Error Handling

- Steps do not use `set -e` — all steps run regardless of prior failures
- Each step records its outcome in `PASSED` / `WARNED` / `FAILED` arrays
- Exit code: 0 if no FAILs, 1 if any FAIL (warnings don't affect exit code)
- Log file: `/tmp/tt-compiletron-install.log` — verbose output from setup-venvs.sh calls
- Version comparison failures (network unreachable) → `WARNED`, never `FAILED`

---

## Gotcha Registry

Named gotchas captured as named checks rather than surfacing as confusing pip errors:

| Gotcha | Detection | Fix hint |
|---|---|---|
| Django `forge` installed | `forge.__version__` looks like semver but package is wrong | `pip install forge --extra-index-url TT_PYPI` |
| `pjrt-plugin-tt` older dev build installed | version contains `+dev` | explicit `pip install pjrt-plugin-tt==X.Y.Z` |
| `triton`/`easydel` dep conflict | non-fatal; only warn if present in XLA venv | "harmless — XLA compile still works" |
| Stale `/dev/shm` segments | glob `sm_segment.tt-quietbox.*.0` | print delete command; don't auto-delete |
| Missing mesh descriptor | file check at known path | symlink from bundled descriptor |
| Forge-fe shim missing | file check | re-run setup-venvs.sh --forge |

---

## Files Changed

| File | Change |
|---|---|
| `scripts/install.sh` | New — ~350 lines |
| `scripts/setup-venvs.sh` | Unchanged |
| `INSTALL.md` | Add one-liner pointing to `install.sh` at the top |
