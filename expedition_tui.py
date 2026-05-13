#!/usr/bin/env python3
# expedition_tui.py
"""
Textual TUI front-end for Expedition Mode.

Three screens:

  SetupScreen  — interactive config + live discovery/download log.
                 User adjusts parameters with keyboard, presses Enter
                 to start HF discovery and pre-download, then auto-advances.

  RunScreen    — equal-sized chip panels with streaming worker output,
                 expedition log, hardware sidebar, score strip.

  SummaryScreen — colorful end-of-run results with full error text
                  (no truncation), medals, points, bestiary count.

Invoked via: python3 expedition.py run --tui [options]
"""
from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Header, RichLog, Static

PROJECT_DIR = Path(__file__).parent

from lib.expedition.run_state import ModelResult, RunState

# ── Roguelike identity tables ────────────────────────────────────────────────

_ADVENTURER_TITLES = [
    "The Forge Mage",
    "The Iron Scribe",
    "The Tensor Sage",
    "The Bit Wanderer",
]

_FAILURE_FLAVOR = [
    "fell to a shape mismatch",
    "consumed by the void",
    "lost in the forward pass",
    "struck down by an unsupported op",
    "scattered across a broken graph",
    "dissolved in a dtype conflict",
    "claimed by the runtime",
    "swallowed by the compiler's maw",
]

_RARITY_MARKUP = {
    "legendary": "[bold magenta]★★★ LEGENDARY[/]",
    "rare":      "[bold bright_magenta]★ RARE[/]",
    "uncommon":  "[bold yellow]◆ UNCOMMON[/]",
    "common":    "[cyan]·[/]",
    "familiar":  "[dim]·[/]",
}

# ── Regex patterns for parsing worker stdout ─────────────────────────────────

_RE_SUCCESS = re.compile(r"✓ SUCCESS")
_RE_FAILURE = re.compile(r"✗ FAILED")
_RE_PTS     = re.compile(r"pts:\s*([+-]?\d+)")
_RE_STREAK  = re.compile(r"🔥×(\d+)")
_RE_FIRST   = re.compile(r"FIRST EVER")
_RE_RARITY  = re.compile(
    r"(LEGENDARY|RARE FIND|UNCOMMON|common|familiar)", re.IGNORECASE)
_RE_OSC     = re.compile(r"\x1b\][^\x1b\x07]*(?:\x1b\\|\x07)")
_RE_ANSI    = re.compile(r"\x1b\[[^m]*m")
# TT Metal C++ loguru writes "Always |" dispatch/init noise to fd1 (stdout),
# bypassing FilteredStderr.  Suppress it here so it never reaches the panel log.
_RE_TT_NOISE = re.compile(r"Always \s*\|")

# Keys present in the model dict that are internal dispatch metadata and must
# not be forwarded to the worker subprocess.  Defined at module level so the
# frozenset is created once rather than on every _launch_model call.
_WORKER_SKIP_KEYS = frozenset({"chips_needed", "decision"})

# Backend suffixes that appear as the last path segment in seed-model IDs like
# "gpt2/pytorch" or "bloom/causal_lm/jax".  Strip them to get a clean name.
_BACKEND_SUFFIXES = frozenset({
    "pytorch", "jax", "onnx", "tensorflow", "flax", "paddle", "paddlepaddle",
})


def _model_short(model_id: str, max_len: int = 22) -> str:
    """Return a display-friendly short name for a model ID.

    Seed-model IDs follow the pattern "<name>/<task>/<backend>" or "<name>/<backend>".
    If the last path segment is a known backend tag, the real model name is the
    *first* segment (e.g. "bloom/causal_lm/jax" → "bloom").  HuggingFace IDs like
    "openai-community/gpt2" have non-backend last segments, so we use that last
    segment (the repo name after the org prefix).
    """
    parts = model_id.split("/")
    if parts[-1].lower() in _BACKEND_SUFFIXES:
        name = parts[0]
    else:
        name = parts[-1]
    return name[:max_len]


def _strip_osc(line: str) -> str:
    return _RE_OSC.sub("", line)


# ── Status file I/O ───────────────────────────────────────────────────────────

def _read_status(chip_id: int, status_dir: str | None = None) -> dict[str, str]:
    status_dir = status_dir or os.environ.get("EXPEDITION_STATUS_DIR", "/tmp")
    path = Path(status_dir) / f"expedition_chip_{chip_id}.status"
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return out


def _absorb_csv_row(run_state: RunState, chip_id: int,
                    results_path: str, is_sq: bool) -> None:
    """Read the last CSV row written by a completed worker and store in RunState.

    Called after proc.wait() — by then the worker has written its CSV row and
    exited.  Silently does nothing on any I/O or parse error.

    Args:
        run_state:    The RunState for this expedition run.
        chip_id:      The chip whose worker just exited.
        results_path: Path to the per-chip CSV file the worker wrote.
        is_sq:        True if this was a side quest model.
    """
    try:
        rows = list(csv.DictReader(open(results_path, newline="")))
        if not rows:
            return
        row    = rows[-1]
        c      = run_state.chip(chip_id)
        result = ModelResult.from_csv_row(
            row, chip_id, rarity=c.rarity, streak=c.streak, is_sq=is_sq
        )
        run_state.add_result(result)
    except Exception:
        pass


def _render_score_row(chip_id: int) -> Text:
    s = _read_status(chip_id)
    if not s:
        return Text.from_markup(
            f"[bold yellow]C{chip_id}[/]  [dim]{'░' * 24}[/]  waiting..."
        )
    current   = int(s.get("current",   0))
    total     = int(s.get("total",     0)) or 1
    pts       = int(s.get("pts",       0))
    streak    = int(s.get("streak",    0))
    successes = int(s.get("successes", 0))
    failures  = int(s.get("failures",  0))
    model     = _model_short(s.get("model", ""))
    done      = s.get("done", "0") == "1"

    # Use completed (successes + failures) as numerator so the bar doesn't jump
    # to 100% when the last model starts loading — only when it finishes.
    completed = successes + failures
    ratio  = min(completed / total, 1.0)
    pct    = int(ratio * 100)
    filled = round(ratio * 24)
    bar    = "█" * filled + "░" * (24 - filled)
    color  = "green" if done else "cyan"
    streak_txt = f"  🔥×{streak}" if streak >= 2 else ""

    if done:
        return Text.from_markup(
            f"[bold yellow]C{chip_id}[/] [{color}]{bar}[/] {pct:3d}%"
            f"  [green]✓{successes}[/]/[red]✗{failures}[/]"
            f"  [gold1]{pts:>5}pts[/]"
            f"  [bold green]✓ DONE[/]"
        )
    return Text.from_markup(
        f"[bold yellow]C{chip_id}[/] [{color}]{bar}[/] {pct:3d}%"
        f"  [green]✓{successes}[/]/[red]✗{failures}[/]"
        f"  [gold1]{pts:>5}pts[/]{streak_txt}"
        f"  [cyan]{model}[/]"
    )


# ── Shared widgets ────────────────────────────────────────────────────────────

class ChipPanel(Widget):
    DEFAULT_CSS = """
    ChipPanel {
        height: 1fr;
        width: 1fr;
        border: solid $primary;
    }
    ChipPanel.done   { border: solid $success; }
    ChipPanel.failed { border: solid $error; }
    ChipPanel > RichLog { height: 1fr; width: 1fr; scrollbar-size: 0 0; }
    """

    def __init__(self, chip_id: int, title: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.chip_id = chip_id
        self._title  = title
        self.border_title = title

    def compose(self) -> ComposeResult:
        yield RichLog(markup=False, highlight=False, auto_scroll=True)

    def write_line(self, raw: str) -> None:
        line = _strip_osc(raw).rstrip("\n")
        if not line:
            return
        if _RE_TT_NOISE.search(line):
            return
        log = self.query_one(RichLog)
        try:
            log.write(Text.from_ansi(line))
        except Exception:
            log.write(line)

    def mark_done(self, success: bool) -> None:
        self.add_class("done" if success else "failed")
        label = "✓ DONE" if success else "✗ FAILED"
        self.border_title = f"{self._title}  {label}"


class HardwareWidget(Static):
    DEFAULT_CSS = """
    HardwareWidget {
        height: 12;
        border-left: solid $accent;
        border-top: solid $accent;
        border-bottom: solid $accent;
        padding-right: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self.border_title = "  HARDWARE  "

    def on_mount(self) -> None:
        self._refresh_hw()
        self.set_interval(2.0, self._refresh_hw)

    def _refresh_hw(self) -> None:
        try:
            result = subprocess.run(
                ["tt-smi", "-s"],
                capture_output=True, text=True, timeout=5,
            )
            data    = json.loads(result.stdout)
            devices = data.get("device_info", [])
            lines: list[str] = []
            for i, dev in enumerate(devices[:4]):
                # tt-smi -s nests board fields under "board_info" and clock/temp under "telemetry"
                binfo  = dev.get("board_info", {})
                btype  = binfo.get("board_type", "?")[:10]
                telem  = dev.get("telemetry", {})
                temp   = telem.get("asic_temperature", "?")
                aiclk  = telem.get("aiclk",            "?")
                lines.append(
                    f"[bold cyan]Chip {i}[/] {btype}"
                    f"  [yellow]{aiclk}MHz[/]  [red]{temp}°C[/]"
                )
            self.update(Text.from_markup(
                "\n".join(lines) if lines else "[dim]No devices[/]"
            ))
        except FileNotFoundError:
            self.update(Text.from_markup("[dim]tt-smi not found[/]"))
        except Exception as exc:
            self.update(Text.from_markup(f"[dim]tt-smi: {str(exc)[:40]}[/]"))


class EventLog(RichLog):
    DEFAULT_CSS = """
    EventLog {
        height: 1fr;
        border-left: solid $secondary;
        border-top: solid $secondary;
        border-bottom: solid $secondary;
        padding-right: 1;
        scrollbar-size: 0 0;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(markup=True, highlight=False, auto_scroll=True, **kwargs)
        self.border_title = "  EXPEDITION LOG  "

    def log_success(self, chip_id: int, model: str, rarity: str,
                    pts: int, first_ever: bool, streak: int) -> None:
        badge  = _RARITY_MARKUP.get(rarity, _RARITY_MARKUP["common"])
        short  = _model_short(model)
        first  = "  [bold gold1]★ BESTIARY[/]" if first_ever else ""
        streak_txt = f"  🔥×{streak}" if streak >= 2 else ""
        self.write(
            f"[bold green]⚔[/] [yellow]C{chip_id}[/] {badge}"
            f"  [cyan]{short}[/]  [gold1]+{pts}pts[/]{streak_txt}{first}"
        )

    def log_failure(self, chip_id: int, model: str) -> None:
        short  = _model_short(model)
        flavor = random.choice(_FAILURE_FLAVOR)
        self.write(
            f"[bold red]☠[/] [yellow]C{chip_id}[/] [dim]{short}[/]"
            f"  [red]−10pts[/]  [dim italic]{flavor}[/]"
        )

    def log_chip_done(self, chip_id: int, pts: int, best_streak: int) -> None:
        streak_note = f"  best 🔥×{best_streak}" if best_streak >= 2 else ""
        self.write(
            f"[bold green]{'═' * 34}[/]\n"
            f"[bold green]  C{chip_id} FLOOR CLEARED"
            f"  {pts}pts{streak_note}\n"
            f"{'═' * 34}[/]"
        )


class RallyBanner(Static):
    """Full-width banner that replaces the chip grid during a RALLY compile.

    Shown when all chips commit to one large model (mesh dispatch). Displays
    model name, chip count, live compile output from the lead chip, and a
    dim status row for each locked chip.
    """

    DEFAULT_CSS = """
    RallyBanner {
        display: none;
        width: 1fr;
        height: 1fr;
        border-left: double gold;
        border-top: double gold;
        border-bottom: double gold;
        padding: 1 2;
        color: $text;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._model_name = ""
        self._chip_ids: list[int] = []
        self._backend = ""
        self._confidence = 0.0

    def start(self, model: dict, chip_ids: list[int], decision) -> None:
        """Activate the banner for the given model and chip configuration."""
        self._model_name = _model_short(model.get("model_id", "?"))
        self._chip_ids   = chip_ids
        self._backend    = decision.backend
        self._confidence = decision.confidence
        chips_str = "·".join(str(c) for c in chip_ids)
        self.update(
            f"[bold gold]⚡⚡ RALLY — CHIPS {chips_str} ASSEMBLED ⚡⚡[/]\n"
            f"[dim]{self._model_name}  ·  {len(chip_ids)}-chip mesh  ·  "
            f"{self._backend}  ·  conf {self._confidence:.2f}[/]\n\n"
            f"[green]▶ Compiling on mesh {chips_str}...[/]\n"
        )

    def append_output(self, line: str) -> None:
        """Stream live output from the lead chip into the banner."""
        current = self.content
        lines = current.split("\n")
        lines.append(line.rstrip())
        header = lines[:4]
        tail   = lines[4:][-6:]
        self.update("\n".join(header + tail))


class ScoreStrip(Static):
    DEFAULT_CSS = """
    ScoreStrip {
        height: 6;
        dock: bottom;
        border-top: solid $primary;
        background: $surface;
    }
    """

    def __init__(self, num_chips: int, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._num_chips = num_chips

    def on_mount(self) -> None:
        self._poll()
        self.set_interval(0.5, self._poll)

    def _poll(self) -> None:
        rows = [_render_score_row(i) for i in range(self._num_chips)]
        while len(rows) < 4:
            rows.append(Text(""))
        self.update(Text("\n").join(rows))


# ── SetupScreen ───────────────────────────────────────────────────────────────

class SetupScreen(Screen):
    """Interactive config panel + live discovery/download log.

    Left pane: shows current parameter values; keyboard bindings let the
    user adjust them before pressing Enter to start discovery.
    Right pane: streams captured output from build_queues + _predownload_queues
    so the user can watch models being found in real time.
    """

    CSS = """
    SetupScreen {
        layout: vertical;
    }
    #setup-main {
        height: 1fr;
        layout: horizontal;
    }
    #config-pane {
        width: 40;
        height: 1fr;
        border-left: solid $primary;
        border-top: solid $primary;
        border-bottom: solid $primary;
        padding: 0 1;
    }
    #setup-log {
        width: 1fr;
        height: 1fr;
        border-left: solid $secondary;
        border-top: solid $secondary;
        border-bottom: solid $secondary;
        scrollbar-size: 0 0;
    }
    """

    BINDINGS = [
        Binding("enter",      "start",          "Start",         show=True),
        Binding("up",         "chips_up",        "Chips +1",      show=False),
        Binding("down",       "chips_down",      "Chips -1",      show=False),
        Binding("plus,equal", "limit_up",        "Limit +1",      show=False),
        Binding("minus",      "limit_down",      "Limit -1",      show=False),
        Binding("right_square_bracket", "dl_up",   "MinDL ×10",  show=False),
        Binding("left_square_bracket",  "dl_down", "MinDL ÷10",  show=False),
        Binding("period",     "likes_up",        "MinLikes +5",   show=False),
        Binding("comma",      "likes_down",      "MinLikes -5",   show=False),
        Binding("m",          "params_up",       "MaxParams +1B", show=False),
        Binding("n",          "params_down",     "MaxParams -1B", show=False),
        Binding("1",          "sources_all",     "Sources: all",  show=False),
        Binding("2",          "sources_seed",    "Sources: seed", show=False),
        Binding("3",          "sources_frontier","Sources: HF",   show=False),
        Binding("4",          "toggle_staples",  "Staples",       show=False),
        Binding("5",          "cycle_backend",   "Backend",       show=False),
        Binding("q",          "quit",            "Quit",          show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        # Param state — initialised from app attrs in on_mount.
        self._chips        = 1
        self._limit        = 0
        self._seed_only    = False
        self._frontier_only= False
        self._no_predownload = False
        self._min_downloads    = 50
        self._min_likes        = 1
        self._max_dl_like_ratio = 300
        self._max_params_b = 0.0
        self._allow_gated  = False
        self._max_cache_gb = 0.0
        self._session_download_max = 0.0
        self._parallel_downloads   = 4
        self._staples      = False  # True → include already-compiled seed models
        self._curated      = False  # True → use hand-curated showcase demo queue
        self._backend      = "auto"   # auto | forge | xla | mixed
        self._discovering  = False  # True while HF discovery / queue-build is running
        self._setup_done   = False  # True once queues are built (no re-run)
        self._autostart_secs = 4   # Countdown seconds before auto-start (0 = auto-fire immediately)

    def on_mount(self) -> None:
        app = self.app
        self._chips         = app.num_chips
        self._limit         = app.limit
        self._seed_only     = app.seed_only
        self._frontier_only = app.frontier_only
        self._no_predownload= app.no_predownload
        self._min_downloads     = app.min_downloads
        self._min_likes         = app.min_likes
        self._max_dl_like_ratio = getattr(app, "max_dl_like_ratio", 300)
        self._max_params_b  = app.max_params_b
        self._allow_gated   = app.allow_gated
        self._max_cache_gb  = app.max_cache_gb
        self._session_download_max = app.session_download_max
        self._parallel_downloads   = app.parallel_downloads
        self._staples              = app.staples
        self._curated              = getattr(app, "curated", False)
        self._backend              = getattr(app, "backend", "auto")
        confirm                    = getattr(app, "confirm", False)
        # confirm=False (default): fire immediately; confirm=True: 4s countdown + Enter.
        if not confirm:
            self._autostart_secs = 0
        self._refresh_config()
        if self._autostart_secs > 0:
            self._autostart_timer = self.set_interval(1.0, self._autostart_tick)
        else:
            # Auto-fire after layout settles (one event-loop tick).
            self.call_after_refresh(self.action_start)

    def _autostart_tick(self) -> None:
        """Decrement the auto-start countdown and fire when it reaches zero."""
        if self._discovering or self._setup_done:
            self._autostart_secs = 0
            return
        self._autostart_secs -= 1
        if self._autostart_secs <= 0:
            self._autostart_secs = 0
            self._autostart_timer.stop()
            self.action_start()
        else:
            self._refresh_config()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="setup-main"):
            yield Static("", id="config-pane")
            yield RichLog(markup=True, highlight=False,
                          auto_scroll=True, id="setup-log")
        yield Footer()

    # ── Config pane rendering ─────────────────────────────────────────────────

    def _refresh_config(self) -> None:
        app = self.app
        rn  = f"#{app.run_number:03d}"
        arch = app.arch.upper()

        limit_str = str(self._limit) if self._limit > 0 else "∞"
        dl_str    = f"{self._min_downloads:,}" if self._min_downloads else "off"
        lk_str    = str(self._min_likes) if self._min_likes else "off"
        ratio_str = str(self._max_dl_like_ratio) if self._max_dl_like_ratio else "off"
        if self._max_params_b > 0:
            pb_str = f"{self._max_params_b:.0f}B"
        else:
            pb_str = "∞"

        if self._seed_only:
            src_str = "seed only"
        elif self._frontier_only:
            src_str = "HF frontier"
        else:
            src_str = "ALL"

        staples_str = "[bold yellow]ON[/]" if self._staples else "off"
        backend_str = {
            "auto":  "[bold green]AUTO[/]  [dim]routes per-model[/]",
            "forge": "[bold]forge[/]",
            "xla":   "[bold cyan]XLA[/]",
            "mixed": "[bold yellow]MIXED[/]",
        }.get(self._backend, self._backend)

        if self._discovering:
            status = "[bold cyan]⚙ Discovering…[/]"
        elif self._autostart_secs > 0:
            status = f"[bold yellow]● ENTER to start  [dim](auto in {self._autostart_secs}s)[/][/]"
        elif getattr(self.app, "confirm", False):
            status = "[bold yellow]● Ready — press ENTER[/]"
        else:
            status = "[bold cyan]⚡ Auto-starting…[/]"

        lines = [
            f"[bold cyan]⚡ EXPEDITION {rn}[/]",
            f"[dim]{arch} · {self._chips} chip(s)[/]",
            "",
            "[bold]CONFIGURATION[/]",
            "─" * 34,
            f"  Chips        [bold]{self._chips}[/]    [dim]↑ / ↓[/]",
            f"  Limit        [bold]{limit_str}[/]   [dim]+ / -[/]",
            f"  Min Downloads[bold]{dl_str}[/]  [dim][ / ][/]",
            f"  Min Likes    [bold]{lk_str}[/]  [dim], / .[/]",
            f"  DL:Like Ratio[bold]{ratio_str}[/]  [dim](bot guard)[/]",
            f"  Max Params   [bold]{pb_str}[/]  [dim]m / n[/]",
            f"  Sources      [bold]{src_str}[/]  [dim]1/2/3[/]",
            f"  Staples      {staples_str}  [dim]4[/]",
            f"  Backend      {backend_str}  [dim]5[/]",
            "",
            "─" * 34,
            status,
        ]
        try:
            self.query_one("#config-pane", Static).update(
                Text.from_markup("\n".join(lines))
            )
        except Exception:
            pass

    # ── Parameter actions ─────────────────────────────────────────────────────

    def _guarded(fn):
        """Decorator: ignore param changes once discovery has started."""
        def wrapper(self):
            if not self._discovering:
                fn(self)
                self._refresh_config()
        return wrapper

    @_guarded
    def action_chips_up(self)   -> None: self._chips = min(self._chips + 1, 8)
    @_guarded
    def action_chips_down(self) -> None: self._chips = max(self._chips - 1, 1)
    @_guarded
    def action_limit_up(self)   -> None: self._limit += 1
    @_guarded
    def action_limit_down(self) -> None: self._limit = max(self._limit - 1, 0)
    @_guarded
    def action_dl_up(self)      -> None: self._min_downloads = max(1, self._min_downloads) * 10
    @_guarded
    def action_dl_down(self)    -> None: self._min_downloads = max(self._min_downloads // 10, 0)
    @_guarded
    def action_likes_up(self)   -> None: self._min_likes += 5
    @_guarded
    def action_likes_down(self) -> None: self._min_likes = max(self._min_likes - 5, 0)
    @_guarded
    def action_params_up(self)  -> None: self._max_params_b += 1.0
    @_guarded
    def action_params_down(self)-> None: self._max_params_b = max(self._max_params_b - 1.0, 0.0)
    @_guarded
    def action_sources_all(self)     -> None: self._seed_only = False; self._frontier_only = False
    @_guarded
    def action_sources_seed(self)    -> None: self._seed_only = True;  self._frontier_only = False
    @_guarded
    def action_sources_frontier(self)-> None: self._seed_only = False; self._frontier_only = True
    @_guarded
    def action_toggle_staples(self)  -> None: self._staples = not self._staples
    @_guarded
    def action_cycle_backend(self)   -> None: self._backend = {"auto": "forge", "forge": "xla", "xla": "mixed", "mixed": "auto"}[self._backend]

    def action_start(self) -> None:
        if self._discovering or self._setup_done:
            return
        self._discovering = True
        self._autostart_secs = 0
        try:
            self._autostart_timer.stop()
        except Exception:
            pass
        self._refresh_config()
        log = self.query_one("#setup-log", RichLog)
        log.write("[bold cyan]⚡ Starting expedition setup…[/]")
        self._do_setup()

    # ── Discovery worker (runs in thread pool) ────────────────────────────────

    @work(thread=True)
    def _do_setup(self) -> None:
        """Build chip queues in a thread and stream progress to the setup log.

        Calls the lower-level _scan_forge_models / _scan_frontier / _predownload
        functions directly rather than going through build_queues.  This avoids
        two problems with calling build_queues from a TUI context:

        1. build_queues calls _with_spinner which starts a daemon thread that
           writes \r-spinner frames directly to sys.stdout.  In TUI mode that
           writes into the Textual terminal driver, corrupting the display.

        2. contextlib.redirect_stdout is process-global (not thread-local) so
           redirecting it in the worker thread intercepts Textual's own output.
        """
        log = self.query_one("#setup-log", RichLog)
        app = self.app

        def _log(markup: str) -> None:
            app.call_from_thread(log.write, Text.from_markup(markup))

        try:
            self._do_setup_body(log, app, _log)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            _log(f"[bold red]✗ Setup failed:[/]\n[dim]{tb}[/]")
            # Also write to a file so it's readable after force-quit.
            try:
                Path("/tmp/expedition_tui_error.txt").write_text(tb)
            except Exception:
                pass

    def _do_setup_body(self, log, app, _log) -> None:
        """Inner body of _do_setup — separated so the outer method can catch all exceptions."""
        # Lazily import lower-level helpers from expedition (avoids circular
        # import at module load time since expedition imports expedition_tui).
        from expedition import (
            _scan_forge_models,
            _scan_frontier,
            _dedup_by_author_family,
            _interleave,
            _predownload_queues,
            _build_curated_queue,
            BESTIARY_PATH,
        )
        from lib.expedition.bestiary import Bestiary

        # ── Curated showcase queue ────────────────────────────────────────────
        if self._curated:
            _log("[bold cyan]⚡ Curated demo queue — 5 hand-picked models[/]")
            chip_queues, side_quest_pool = _build_curated_queue(self._chips)
            for ci, q in enumerate(chip_queues):
                for item in q:
                    mid   = item.get("model_id", "?")
                    task  = item.get("task", "")
                    mesh  = item.get("mesh_chips", 1)
                    chips_str = f"  [yellow]×{mesh} chips[/]" if mesh > 1 else ""
                    _log(f"  [dim]C{ci}[/] [bold]{mid}[/]  [dim]{task}[/]{chips_str}")
            total = sum(len(q) for q in chip_queues)
            _log(f"[bold green]✓ {total} model(s) → {self._chips} chip(s)[/]")
            app.call_from_thread(self._advance_to_run, chip_queues, side_quest_pool)
            return

        bestiary     = Bestiary(path=str(BESTIARY_PATH))
        compiled_ids = set(bestiary.compiled.keys())

        seed_items:     list[dict] = []
        frontier_items: list[dict] = []

        # ── Seed scan (forge-models library) ─────────────────────────────────
        # Framework map must be defined before the frontier_only branch so that
        # scan_fw is available for the frontier library filter regardless of mode.
        fw_map  = {"auto": None, "forge": "pytorch", "xla": "jax", "mixed": None}
        scan_fw = fw_map.get(self._backend, "pytorch")
        seed_items: list[dict] = []  # populated below; empty when frontier_only=True

        # For mixed backend, scan both frameworks and separate into two pools so
        # each chip gets loaders it can actually run.
        if not self._frontier_only:
            label = "⚙ Scanning tt-forge-models library (staples — all included)..." if self._staples \
                else "⚙ Scanning tt-forge-models library..."
            _log(f"[cyan]{label}[/]")
            seed_items = _scan_forge_models(compiled_ids, include_all=self._staples,
                                            framework=scan_fw)
            _log(f"[green]✓ {len(seed_items)} seed model(s) found[/]")
            for item in seed_items:
                mid  = item.get("model_id", "?")
                task = item.get("task") or item.get("source") or ""
                _log(f"  [dim]· {mid}  {task}[/]")

            # Canary injection: if all forge-models are already compiled,
            # pick one random one anyway so we always have ≥1 model to run.
            if not seed_items and not self._staples:
                all_seeds = _scan_forge_models(set(), include_all=True, framework=scan_fw)
                if all_seeds:
                    import random as _random
                    canary = dict(_random.choice(all_seeds))
                    seed_items = [canary]
                    _log(f"[yellow]⚡ canary injected — all forge models compiled; re-testing {canary['model_id']}[/]")

        forge_ids = {item["model_id"] for item in seed_items}

        # ── HF frontier discovery (slow — network) ────────────────────────────
        if not self._seed_only:
            _log("[cyan]⚙ Querying HuggingFace frontier (may take 30-60s)...[/]")
            frontier_items = _scan_frontier(
                compiled_ids,
                forge_ids,
                min_downloads    = self._min_downloads,
                min_likes        = self._min_likes,
                max_dl_like_ratio = self._max_dl_like_ratio,
                max_params_b     = self._max_params_b,
                skip_gated       = not self._allow_gated,
                library          = scan_fw,
            )
            _log(f"[green]✓ {len(frontier_items)} frontier model(s) discovered[/]")
            for item in frontier_items:
                mid  = item.get("model_id", "?")
                task = item.get("task") or ""
                _log(f"  [cyan]+ {mid}  [dim]{task}[/][/]")

        # ── Dedup + interleave ────────────────────────────────────────────────
        if frontier_items and (self._limit == 0 or self._limit < 100):
            frontier_items, n_dropped = _dedup_by_author_family(
                frontier_items, target_params_b=self._max_params_b
            )
            if n_dropped:
                _log(f"[dim]{n_dropped} family duplicate(s) dropped[/]")

        all_items = _interleave(seed_items, frontier_items, seed_ratio=0.6)

        # Deduplicate (belt-and-suspenders against duplicate model IDs).
        seen: set[str] = set()
        deduped: list[dict] = []
        for item in all_items:
            mid = item["model_id"]
            if mid not in seen:
                seen.add(mid)
                deduped.append(item)
        all_items = deduped

        if self._limit > 0:
            all_items = all_items[: self._limit]

        chip_queues: list[list[dict]] = [[] for _ in range(self._chips)]
        for i, item in enumerate(all_items):
            chip_queues[i % self._chips].append(item)

        total = sum(len(q) for q in chip_queues)
        per   = total // max(self._chips, 1)
        _log(f"[bold green]✓ {total} model(s) → {self._chips} chip(s)  (~{per} each)[/]")

        # ── Pre-download frontier weights ─────────────────────────────────────
        if not self._no_predownload:
            frontier_total = sum(
                1 for q in chip_queues for m in q if m.get("is_frontier")
            )
            if frontier_total > 0:
                _log(f"[cyan]⚙ Pre-downloading {frontier_total} frontier model(s)...[/]")
                # Capture predownload stdout (progress bars etc.) with a writer
                # that handles \r overwrite lines and forwards non-empty content.
                buf = [""]
                class _DLWriter(io.TextIOBase):
                    def write(self_, text: str) -> int:
                        buf[0] += text
                        if "\r" in buf[0]:
                            buf[0] = buf[0].split("\r")[-1]
                        lines = buf[0].split("\n")
                        buf[0] = lines.pop()
                        for line in lines:
                            stripped = _RE_ANSI.sub("", line).strip()
                            if stripped:
                                try:
                                    entry = Text.from_ansi(line)
                                except Exception:
                                    entry = Text(stripped)
                                app.call_from_thread(log.write, entry)
                        return len(text)
                    def flush(self_) -> None: pass

                with contextlib.redirect_stdout(_DLWriter()):
                    _predownload_queues(
                        chip_queues,
                        max_cache_gb            = self._max_cache_gb,
                        session_download_max_gb = self._session_download_max,
                        parallel_downloads      = self._parallel_downloads,
                    )
                _log("[green]✓ Pre-download complete[/]")

        # ── Write queue files ─────────────────────────────────────────────────
        for chip_id, queue in enumerate(chip_queues):
            Path(f"/tmp/expedition_queue_chip{chip_id}.json").write_text(
                json.dumps(queue, indent=2)
            )

        _log(f"\n[bold green]✓ Ready — launching {self._chips} chip(s)...[/]")
        app.call_from_thread(self._advance_to_run, chip_queues)

    def _advance_to_run(self, chip_queues: list[list[dict]],
                        side_quest_pool: list[dict] | None = None) -> None:
        """Called on the event loop thread when setup completes.

        push_screen is synchronous in Textual 7.x (returns AwaitMount, not a
        coroutine), so this method stays sync.  call_from_thread schedules it on
        the event loop via run_coroutine_threadsafe.
        """
        self._discovering = False
        self._setup_done  = True
        self.app.push_screen(
            RunScreen(
                chip_queues      = chip_queues,
                num_chips        = self._chips,
                run_number       = self.app.run_number,
                arch             = self.app.arch,
                project_dir      = self.app._project_dir,
                backend          = self._backend,
                side_quest_pool  = side_quest_pool or [],
            )
        )


def _chip_backend(chip_id: int, backend: str) -> str:
    """Return the effective backend for a chip given the run's backend setting."""
    if backend == "mixed":
        return "forge" if chip_id % 2 == 0 else "xla"
    return backend


def _chip_label(chip_id: int, backend: str) -> str:
    """Return a badge string like '[XLA]' or '' for the chip panel title."""
    be = _chip_backend(chip_id, backend)
    return "\033[96m[XLA]\033[0m" if be == "xla" else ""


# ── RunScreen ─────────────────────────────────────────────────────────────────

class RunScreen(Screen):
    """Live chip-panel layout. Refactored from the original ExpeditionTUI body."""

    CSS = """
    RunScreen {
        layout: vertical;
    }
    #main {
        height: 1fr;
        layout: horizontal;
    }
    #chip-grid {
        width: 3fr;
        height: 1fr;
        layout: vertical;
    }
    #chip-row-top {
        height: 1fr;
        layout: horizontal;
    }
    #chip-row-bottom {
        height: 1fr;
        layout: horizontal;
    }
    #sidebar {
        width: 36;
        height: 1fr;
        layout: vertical;
    }
    #rally-banner {
        display: none;
        width: 3fr;
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("b", "show_bestiary", "Bestiary",   show=True),
        Binding("t", "show_toplike",  "tt-toplike", show=True),
        Binding("q", "quit",          "Quit",       show=True),
    ]

    def __init__(self, chip_queues: list[list[dict]], num_chips: int,
                 run_number: int, arch: str, project_dir: Path,
                 backend: str = "forge",
                 side_quest_pool: list[dict] | None = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.chip_queues  = chip_queues
        self.num_chips    = num_chips
        self.run_number   = run_number
        self.arch         = arch
        self._project_dir = project_dir
        self.backend      = backend   # auto | forge | xla | mixed
        self._run_state   = RunState(num_chips, run_number)
        self._done_count  = 0
        # ── Per-model dispatcher state ────────────────────────────────────────
        # Flatten chip_queues round-robin into a single ordered pool.
        self._model_pool: list[dict] = []
        for i in range(max(len(q) for q in chip_queues) if chip_queues else 0):
            for q in chip_queues:
                if i < len(q):
                    self._model_pool.append(q[i])
        self._free_chips: set[int]  = set(range(num_chips))
        self._mesh_holding: dict | None = None
        self._opportunist_active: bool  = False
        self._chip_first_dispatch: set[int] = set()
        self._bestiary = None
        self._all_done: bool = False
        # Side quest state — chips juggle bonus models while waiting for RALLY.
        self._side_quest_pool:      list[dict]                                 = list(side_quest_pool or [])
        self._side_quest_chips:     set[int]                                   = set()
        self._side_quest_procs:     dict[int, asyncio.subprocess.Process]      = {}
        self._rally_interrupt_flag: bool                                        = False
        self._rally_in_progress:    bool                                        = False

    def compose(self) -> ComposeResult:
        rn = f"Run #{self.run_number:03d}"
        yield Header(show_clock=True)
        self.app.title     = f"EXPEDITION  {rn}"
        self.app.sub_title = f"{self.num_chips} chip(s) · {self.arch.upper()}"

        with Horizontal(id="main"):
            with Vertical(id="chip-grid"):
                with Horizontal(id="chip-row-top"):
                    if self.num_chips >= 1:
                        yield ChipPanel(0, f"⚔ CHIP 0  {_chip_label(0, self.backend)}  {_ADVENTURER_TITLES[0]}", id="chip-0")
                    if self.num_chips >= 2:
                        yield ChipPanel(1, f"⚔ CHIP 1  {_chip_label(1, self.backend)}  {_ADVENTURER_TITLES[1]}", id="chip-1")
                if self.num_chips >= 3:
                    with Horizontal(id="chip-row-bottom"):
                        yield ChipPanel(2, f"⚔ CHIP 2  {_chip_label(2, self.backend)}  {_ADVENTURER_TITLES[2]}", id="chip-2")
                        if self.num_chips >= 4:
                            yield ChipPanel(3, f"⚔ CHIP 3  {_chip_label(3, self.backend)}  {_ADVENTURER_TITLES[3]}", id="chip-3")
            yield RallyBanner(id="rally-banner")
            with Vertical(id="sidebar"):
                yield HardwareWidget(id="hw")
                yield EventLog(id="event-log")

        yield ScoreStrip(self.num_chips, id="score-strip")
        yield Footer()

    def on_mount(self) -> None:
        # Write a placeholder run JSON immediately so the next run's counter
        # increments correctly.  expedition.py computes run_number as
        # len(glob("run_*.json")) + 1, so without this file the TUI number
        # never advances past the last non-TUI run.
        try:
            runs_dir = self._project_dir / "data" / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_file = runs_dir / f"run_{self.run_number:03d}.json"
            if not run_file.exists():
                run_file.write_text(json.dumps({
                    "run":       self.run_number,
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "chips":     self.num_chips,
                    "tui":       True,
                }, indent=2))
        except Exception:
            pass

        # Clear stale per-chip temp files from previous runs so the score strip
        # and summary screen don't show stale state.
        status_dir = Path(os.environ.get("EXPEDITION_STATUS_DIR", "/tmp"))
        for chip_id in range(self.num_chips):
            for fname in (
                status_dir / f"expedition_chip_{chip_id}.status",
                Path(f"/tmp/expedition_results_chip{chip_id}.csv"),
            ):
                try:
                    fname.unlink(missing_ok=True)
                except Exception:
                    pass

        # Load bestiary for router queries (read-only at run time).
        from lib.expedition.bestiary import Bestiary as _Bestiary
        self._bestiary = _Bestiary(path=str(self._project_dir / "data" / "bestiary.json"))

        # Seed the dispatcher — each free chip gets its first model.
        self._dispatch_next()

        # Watchdog: if the reactive dispatch misses the "all done" condition
        # (e.g. a @work coroutine raised before calling _on_chip_free), a
        # periodic timer detects when all status files say done=1 and fires
        # _on_all_done directly.
        self.set_interval(2.0, self._watchdog_check)

    def _watchdog_check(self) -> None:
        """Periodic fallback: if all chips report done but the dispatcher
        never reached _on_all_done (e.g. due to a swallowed exception in
        a @work coroutine), detect it here and fire the transition."""
        if self._all_done:
            return
        # Do not fire while a RALLY subprocess is running — _launch_model will
        # push WaveFinaleScreen after the subprocess exits.
        if self._rally_in_progress:
            return
        # Only trigger once pool is drained and no mesh is assembling.
        if self._model_pool or self._mesh_holding:
            return
        # Check that every chip's status file says done=1.
        all_done = all(
            _read_status(cid).get("done", "0") == "1"
            for cid in range(self.num_chips)
        )
        if all_done:
            # Forcefully free every chip so _on_all_done can proceed.
            self._free_chips = set(range(self.num_chips))
            self._on_all_done()

    def _get_decision(self, model: dict):
        """Compute a DispatchDecision for model, respecting self.backend override."""
        from lib.expedition.router import route_model, DispatchDecision
        if self.backend == "auto":
            return route_model(model, self._bestiary,
                               available_chips=set(range(self.num_chips)))
        else:
            chip_be = _chip_backend(0, self.backend) if self.backend != "mixed" else "forge"
            return DispatchDecision(
                backend=chip_be,
                chips=min(max(1, int(model.get("mesh_chips", 1) or 1)), self.num_chips),
                confidence=1.0,
                reason="manual",
            )

    def _dispatch_next(self) -> None:
        """Find the next dispatchable model and launch it. Called at mount and after each chip completes."""
        # Check if a waiting mesh model now has quorum.
        if self._mesh_holding:
            chips_needed: int = self._mesh_holding["chips_needed"]
            # Count idle + still-running side-quest chips toward eventual quorum.
            all_available = self._free_chips | self._side_quest_chips
            if len(all_available) >= chips_needed:
                # Enough chips will eventually be free — stop dispatching new side quests
                # but let existing ones finish naturally (no killing).
                self._rally_interrupt_flag = True
                # Fire RALLY only once all needed chips are genuinely idle.
                if len(self._free_chips) >= chips_needed:
                    chip_ids = sorted(self._free_chips)[:chips_needed]
                    self._fire_rally(self._mesh_holding, chip_ids)
                    return
                # Rally is pending — chips still finishing side quests. Wait.
                return
            # Not enough chips will be available yet — fall through to side quest dispatch.

        # Scan the pool for a dispatchable model.
        for i, model in enumerate(self._model_pool):
            decision = self._get_decision(model)

            if decision.chips == 1:
                if self._free_chips:
                    chip_id = min(self._free_chips)
                    self._free_chips.discard(chip_id)
                    self._model_pool.pop(i)
                    self._launch_model(chip_id, model, decision)
                    # Keep scanning — other free chips may still need work.
                    self._dispatch_next()
                    return
            else:
                # Multi-chip model: hold it and keep scanning for single-chip work.
                if self._mesh_holding is None:
                    self._mesh_holding = {
                        **model,
                        "chips_needed": decision.chips,
                        "decision": decision,
                    }
                    self._opportunist_active = True
                    self._model_pool.pop(i)
                    try:
                        el = self.query_one("#event-log", EventLog)
                        el.write(
                            f"[yellow]⏳ MESH ASSEMBLING — "
                            f"{model.get('model_id', '?').split('/')[-1]} "
                            f"needs {decision.chips} chips[/]"
                        )
                    except Exception:
                        pass
                    self._dispatch_next()
                    return
                continue

        # Juggle while waiting for RALLY quorum: launch a side quest on any free chip.
        # Only fires when a mesh model is assembling, there's a free chip, and we have
        # models in the side quest pool and the RALLY interrupt hasn't been signalled.
        if (self._mesh_holding
                and self._free_chips
                and self._side_quest_pool
                and not self._rally_interrupt_flag):
            chip_id = min(self._free_chips)
            self._free_chips.discard(chip_id)
            self._side_quest_chips.add(chip_id)
            model = self._side_quest_pool.pop(0)
            self._launch_side_quest(chip_id, model)
            # Keep scanning — multiple chips may be free simultaneously.
            self._dispatch_next()
            return

        # Check if run is complete: pool empty, no mesh holding, all chips free.
        if not self._model_pool and not self._mesh_holding and len(self._free_chips) == self.num_chips:
            self._on_all_done()

    @work
    async def _launch_model(self, chip_id: int, model: dict, decision,
                            mesh_chip_ids: list[int] | None = None) -> None:
        """Launch a worker subprocess for one model on one or more chips."""
        # First dispatch to each chip: stagger by chip_id * 2 seconds.
        if chip_id not in self._chip_first_dispatch:
            self._chip_first_dispatch.add(chip_id)
            if chip_id > 0:
                await asyncio.sleep(chip_id * 2)

        # Write the model dict to a temp JSON file.
        # Strip dispatcher-internal keys that were added by _dispatch_next/_fire_rally
        # and would break QueueItem(**data) deserialization in the worker.
        # (_WORKER_SKIP_KEYS is defined at module level to avoid recreating it per call.)
        model_for_worker = {k: v for k, v in model.items() if k not in _WORKER_SKIP_KEYS}
        model_json_path = f"/tmp/expedition_model_chip{chip_id}.json"
        Path(model_json_path).write_text(json.dumps(model_for_worker))

        # Results CSV path (append mode — one file per chip across all models).
        results_path = f"/tmp/expedition_results_chip{chip_id}.csv"

        # Determine backend.
        if self.backend == "mixed":
            chip_be = _chip_backend(chip_id, "mixed")
        else:
            chip_be = decision.backend

        # Build env for this chip.
        if mesh_chip_ids:
            visible = ",".join(str(c) for c in mesh_chip_ids)
        else:
            visible = str(chip_id)

        if chip_be == "xla":
            python_exe  = str(Path.home() / "tt-xla" / "venv" / "bin" / "python3")
            worker_path = str(self._project_dir / "lib" / "expedition" / "expedition_worker_xla.py")
            # Start from parent env minus forge-specific TT-Metal vars.
            # TT_MESH_GRAPH_DESC_PATH must NOT be inherited — it holds the forge
            # single-chip p100 descriptor which overrides TT_VISIBLE_DEVICES and
            # limits JAX to 1 device even when 4 are requested (verified: the p100
            # textproto declares dims:[1,1]).  JAX enumerates devices from
            # TT_VISIBLE_DEVICES on its own without needing a mesh descriptor file.
            env = {
                k: v for k, v in os.environ.items()
                if k not in ("TT_METAL_HOME", "TT_MESH_GRAPH_DESC_PATH")
            }
            env.update({
                "TT_VISIBLE_DEVICES":      visible,
                "TT_METAL_LOGGER_LEVEL":   "FATAL",
                "JAX_PLATFORMS":           "tt",
                "PYTHONUNBUFFERED":        "1",
            })
        else:
            python_exe  = sys.executable
            worker_path = str(self._project_dir / "lib" / "expedition" / "expedition_worker.py")
            env = {
                **os.environ,
                "TT_VISIBLE_DEVICES":      visible,
                "TT_METAL_ARCH_NAME":      self.arch,
                "TT_METAL_LOGGER_LEVEL":   "FATAL",
                "TT_MESH_GRAPH_DESC_PATH": str(
                    self._project_dir / "mesh_graph_descriptors"
                    / "p100_mesh_graph_descriptor.textproto"
                ),
                "PYTHONUNBUFFERED":        "1",
            }

        # ── RALLY: suspend TUI entirely so MOGWAI RAVE TAPES renders raw ────────────
        # Uses the same app.suspend() pattern as action_show_toplike. The event
        # loop blocks on subprocess.run() but the TUI is suspended so nothing needs
        # to refresh. When the subprocess exits TUI resumes automatically.
        if mesh_chip_ids:
            cmd = [
                python_exe, worker_path,
                "--chip",       str(chip_id),
                "--run",        str(self.run_number),
                "--bestiary",   str(self._project_dir / "data" / "bestiary.json"),
                "--model-json", model_json_path,
                "--results",    results_path,
            ]
            with self.app.suspend():
                subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL)
            self._rally_in_progress = False
            # Update state counters — do NOT call _on_chip_free/_dispatch_next/_on_all_done,
            # which would flash the RunScreen grid and show the countdown before pushing
            # WaveFinaleScreen.  Go straight to the finale instead.
            for cid in mesh_chip_ids:
                self._free_chips.add(cid)
                self._done_count += 1
            if not self._all_done:
                self._all_done = True
                self.app.push_screen(WaveFinaleScreen(
                    self.num_chips, self.run_number,
                    auto_quit_secs=getattr(self.app, "auto_quit_secs", 0),
                ))
            return

        # Write confidence label to the chip panel.
        try:
            panel = self.query_one(f"#chip-{chip_id}", ChipPanel)
            chip_label = mesh_chip_ids or [chip_id]
            panel.write_line(
                f"\033[2m  routing: {chip_be} · conf {decision.confidence:.2f} "
                f"· {len(chip_label)}-chip\033[0m\n"
            )
        except Exception:
            pass

        # Tell RunState which model is about to compile so _parse_for_events
        # can get the model name without reading the status file.
        self._run_state.set_current(chip_id, model.get("model_id", ""))

        proc = await asyncio.create_subprocess_exec(
            python_exe,
            worker_path,
            "--chip",       str(chip_id),
            "--run",        str(self.run_number),
            "--bestiary",   str(self._project_dir / "data" / "bestiary.json"),
            "--model-json", model_json_path,
            "--results",    results_path,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )

        try:
            panel = self.query_one(f"#chip-{chip_id}", ChipPanel)
        except Exception:
            panel = None

        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace")
            if panel:
                panel.write_line(line)
            self._parse_for_events(chip_id, line)

        await proc.wait()
        if panel:
            panel.mark_done(proc.returncode == 0)

        # Absorb the CSV row the worker just wrote into RunState so that
        # WaveFinaleScreen and SummaryScreen never need to read CSV files.
        _absorb_csv_row(self._run_state, chip_id, results_path, is_sq=False)
        c = self._run_state.chip(chip_id)
        try:
            el = self.query_one("#event-log", EventLog)
            el.log_chip_done(chip_id, c.pts, c.best_streak)
        except Exception:
            pass

        # Free the chip and continue dispatching.
        # (mesh_chip_ids path returns early above — only single-chip runs reach here.)
        self._on_chip_free(chip_id)

    def _on_chip_free(self, chip_id: int) -> None:
        """Mark a chip as free and trigger the next dispatch cycle."""
        self._free_chips.add(chip_id)
        self._done_count += 1
        self._dispatch_next()

    @work
    async def _launch_side_quest(self, chip_id: int, model: dict) -> None:
        """Run a bonus model on an idle chip while waiting for RALLY quorum.

        Uses the same forge worker subprocess infrastructure as _launch_model.
        Exits cleanly without touching chip state if _rally_interrupt_flag is
        set — _fire_rally owns the chip at that point.
        """
        import time as _time
        start_t = _time.time()
        display = model.get("display_name") or _model_short(model.get("model_id", "?"))

        try:
            el = self.query_one("#event-log", EventLog)
            el.write(f"[cyan]⚡ C{chip_id} SIDE QUEST — {display}[/]")
        except Exception:
            pass

        try:
            panel = self.query_one(f"#chip-{chip_id}", ChipPanel)
            panel.write_line(f"\033[36m⚡ SIDE QUEST: {display}\033[0m\n")
        except Exception:
            panel = None

        # Write model JSON to a separate temp file (avoids clobbering main worker files).
        model_for_worker = {k: v for k, v in model.items() if k not in _WORKER_SKIP_KEYS}
        model_json_path  = f"/tmp/expedition_model_chip{chip_id}_sq.json"
        # Side quests write to a separate results CSV so the summary screen can
        # distinguish them, and use a separate EXPEDITION_STATUS_DIR so the
        # score strip status file isn't clobbered (wiping the main-queue pts).
        sq_status_dir    = "/tmp/expedition_sq"
        Path(sq_status_dir).mkdir(exist_ok=True)
        results_path     = f"/tmp/expedition_results_chip{chip_id}_sq.csv"
        Path(model_json_path).write_text(json.dumps(model_for_worker))

        python_exe  = sys.executable
        worker_path = str(self._project_dir / "lib" / "expedition" / "expedition_worker.py")
        env = {
            **os.environ,
            "TT_VISIBLE_DEVICES":        str(chip_id),
            "TT_METAL_ARCH_NAME":        self.arch,
            "TT_METAL_LOGGER_LEVEL":     "FATAL",
            "TT_MESH_GRAPH_DESC_PATH":   str(
                self._project_dir / "mesh_graph_descriptors"
                / "p100_mesh_graph_descriptor.textproto"
            ),
            "EXPEDITION_STATUS_DIR":     sq_status_dir,
            "PYTHONUNBUFFERED":          "1",
        }

        # Tell RunState which side quest model is about to compile.
        self._run_state.set_current(chip_id, model.get("model_id", ""))

        proc = await asyncio.create_subprocess_exec(
            python_exe, worker_path,
            "--chip",       str(chip_id),
            "--run",        str(self.run_number),
            "--bestiary",   str(self._project_dir / "data" / "bestiary.json"),
            "--model-json", model_json_path,
            "--results",    results_path,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )
        self._side_quest_procs[chip_id] = proc

        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace")
            if panel:
                panel.write_line(line)
            self._parse_for_events(chip_id, line, status_dir=sq_status_dir)

        await proc.wait()
        elapsed = _time.time() - start_t

        # Clean up side quest tracking before freeing the chip.
        self._side_quest_procs.pop(chip_id, None)
        self._side_quest_chips.discard(chip_id)

        if panel:
            panel.mark_done(proc.returncode == 0)

        # Absorb the side quest CSV row into RunState (is_sq=True).
        _absorb_csv_row(self._run_state, chip_id, results_path, is_sq=True)
        sq_pts = sum(r.pts for r in self._run_state.chip(chip_id).results if r.is_sq)

        if proc.returncode == 0:
            if   elapsed < 10: speed_label = f"⚡ {elapsed:.1f}s — BLAZING"
            elif elapsed < 20: speed_label = f"⚡ {elapsed:.1f}s — fast"
            else:               speed_label = f"{elapsed:.1f}s"
            try:
                el = self.query_one("#event-log", EventLog)
                el.write(
                    f"[bold cyan]⚡ C{chip_id} BONUS [gold1]+{sq_pts}pts[/][bold cyan]"
                    f" ★ {display} — {speed_label}[/]"
                )
            except Exception:
                pass
        else:
            try:
                el = self.query_one("#event-log", EventLog)
                el.write(f"[dim]⚡ C{chip_id} SIDE QUEST FAIL — {display} ({elapsed:.1f}s)[/]")
            except Exception:
                pass

        # Free chip — may trigger another side quest or RALLY quorum check.
        self._free_chips.add(chip_id)
        self._dispatch_next()

    @work
    async def _on_all_done(self) -> None:
        """Animate the completion banner then push to SummaryScreen.

        Guard against double-invocation: if _chip_complete fires simultaneously
        for the last two chips (e.g. mesh run), only the first call proceeds.
        """
        if self._all_done:
            return
        self._all_done = True
        try:
            el = self.query_one("#event-log", EventLog)
            el.write(f"\n[bold green]{'═'*34}[/]")
            el.write("[bold green]  ⚡ ALL CHIPS COMPLETE[/]")
            for n in (3, 2, 1):
                el.write(f"[dim]  → Results in {n}...[/]")
                await asyncio.sleep(0.8)
            el.write(f"[bold green]{'═'*34}[/]")
        except Exception:
            await asyncio.sleep(2.4)
        self.app.push_screen(WaveFinaleScreen(
            self.num_chips,
            self.run_number,
            self._run_state,
            auto_quit_secs=getattr(self.app, "auto_quit_secs", 0),
        ))

    @work
    async def _fire_rally(self, mesh_model: dict, chip_ids: list[int]) -> None:
        """Handle a RALLY event: show banner, fire mesh subprocess.

        All side quests have already finished naturally before this is called —
        _dispatch_next only fires here once _free_chips >= chips_needed.
        No killing, no device-settle wait.
        """
        self._mesh_holding       = None
        self._opportunist_active = False
        self._rally_in_progress  = True   # block watchdog from firing _on_all_done

        # Clear /dev/shm forge segments from completed side-quest workers.
        # They exited cleanly, but forge sometimes leaves a segment behind.
        import glob as _glob
        for seg in _glob.glob("/dev/shm/sm_segment.*"):
            try:
                __import__("os").unlink(seg)
            except Exception:
                pass

        for cid in chip_ids:
            self._free_chips.discard(cid)

        decision = mesh_model.get("decision")

        try:
            el = self.query_one("#event-log", EventLog)
            chips_str = "+".join(str(c) for c in chip_ids)
            el.write(
                f"[bold gold]⚡ RALLY — {mesh_model.get('model_id','?').split('/')[-1]} "
                f"on chips {chips_str} — dropping to raw terminal[/]"
            )
        except Exception:
            pass

        # Launch a single multi-chip subprocess on the lead chip.
        # _rally_in_progress remains True — _launch_model clears it after subprocess exits.
        lead = chip_ids[0]
        self._launch_model(lead, mesh_model, decision, mesh_chip_ids=chip_ids)

    def _parse_for_events(self, chip_id: int, line: str,
                          status_dir: str | None = None) -> None:
        # Rarity label arrives before SUCCESS/FAILED — store it in RunState so
        # the event-log call below can read it without a status-file read.
        rm = _RE_RARITY.search(line)
        if rm:
            raw    = rm.group(1).upper()
            rarity = ("legendary" if "LEGENDARY" in raw
                      else "rare"      if "RARE"      in raw
                      else "uncommon"  if "UNCOMMON"  in raw
                      else "common")
            self._run_state.set_rarity(chip_id, rarity)

        try:
            combat = self.query_one("#event-log", EventLog)
        except Exception:
            return

        if _RE_SUCCESS.search(line):
            pts_m  = _RE_PTS.search(line)
            pts    = abs(int(pts_m.group(1))) if pts_m else 0
            str_m  = _RE_STREAK.search(line)
            streak = int(str_m.group(1)) if str_m else 0
            first  = bool(_RE_FIRST.search(line))
            c      = self._run_state.chip(chip_id)
            model  = c.current_model   # set by _launch_model before subprocess starts
            rarity = c.rarity
            combat.log_success(chip_id, model, rarity, pts, first, streak)
            self._run_state.set_rarity(chip_id, "common")

        elif _RE_FAILURE.search(line):
            c     = self._run_state.chip(chip_id)
            model = c.current_model
            combat.log_failure(chip_id, model)
            self._run_state.set_rarity(chip_id, "common")

    def action_show_toplike(self) -> None:
        if not shutil.which("tt-toplike"):
            try:
                combat = self.query_one("#event-log", EventLog)
                combat.write("[red]tt-toplike not found — install tenstorrent-software-utils[/]")
            except Exception:
                pass
            return
        with self.app.suspend():
            subprocess.run(["tt-toplike", "--mode", "arcade"])

    def action_show_bestiary(self) -> None:
        try:
            result = subprocess.run(
                [sys.executable, str(self._project_dir / "expedition.py"), "summary"],
                capture_output=True, text=True, timeout=15,
            )
            output = result.stdout or "(no bestiary output)"
        except Exception as exc:
            output = f"Error running summary: {exc}"

        try:
            combat = self.query_one("#event-log", EventLog)
            combat.write(f"\n[bold cyan]{'═' * 34}[/]")
            combat.write("[bold cyan]  ★ EXPEDITION BESTIARY[/]")
            combat.write(f"[bold cyan]{'═' * 34}[/]")
            for ln in output.splitlines()[-20:]:
                try:
                    combat.write(Text.from_ansi(ln))
                except Exception:
                    combat.write(ln)
            combat.write(f"[bold cyan]{'═' * 34}[/]\n")
        except Exception:
            pass


# ── Finale wave animation ──────────────────────────────────────────────────────

# 256-colour ANSI palette matching the Tenstorrent dark theme
_WAVE_PAL = {
    'teal':  "\033[38;5;87m",
    'cyan':  "\033[38;5;51m",
    'lteal': "\033[38;5;159m",
    'blue':  "\033[38;5;75m",
    'pink':  "\033[38;5;213m",
    'lpurp': "\033[38;5;183m",
    'gold':  "\033[38;5;220m",
}

# Mirrored wave geometry: (base_y_frac, amp_frac, freq_cycles, time_speed, phase, color)
# Gold hairline at dead centre; teal/cyan bookend top and bottom.
_WAVE_DEFS = [
    (0.04, 0.030, 2.0,  0.9, 0.00, 'teal'),
    (0.11, 0.040, 3.3,  1.4, 0.55, 'cyan'),
    (0.19, 0.055, 2.6,  1.0, 1.10, 'lteal'),
    (0.28, 0.045, 1.9,  1.7, 1.65, 'blue'),
    (0.37, 0.060, 2.9,  1.2, 2.20, 'pink'),
    (0.44, 0.035, 4.1,  1.8, 2.75, 'lpurp'),
    (0.50, 0.020, 5.0,  2.5, 3.14, 'gold'),   # center hairline
    (0.56, 0.035, 4.1,  1.8, 3.53, 'lpurp'),
    (0.63, 0.060, 2.9,  1.2, 4.08, 'pink'),
    (0.72, 0.045, 1.9,  1.7, 4.63, 'blue'),
    (0.81, 0.055, 2.6,  1.0, 5.18, 'lteal'),
    (0.89, 0.040, 3.3,  1.4, 5.73, 'cyan'),
    (0.96, 0.030, 2.0,  0.9, 6.28, 'teal'),
]
_DENSITY_CHARS = {0: ' ', 1: '░', 2: '▒', 3: '▓', 4: '█'}


def _wave_frame_tui(t: float, frame_i: int, cols: int, rows: int,
                    run_number: int = 0, stats: dict | None = None) -> Text:
    """Render one finale wave frame as a Rich Text (built from ANSI codes).

    Mirror of gen_demo_cast._render_wave_frame but sized to the live terminal
    and populated with actual run stats in the centre banner.
    """
    RS_ = "\033[0m"
    B_  = "\033[1m"
    TWO_PI = 2.0 * math.pi

    stats  = stats or {}
    fade   = min(1.0, frame_i / 4.0)
    tc     = rows // 2
    txt_top, txt_bot = tc - 4, tc + 4

    # Build per-cell arrays: character, ANSI colour code, brightness
    cell_ch = [[' '] * cols for _ in range(rows)]
    cell_cl = [['' ] * cols for _ in range(rows)]
    cell_br = [[0  ] * cols for _ in range(rows)]

    for base_frac, amp_frac, freq, spd, phase, ckey in _WAVE_DEFS:
        base_y = base_frac * rows
        amp    = amp_frac * rows * fade
        color  = _WAVE_PAL[ckey]
        for col in range(cols):
            wave_y = base_y + amp * math.sin(
                freq * col / max(cols, 1) * TWO_PI + t * spd + phase
            )
            iy = int(round(wave_y))
            for dy in range(-3, 4):
                y = iy + dy
                if y < 0 or y >= rows or txt_top <= y <= txt_bot:
                    continue
                dist = abs(dy) + abs(wave_y - iy) * 0.5
                br   = max(0, 4 - int(dist * 1.8))
                if br > cell_br[y][col]:
                    cell_br[y][col] = br
                    cell_ch[y][col] = _DENSITY_CHARS[br]
                    cell_cl[y][col] = color

    # Achievement banner fades in after the first 4 frames
    text_rows: dict[int, str] = {}
    if frame_i >= 4:
        def _ct(s: str, col: str) -> str:
            pad = max(0, (cols - len(s)) // 2)
            return ' ' * pad + B_ + col + s + RS_

        att    = stats.get('attempted',  0)
        comp   = stats.get('compiled',   0)
        fail   = stats.get('failed',     0)
        pts    = stats.get('points',     0)
        new    = stats.get('new_models', 0)
        streak = stats.get('streak',     0)
        rarity = stats.get('rarity',     'COMPLETE')
        p      = _WAVE_PAL
        sep    = '━' * min(56, max(cols - 4, 0))
        text_rows = {
            tc - 3: _ct(sep, p['gold']),
            tc - 2: _ct(f'⚡   EXPEDITION  #{run_number:03d}  COMPLETE   ⚡', p['gold']),
            tc - 1: _ct(f'{att} attempted  ·  {comp} compiled  ·  {fail} failed', p['teal']),
            tc:     _ct(f'+{pts:,} PTS  ·  {new} NEW TO BESTIARY', p['gold']),
            tc + 1: _ct(f'🔥 STREAK ×{streak}  ·  {rarity}  ★ ★ ★', p['pink']),
            tc + 2: _ct(sep, p['gold']),
        }

    # Assemble ANSI string with run-length colour encoding
    parts: list[str] = []
    for y in range(rows):
        if y in text_rows:
            parts.append(text_rows[y])
        else:
            cur_cl = None
            for x in range(cols):
                ch = cell_ch[y][x]
                cl = cell_cl[y][x]
                if cl != cur_cl:
                    parts.append(cl if cl else RS_)
                    cur_cl = cl
                parts.append(ch)
            if cur_cl:
                parts.append(RS_)
        parts.append('\n')

    return Text.from_ansi(''.join(parts))


# ── WaveCanvas widget ─────────────────────────────────────────────────────────

class WaveCanvas(Widget):
    """Lightweight canvas that renders wave frames via render() + refresh().

    Using render() instead of Static.update() avoids triggering a full Textual
    layout recalculation on every frame — keeping animation fluid even on large
    terminals.
    """

    DEFAULT_CSS = "WaveCanvas { width: 100%; height: 100%; overflow: hidden; }"

    def __init__(self) -> None:
        super().__init__()
        self._text: Text = Text("")

    def set_frame(self, text: Text) -> None:
        self._text = text
        self.refresh()

    def render(self) -> Text:
        return self._text


# ── WaveFinaleScreen ───────────────────────────────────────────────────────────

class WaveFinaleScreen(Screen):
    """Full-screen Rave-Tapes wave animation shown between RunScreen and SummaryScreen."""

    CSS = """
    WaveFinaleScreen {
        layout: vertical;
        overflow: hidden;
    }
    """

    # Pre-compute 26 frames but only play the first 8 (≈ 1 s of motion),
    # then freeze on the last frame.  Continuing to render past that point
    # shows visible slowdown from widget state accumulation.  Keypress advances.
    TOTAL_FRAMES   = 26
    _PLAY_FRAMES   = 8

    def __init__(self, num_chips: int, run_number: int, run_state: RunState,
                 auto_quit_secs: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_chips       = num_chips
        self.run_number      = run_number
        self._run_state      = run_state
        self._auto_quit_secs = auto_quit_secs
        self._stats: dict    = {}
        self._finished       = False

    def compose(self) -> ComposeResult:
        yield WaveCanvas()

    def on_mount(self) -> None:
        self._load_stats()
        self._frames: list = []
        self._frame_idx    = 0
        self._finished     = False
        self.run_worker(self._precompute, exclusive=False)

    async def _precompute(self) -> None:
        """Compute all TOTAL_FRAMES wave frames in parallel, then start playback."""
        await asyncio.sleep(0)          # yield once so layout settles and size is known
        cols = self.size.width  or 220
        rows = self.size.height or 50
        loop = asyncio.get_event_loop()
        frames = await asyncio.gather(*[
            loop.run_in_executor(
                None, _wave_frame_tui,
                (i + 1) * 0.30, i + 1, cols, rows, self.run_number, self._stats,
            )
            for i in range(self.TOTAL_FRAMES)
        ])
        self._frames = list(frames)
        self._timer = self.set_interval(0.13, self._tick)

    def _load_stats(self) -> None:
        """Populate the achievement banner stats from RunState (no file I/O)."""
        self._stats = self._run_state.get_stats()

    async def _tick(self) -> None:
        if self._finished:
            return
        if self._frame_idx < len(self._frames):
            try:
                self.query_one(WaveCanvas).set_frame(self._frames[self._frame_idx])
            except Exception:
                pass
        self._frame_idx += 1
        if self._frame_idx >= self._PLAY_FRAMES:
            # Freeze on the last frame — stop the timer, hold the image.
            # Always auto-advance after 2 s; keypress also works immediately.
            try:
                self._timer.stop()
            except Exception:
                pass
            self.set_timer(2.0, self._advance)

    def _advance(self) -> None:
        if self._finished:
            return
        self._finished = True
        self.app.push_screen(SummaryScreen(
            self.num_chips,
            self.run_number,
            self._run_state,
            auto_quit_secs=self._auto_quit_secs,
        ))

    def on_key(self, _event: object) -> None:
        """Any keypress advances past the frozen wave to the SummaryScreen.
        Guard of 3 frames prevents queued RunScreen keypresses from skipping instantly."""
        if self._frame_idx >= 3:
            self._advance()


# ── SummaryScreen ─────────────────────────────────────────────────────────────

class SummaryScreen(Screen):
    """Colorful post-run results with full error text and chip leaderboard."""

    CSS = """
    SummaryScreen {
        layout: vertical;
    }
    #summary-log {
        height: 1fr;
        border-left: solid $primary;
        border-top: solid $primary;
        border-bottom: solid $primary;
        scrollbar-size: 0 0;
    }
    """

    BINDINGS = [
        Binding("r,enter", "rerun",           "Run Again", show=True),
        Binding("q",       "quit",            "Quit",      show=True),
        Binding("b",       "show_bestiary",   "Bestiary",  show=True),
        Binding("up",      "scroll_up",       "",          show=False),
        Binding("down",    "scroll_down",     "",          show=False),
        Binding("pageup",  "scroll_page_up",  "",          show=False),
        Binding("pagedown","scroll_page_down","",          show=False),
    ]

    def __init__(self, num_chips: int, run_number: int, run_state: RunState,
                 auto_quit_secs: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_chips       = num_chips
        self.run_number      = run_number
        self._run_state      = run_state
        self._auto_quit_secs = auto_quit_secs

    def compose(self) -> ComposeResult:
        self.app.title     = f"EXPEDITION #{self.run_number:03d}  COMPLETE"
        self.app.sub_title = "r=Run Again  q=Quit"
        yield Header(show_clock=False)
        yield RichLog(markup=True, highlight=False, auto_scroll=False, id="summary-log")
        yield Footer()

    def on_mount(self) -> None:
        self._populate()
        if self._auto_quit_secs > 0:
            self.set_timer(self._auto_quit_secs, self.app.exit)

    def action_rerun(self) -> None:
        self.app.switch_screen(SetupScreen())

    def action_scroll_up(self)       -> None: self.query_one("#summary-log").scroll_up()
    def action_scroll_down(self)     -> None: self.query_one("#summary-log").scroll_down()
    def action_scroll_page_up(self)  -> None: self.query_one("#summary-log").scroll_page_up()
    def action_scroll_page_down(self)-> None: self.query_one("#summary-log").scroll_page_down()

    def _populate(self) -> None:
        log = self.query_one("#summary-log", RichLog)
        rn  = self.run_number

        # ── Load results from RunState (no file I/O) ─────────────────────────
        chip_results = self._run_state.get_chip_results()   # sorted by -pts
        sq_by_chip   = self._run_state.get_sq_results()

        if not chip_results:
            log.write(
                f"[bold red]No results found for run #{rn:03d}.[/]\n"
                f"[dim]Check /tmp/expedition_results_chip*.csv[/]"
            )
            return

        # ── Derived totals ────────────────────────────────────────────────────
        all_successes  = [r for c in chip_results for r in c["successes"]]
        all_failures   = [r for c in chip_results for r in c["failures"]]
        all_first      = [r for c in chip_results for r in c["first_evers"]]
        all_fv         = [r for c in chip_results for r in c["first_voice"]]
        all_times      = [t for c in chip_results for t in c["times"]]
        total_ns       = len(all_successes)
        total_nf       = len(all_failures)
        total_attempted= total_ns + total_nf
        total_pts      = sum(c["pts"] for c in chip_results)
        rate           = total_ns / max(total_attempted, 1)

        all_sq_s     = [r for sq in sq_by_chip.values() for r in sq["successes"]]
        all_sq_f     = [r for sq in sq_by_chip.values() for r in sq["failures"]]
        total_sq_pts = sum(sq["pts"] for sq in sq_by_chip.values())
        has_sq       = bool(sq_by_chip)

        import datetime
        now_str = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")

        # ── Classification badge ──────────────────────────────────────────────
        if rate >= 0.80:
            classification = "OUTSTANDING"
        elif rate >= 0.60:
            classification = "COMPILED"
        elif rate >= 0.40:
            classification = "PARTIAL"
        else:
            classification = "CRITICAL"

        pts_col  = "gold1" if total_pts > 0 else ("red" if total_pts < 0 else "dim")
        rate_col = "green" if rate > 0.6 else ("yellow" if rate > 0.3 else "red")

        # ── Header ────────────────────────────────────────────────────────────
        log.write(f"[bold cyan]╔{'═' * 62}[/]")
        log.write(f"[bold cyan]║  ██ TENSTORRENT SILICON DIVISION — FIELD REPORT #{rn:03d}[/]")
        log.write(f"[bold cyan]║  ██ CLASSIFICATION: {classification:<12}  DATE: {now_str}[/]")
        log.write(f"[bold cyan]╚{'═' * 62}[/]\n")

        # ── MISSION SUMMARY ───────────────────────────────────────────────────
        bar_w = round(rate * 53)
        rate_bar = (
            f"[{'green' if rate > 0.6 else 'yellow'}]{'█' * bar_w}[/]"
            f"[dim]{'░' * (53 - bar_w)}[/]"
        )
        log.write("[bold cyan]  MISSION SUMMARY[/]")
        log.write(f"[cyan]  {'─' * 57}[/]")
        log.write(
            f"  [dim]TARGETS ACQUIRED:[/]   [bold]{total_ns} / {total_attempted}[/]"
            f"    [dim]SUCCESS RATE:[/] [{rate_col}]{rate:.0%}[/]"
        )
        log.write(
            f"  [dim]POINTS EXTRACTED:[/]  [{pts_col}]{total_pts:+,}[/]"
            f"     [dim]NEW ASSETS:[/]   [bold gold1]{len(all_first)}[/]"
        )
        log.write(f"  {rate_bar}")
        log.write("")

        # ── OPERATIVE STATUS (chip leaderboard with NATO codenames) ───────────
        _CODENAMES = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]
        log.write("[bold cyan]  OPERATIVE STATUS[/]")
        log.write(f"[cyan]  {'─' * 57}[/]")
        for rank, c in enumerate(chip_results):
            codename = _CODENAMES[rank] if rank < 4 else f"CHIP-{rank}"
            ns    = len(c["successes"])
            nf    = len(c["failures"])
            pts   = c["pts"]
            times = c["times"]
            sq    = sq_by_chip.get(c["chip_id"], {})
            sq_ns = len(sq.get("successes", []))
            sq_nf = len(sq.get("failures",  []))
            ratio  = ns / max(ns + nf, 1)
            filled = round(ratio * 12)
            bar    = "█" * filled + "░" * (12 - filled)
            bc     = "green" if ratio > 0.6 else ("yellow" if ratio > 0.3 else "red")
            pc     = "gold1" if pts > 0 else ("red" if pts < 0 else "dim")
            avg_t  = f"{sum(times)/len(times):.0f}s" if times else "—"
            sq_note = f"  [yellow]SQ✓{sq_ns}[/]" if sq_ns else (f"  [dim]SQ✗{sq_nf}[/]" if sq_nf else "")
            log.write(
                f"  [bold]{codename:<7}[/] [yellow][C{c['chip_id']}][/]"
                f"  [{bc}]{bar}[/]"
                f"  [bold]{ns}[/] acquired  [dim]{nf}[/] lost"
                f"  [{pc}]{pts:>+,}[/]  [dim]{avg_t}[/]{sq_note}"
            )
        log.write("")

        # ── NEW INTELLIGENCE (new-to-bestiary entries with first voice / artifact) ──
        _RARITY_COLOR = {
            "legendary": "gold1", "rare": "magenta",
            "uncommon":  "cyan",  "common": "dim",
            "familiar":  "dim",
        }
        if all_first:
            log.write(f"[bold gold1]  ★ NEW INTELLIGENCE ({len(all_first)})[/]")
            log.write(f"[cyan]  {'─' * 57}[/]")
            for r in all_first:
                rar   = r.get("rarity", "common").lower()
                rc    = _RARITY_COLOR.get(rar, "dim")
                tag   = f"[{rc}][{rar.upper()}][/]"
                art   = (r.get("artifact") or "").strip()
                is_fv = r.get("first_voice") == "True"
                label = "INTERCEPT:" if is_fv else "ARTIFACT: "
                log.write(f"  {tag}  [bold]{r['model']}[/]")
                if art:
                    log.write(f"    [dim]{label}[/] [{rc}]{art}[/]")
                log.write("")

        # ── CATALOGUED (other compiled models, compact) ───────────────────────
        shown = {r["model"] for r in all_first}
        other = [r for r in all_successes if r["model"] not in shown]
        if other:
            names = "  ·  ".join(_model_short(r["model"]) for r in other)
            log.write(f"[dim]  CATALOGUED ({len(other)})[/]")
            log.write(f"  [dim]{names}[/]")
            log.write("")

        # ── SIDE QUEST bonus haul (compact one-liner) ─────────────────────────
        if has_sq:
            sq_ns_total = sum(len(sq["successes"]) for sq in sq_by_chip.values())
            sq_nf_total = sum(len(sq["failures"])  for sq in sq_by_chip.values())
            sqc = "gold1" if total_sq_pts > 0 else "dim"
            log.write(
                f"[yellow]  ⚡ BONUS HAUL[/]  ·  "
                f"[dim]{sq_ns_total + sq_nf_total} bonus models[/]  ·  "
                f"[green]{sq_ns_total} compiled[/]  ·  "
                f"[{sqc}]{total_sq_pts:+,} pts[/]"
            )
            log.write("")

        # ── TARGETS AT LARGE (failures) ───────────────────────────────────────
        all_fails_combined = all_failures + all_sq_f
        sq_model_set       = {r["model"] for r in all_sq_f}
        if all_fails_combined:
            from lib.expedition.bestiary import _classify_error
            log.write(f"[bold red]  TARGETS AT LARGE ({len(all_fails_combined)})[/]")
            log.write(f"[red]  {'─' * 57}[/]")
            for r in all_fails_combined:
                err       = (r.get("error") or "").strip()
                key, label, hint = _classify_error(err)
                err_lines = [l for l in err.splitlines() if l.strip()]
                err_short = err_lines[-1].strip()[:80] if err_lines else ""
                sq_tag    = " [yellow][SQ][/]" if r["model"] in sq_model_set else ""
                log.write(
                    f"  [red]{r['model']}[/]{sq_tag}"
                    f"  [bold red]{label.upper()}[/]"
                    + (f"  [dim]{err_short}[/]" if err_short else "")
                )
            log.write("")

        # ── Field journal snippet ─────────────────────────────────────────────
        try:
            from lib.expedition.notes import read_journal
            journal_text = read_journal(rn, project_dir=PROJECT_DIR)
            if journal_text:
                log.write(f"[cyan]  {'─' * 57}[/]")
                log.write("[bold cyan]  📓 FIELD JOURNAL[/]")
                log.write(f"[cyan]  {'─' * 57}[/]")
                for line in journal_text.splitlines()[:20]:
                    if line.startswith("## "):
                        log.write(f"[bold yellow]{line[3:]}[/]")
                    elif line.startswith("> "):
                        log.write(f"[italic]{line[2:]}[/]")
                    else:
                        log.write(line or " ")
                log.write("")
        except Exception:
            pass

        # ── All-time stats + footer ───────────────────────────────────────────
        try:
            from lib.expedition.bestiary import Bestiary
            b      = Bestiary(path=str(PROJECT_DIR / "data" / "bestiary.json"))
            total  = len(b.compiled)
            totals = getattr(b, "chip_totals", {})
            best_chip    = max(totals, key=lambda k: totals[k].get("pts", 0), default=None)
            best_chip_pt = totals[best_chip].get("pts", 0) if best_chip else 0
            best_streak  = max((v.get("best_streak", 0) for v in totals.values()), default=0)
            streak_note  = f"  🔥×{best_streak}" if best_streak >= 2 else ""
            chip_note    = f"  C{best_chip} leads {best_chip_pt:,}pts" if best_chip else ""
            log.write(f"[cyan]  {'─' * 57}[/]")
            log.write(f"  [dim]ALL-TIME:[/]  {total} compiled{streak_note}{chip_note}")
        except Exception:
            pass

        log.write(f"[bold cyan]  {'═' * 57}[/]")
        log.write(
            "[bold green]  ══[/]"
            " [bold]\\[R][/] Run Again"
            "  [bold]\\[Q][/] Quit"
            "  [bold]\\[B][/] Bestiary"
            "  [dim]↑/↓ scroll[/]"
            "  [bold green]══[/]"
        )

    def action_show_bestiary(self) -> None:
        try:
            result = subprocess.run(
                [sys.executable, str(PROJECT_DIR / "expedition.py"), "summary"],
                capture_output=True, text=True, timeout=15,
            )
            output = result.stdout or "(no bestiary output)"
        except Exception as exc:
            output = f"Error: {exc}"
        log = self.query_one("#summary-log", RichLog)
        log.write(f"\n[bold cyan]{'═' * 60}[/]")
        for ln in output.splitlines():
            try:
                log.write(Text.from_ansi(ln))
            except Exception:
                log.write(ln)
        log.scroll_end(animate=False)


# ── Main Application ──────────────────────────────────────────────────────────

class ExpeditionTUI(App[None]):
    """Three-screen Expedition TUI: setup → run → summary.

    All CLI parameters are stored on the app so each Screen can read them
    without needing a full constructor parameter list.
    """

    TITLE = "EXPEDITION"

    def __init__(
        self,
        num_chips:   int,
        run_number:  int,
        arch:        str,
        project_dir: Path,
        # Setup params (forwarded to SetupScreen):
        limit:                  int   = 0,
        seed_only:              bool  = False,
        frontier_only:          bool  = False,
        no_predownload:         bool  = False,
        min_downloads:          int   = 50,
        min_likes:              int   = 1,
        max_dl_like_ratio:      int   = 300,
        max_params_b:           float = 0.0,
        allow_gated:            bool  = False,
        max_cache_gb:           float = 0.0,
        session_download_max:   float = 0.0,
        parallel_downloads:     int   = 4,
        staples:                bool  = False,
        curated:                bool  = False,
        backend:                str   = "auto",
        auto_quit_secs:         int   = 0,
        confirm:                bool  = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_chips    = num_chips
        self.run_number   = run_number
        self.arch         = arch
        self._project_dir = project_dir
        # Setup params (SetupScreen reads these as defaults).
        self.limit                = limit
        self.seed_only            = seed_only
        self.frontier_only        = frontier_only
        self.no_predownload       = no_predownload
        self.min_downloads        = min_downloads
        self.min_likes            = min_likes
        self.max_dl_like_ratio    = max_dl_like_ratio
        self.max_params_b         = max_params_b
        self.allow_gated          = allow_gated
        self.max_cache_gb         = max_cache_gb
        self.session_download_max = session_download_max
        self.parallel_downloads   = parallel_downloads
        self.staples              = staples
        self.curated              = curated
        self.backend              = backend
        self.auto_quit_secs       = auto_quit_secs
        self.confirm              = confirm

    def on_mount(self) -> None:
        # Purge stale forge shared-memory segments and TT device handles from any
        # previous killed run — stale sm_segment.* files cause forge.compile() to
        # hang indefinitely on the next AlexNet (or any forge model) compile.
        import glob as _glob
        for _f in _glob.glob("/dev/shm/sm_segment.tt-quietbox.*.0"):
            try:
                import os as _os; _os.unlink(_f)
            except OSError:
                pass
        for _f in _glob.glob("/dev/shm/tt_device_*_memory"):
            try:
                import os as _os; _os.unlink(_f)
            except OSError:
                pass

        rn = f"Run #{self.run_number:03d}"
        self.title     = f"EXPEDITION  {rn}"
        self.sub_title = f"{self.num_chips} chip(s) · {self.arch.upper()}"
        self.push_screen(SetupScreen())
