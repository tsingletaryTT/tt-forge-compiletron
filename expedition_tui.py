#!/usr/bin/env python3
# expedition_tui.py
"""
Textual TUI front-end for Expedition Mode.

Three screens:

  SetupScreen  — interactive config + live discovery/download log.
                 User adjusts parameters with keyboard, presses Enter
                 to start HF discovery and pre-download, then auto-advances.

  RunScreen    — equal-sized chip panels with streaming worker output,
                 roguelike combat log, hardware sidebar, score strip.

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
import os
import random
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Header, RichLog, Static

PROJECT_DIR = Path(__file__).parent

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


def _strip_osc(line: str) -> str:
    return _RE_OSC.sub("", line)


# ── Status file I/O ───────────────────────────────────────────────────────────

def _read_status(chip_id: int) -> dict[str, str]:
    status_dir = os.environ.get("EXPEDITION_STATUS_DIR", "/tmp")
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
    model     = s.get("model", "")[:22]
    done      = s.get("done", "0") == "1"

    ratio  = min(current / total, 1.0)
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
    ChipPanel > RichLog { height: 1fr; width: 1fr; }
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
        border: solid $accent;
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
            for dev in devices[:4]:
                dev_id = dev.get("board_id",   "?")
                btype  = dev.get("board_type", "?")[:10]
                telem  = dev.get("telemetry", {})
                temp   = telem.get("asic_temperature", "?")
                aiclk  = telem.get("ai_clk",           "?")
                lines.append(
                    f"[bold cyan]Chip {dev_id}[/] {btype}"
                    f"  [yellow]{aiclk}MHz[/]  [red]{temp}°C[/]"
                )
            self.update(Text.from_markup(
                "\n".join(lines) if lines else "[dim]No devices[/]"
            ))
        except FileNotFoundError:
            self.update(Text.from_markup("[dim]tt-smi not found[/]"))
        except Exception as exc:
            self.update(Text.from_markup(f"[dim]tt-smi: {str(exc)[:40]}[/]"))


class CombatLog(RichLog):
    DEFAULT_CSS = """
    CombatLog {
        height: 1fr;
        border: solid $secondary;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(markup=True, highlight=False, auto_scroll=True, **kwargs)
        self.border_title = "  COMBAT LOG  "

    def log_success(self, chip_id: int, model: str, rarity: str,
                    pts: int, first_ever: bool, streak: int) -> None:
        badge  = _RARITY_MARKUP.get(rarity, _RARITY_MARKUP["common"])
        short  = model.split("/")[-1][:22] if model else "unknown"
        first  = "  [bold gold1]★ BESTIARY[/]" if first_ever else ""
        streak_txt = f"  🔥×{streak}" if streak >= 2 else ""
        self.write(
            f"[bold green]⚔[/] [yellow]C{chip_id}[/] {badge}"
            f"  [cyan]{short}[/]  [gold1]+{pts}pts[/]{streak_txt}{first}"
        )

    def log_failure(self, chip_id: int, model: str) -> None:
        short  = model.split("/")[-1][:22] if model else "unknown"
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
        border: solid $primary;
        padding: 0 1;
    }
    #setup-log {
        width: 1fr;
        height: 1fr;
        border: solid $secondary;
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
        self._min_downloads= 0
        self._min_likes    = 0
        self._max_params_b = 0.0
        self._allow_gated  = False
        self._max_cache_gb = 0.0
        self._session_download_max = 0.0
        self._parallel_downloads   = 4
        self._running      = False  # True while discovery is in progress
        self._setup_done   = False  # True once queues are built (no re-run)

    def on_mount(self) -> None:
        app = self.app
        self._chips         = app.num_chips
        self._limit         = app.limit
        self._seed_only     = app.seed_only
        self._frontier_only = app.frontier_only
        self._no_predownload= app.no_predownload
        self._min_downloads = app.min_downloads
        self._min_likes     = app.min_likes
        self._max_params_b  = app.max_params_b
        self._allow_gated   = app.allow_gated
        self._max_cache_gb  = app.max_cache_gb
        self._session_download_max = app.session_download_max
        self._parallel_downloads   = app.parallel_downloads
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

        status = (
            "[bold yellow]● Ready — press ENTER[/]"
            if not self._running else
            "[bold cyan]⚙ Discovering…[/]"
        )

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
            f"  Max Params   [bold]{pb_str}[/]  [dim]m / n[/]",
            f"  Sources      [bold]{src_str}[/]  [dim]1/2/3[/]",
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
            if not self._running:
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

    def action_start(self) -> None:
        if self._running or self._setup_done:
            return
        self._running = True
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
            BESTIARY_PATH,
        )
        from lib.expedition.bestiary import Bestiary

        bestiary     = Bestiary(path=str(BESTIARY_PATH))
        compiled_ids = set(bestiary.compiled.keys())

        seed_items:     list[dict] = []
        frontier_items: list[dict] = []

        # ── Seed scan (forge-models library) ─────────────────────────────────
        if not self._frontier_only:
            _log("[cyan]⚙ Scanning tt-forge-models library...[/]")
            seed_items = _scan_forge_models(compiled_ids)
            _log(f"[green]✓ {len(seed_items)} seed model(s) found[/]")
            for item in seed_items:
                mid  = item.get("model_id", "?")
                task = item.get("task") or item.get("source") or ""
                _log(f"  [dim]· {mid}  {task}[/]")

        forge_ids = {item["model_id"] for item in seed_items}

        # ── HF frontier discovery (slow — network) ────────────────────────────
        if not self._seed_only:
            _log("[cyan]⚙ Querying HuggingFace frontier (may take 30-60s)...[/]")
            frontier_items = _scan_frontier(
                compiled_ids,
                forge_ids,
                min_downloads = self._min_downloads,
                min_likes     = self._min_likes,
                max_params_b  = self._max_params_b,
                skip_gated    = not self._allow_gated,
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
        # post_message is not thread-safe in Textual 7.x; use call_from_thread
        # to run the screen transition directly on the event loop.
        app.call_from_thread(self._advance_to_run, chip_queues)

    def _advance_to_run(self, chip_queues: list[list[dict]]) -> None:
        """Called on the event loop thread when setup completes."""
        self._running    = False
        self._setup_done = True
        self.app.push_screen(
            RunScreen(
                chip_queues  = chip_queues,
                num_chips    = self._chips,
                run_number   = self.app.run_number,
                arch         = self.app.arch,
                project_dir  = self.app._project_dir,
            )
        )


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
    """

    BINDINGS = [
        Binding("b", "show_bestiary", "Bestiary",   show=True),
        Binding("t", "show_toplike",  "tt-toplike", show=True),
        Binding("q", "quit",          "Quit",       show=True),
    ]

    def __init__(self, chip_queues: list[list[dict]], num_chips: int,
                 run_number: int, arch: str, project_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self.chip_queues  = chip_queues
        self.num_chips    = num_chips
        self.run_number   = run_number
        self.arch         = arch
        self._project_dir = project_dir
        self._chip_rarity: list[str] = ["common"] * 4
        self._chip_streak: list[int] = [0] * 4
        self._chip_best:   list[int] = [0] * 4
        self._done_count  = 0

    def compose(self) -> ComposeResult:
        rn = f"Run #{self.run_number:03d}"
        yield Header(show_clock=True)
        self.app.title     = f"EXPEDITION  {rn}"
        self.app.sub_title = f"{self.num_chips} chip(s) · {self.arch.upper()}"

        with Horizontal(id="main"):
            with Vertical(id="chip-grid"):
                with Horizontal(id="chip-row-top"):
                    if self.num_chips >= 1:
                        yield ChipPanel(0, f"⚔ CHIP 0  {_ADVENTURER_TITLES[0]}", id="chip-0")
                    if self.num_chips >= 2:
                        yield ChipPanel(1, f"⚔ CHIP 1  {_ADVENTURER_TITLES[1]}", id="chip-1")
                if self.num_chips >= 3:
                    with Horizontal(id="chip-row-bottom"):
                        yield ChipPanel(2, f"⚔ CHIP 2  {_ADVENTURER_TITLES[2]}", id="chip-2")
                        if self.num_chips >= 4:
                            yield ChipPanel(3, f"⚔ CHIP 3  {_ADVENTURER_TITLES[3]}", id="chip-3")
            with Vertical(id="sidebar"):
                yield HardwareWidget(id="hw")
                yield CombatLog(id="combat-log")

        yield ScoreStrip(self.num_chips, id="score-strip")
        yield Footer()

    def on_mount(self) -> None:
        for chip_id in range(self.num_chips):
            self._launch_chip(chip_id)

    @work
    async def _launch_chip(self, chip_id: int) -> None:
        await asyncio.sleep(chip_id * 4)

        env = {
            **os.environ,
            "TT_VISIBLE_DEVICES":      str(chip_id),
            "TT_METAL_ARCH_NAME":      self.arch,
            "TT_METAL_LOGGER_LEVEL":   "FATAL",
            "TT_MESH_GRAPH_DESC_PATH": str(
                self._project_dir / "mesh_graph_descriptors"
                / "p100_mesh_graph_descriptor.textproto"
            ),
            "PYTHONUNBUFFERED":        "1",
        }

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(self._project_dir / "lib" / "expedition" / "expedition_worker.py"),
            "--chip",     str(chip_id),
            "--run",      str(self.run_number),
            "--bestiary", str(self._project_dir / "data" / "bestiary.json"),
            "--queue",    f"/tmp/expedition_queue_chip{chip_id}.json",
            "--results",  f"/tmp/expedition_results_chip{chip_id}.csv",
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )

        panel = self.query_one(f"#chip-{chip_id}", ChipPanel)

        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace")
            panel.write_line(line)
            self._parse_for_events(chip_id, line)

        await proc.wait()
        panel.mark_done(proc.returncode == 0)
        status = _read_status(chip_id)
        pts    = int(status.get("pts", 0))
        try:
            combat = self.query_one("#combat-log", CombatLog)
            combat.log_chip_done(chip_id, pts, self._chip_best[chip_id])
        except Exception:
            pass

        self._done_count += 1
        if self._done_count >= self.num_chips:
            await asyncio.sleep(1.5)   # brief pause so user sees final state
            self.app.push_screen(SummaryScreen(self.num_chips, self.run_number))

    def _parse_for_events(self, chip_id: int, line: str) -> None:
        rm = _RE_RARITY.search(line)
        if rm:
            raw = rm.group(1).upper()
            if "LEGENDARY" in raw:
                self._chip_rarity[chip_id] = "legendary"
            elif "RARE" in raw:
                self._chip_rarity[chip_id] = "rare"
            elif "UNCOMMON" in raw:
                self._chip_rarity[chip_id] = "uncommon"
            else:
                self._chip_rarity[chip_id] = "common"

        try:
            combat = self.query_one("#combat-log", CombatLog)
        except Exception:
            return

        if _RE_SUCCESS.search(line):
            pts_m  = _RE_PTS.search(line)
            pts    = abs(int(pts_m.group(1))) if pts_m else 0
            str_m  = _RE_STREAK.search(line)
            streak = int(str_m.group(1)) if str_m else 0
            first  = bool(_RE_FIRST.search(line))
            self._chip_streak[chip_id] = streak
            self._chip_best[chip_id]   = max(self._chip_best[chip_id], streak)
            model  = _read_status(chip_id).get("model", "")
            combat.log_success(chip_id, model, self._chip_rarity[chip_id],
                               pts, first, streak)
            self._chip_rarity[chip_id] = "common"

        elif _RE_FAILURE.search(line):
            self._chip_streak[chip_id] = 0
            model = _read_status(chip_id).get("model", "")
            combat.log_failure(chip_id, model)
            self._chip_rarity[chip_id] = "common"

    def action_show_toplike(self) -> None:
        if not shutil.which("tt-toplike"):
            try:
                combat = self.query_one("#combat-log", CombatLog)
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
            combat = self.query_one("#combat-log", CombatLog)
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


# ── SummaryScreen ─────────────────────────────────────────────────────────────

class SummaryScreen(Screen):
    """Colorful post-run results with full error text and chip leaderboard."""

    CSS = """
    SummaryScreen {
        layout: vertical;
    }
    #summary-log {
        height: 1fr;
        border: solid $primary;
    }
    """

    BINDINGS = [
        Binding("q",     "quit",           "Quit",    show=True),
        Binding("b",     "show_bestiary",   "Bestiary",show=True),
        Binding("up",    "scroll_up",       "",        show=False),
        Binding("down",  "scroll_down",     "",        show=False),
        Binding("pageup","scroll_page_up",  "",        show=False),
        Binding("pagedown","scroll_page_down","",      show=False),
    ]

    def __init__(self, num_chips: int, run_number: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_chips   = num_chips
        self.run_number  = run_number

    def compose(self) -> ComposeResult:
        self.app.title     = f"EXPEDITION #{self.run_number:03d}  COMPLETE"
        self.app.sub_title = "q to quit"
        yield Header(show_clock=False)
        yield RichLog(markup=True, highlight=False, auto_scroll=False, id="summary-log")
        yield Footer()

    def on_mount(self) -> None:
        self._populate()

    def action_scroll_up(self)       -> None: self.query_one("#summary-log").scroll_up()
    def action_scroll_down(self)     -> None: self.query_one("#summary-log").scroll_down()
    def action_scroll_page_up(self)  -> None: self.query_one("#summary-log").scroll_page_up()
    def action_scroll_page_down(self)-> None: self.query_one("#summary-log").scroll_page_down()

    def _populate(self) -> None:
        log = self.query_one("#summary-log", RichLog)
        rn  = self.run_number

        # ── Load results ──────────────────────────────────────────────────────
        chip_results: list[dict] = []
        for chip_id in range(self.num_chips):
            path = Path(f"/tmp/expedition_results_chip{chip_id}.csv")
            if not path.exists():
                continue
            try:
                rows = list(csv.DictReader(path.open()))
            except Exception:
                continue
            successes   = [r for r in rows if r.get("status") == "success"]
            failures    = [r for r in rows if r.get("status") == "failed"]
            total_pts   = sum(int(r.get("pts") or 0) for r in rows)
            first_evers = [r for r in successes if r.get("first_ever") == "True"]
            chip_results.append({
                "chip_id":    chip_id,
                "pts":        total_pts,
                "successes":  successes,
                "failures":   failures,
                "first_evers":first_evers,
            })

        chip_results.sort(key=lambda x: -x["pts"])

        if not chip_results:
            log.write(
                f"[bold red]No results found for run #{rn:03d}.[/]\n"
                f"[dim]Check /tmp/expedition_results_chip*.csv[/]"
            )
            return

        # ── Header ────────────────────────────────────────────────────────────
        log.write(f"[bold cyan]{'═' * 60}[/]")
        log.write(f"[bold cyan]  ⚡ EXPEDITION #{rn:03d} COMPLETE[/]")
        log.write(f"[bold cyan]{'═' * 60}[/]\n")

        # ── Chip leaderboard ──────────────────────────────────────────────────
        medals = ["[bold yellow]🥇[/]", "[bold white]🥈[/]",
                  "[bold orange3]🥉[/]", "  "]
        for i, c in enumerate(chip_results):
            medal    = medals[min(i, 3)]
            ns       = len(c["successes"])
            nf       = len(c["failures"])
            fe       = len(c["first_evers"])
            pts      = c["pts"]
            pts_col  = "gold1" if pts > 0 else ("red" if pts < 0 else "dim")
            ratio    = ns / max(ns + nf, 1)
            filled   = round(ratio * 16)
            bar      = "█" * filled + "░" * (16 - filled)
            bar_col  = "green" if ratio > 0.6 else ("yellow" if ratio > 0.3 else "red")
            fe_str   = f"  [bold gold1]★{fe} first-ever(s)[/]" if fe else ""
            log.write(
                f"{medal} [yellow]CHIP {c['chip_id']}[/]"
                f"  [{bar_col}]{bar}[/]"
                f"  [green]✓{ns}[/]/[red]✗{nf}[/]"
                f"  [{pts_col}]{pts:>+,}pts[/]{fe_str}"
            )

        # ── New bestiary entries ──────────────────────────────────────────────
        all_first = [r for c in chip_results for r in c["first_evers"]]
        if all_first:
            log.write(f"\n[bold cyan]{'─' * 60}[/]")
            log.write(f"[bold cyan]  ★ NEW TO BESTIARY ({len(all_first)})[/]")
            log.write(f"[bold cyan]{'─' * 60}[/]")
            for r in all_first:
                artifact = (r.get("artifact") or "").strip()
                log.write(
                    f"[bold gold1]★[/] [cyan]{r['model']}[/]"
                    + (f"\n    [dim]{artifact}[/]" if artifact else "")
                )

        # ── Failures ──────────────────────────────────────────────────────────
        all_failures = [r for c in chip_results for r in c["failures"]]
        if all_failures:
            from lib.expedition.bestiary import _classify_error
            log.write(f"\n[bold red]{'─' * 60}[/]")
            log.write(f"[bold red]  ✗ FAILED THIS RUN ({len(all_failures)})[/]")
            log.write(f"[bold red]{'─' * 60}[/]")
            for r in all_failures:
                err = (r.get("error") or "").strip()
                key, label, hint = _classify_error(err)
                log.write(
                    f"[bold red]✗[/] [white]{r['model']}[/]"
                    f"  [dim]\\[{label}][/]"
                )
                if err:
                    # Show full error, indented, wrapped naturally by RichLog.
                    for eline in err.splitlines():
                        log.write(f"    [dim]{eline}[/]")
                log.write(f"  [dim italic]{hint}[/]\n")

        # ── Bestiary total ────────────────────────────────────────────────────
        try:
            from lib.expedition.bestiary import Bestiary
            b = Bestiary(path=str(PROJECT_DIR / "data" / "bestiary.json"))
            total = len(b.compiled)
            log.write(f"[bold cyan]{'─' * 60}[/]")
            log.write(f"[bold cyan]  BESTIARY: {total} total compiled[/]")
            log.write(f"[bold cyan]{'═' * 60}[/]")
        except Exception:
            pass

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
        min_downloads:          int   = 0,
        min_likes:              int   = 0,
        max_params_b:           float = 0.0,
        allow_gated:            bool  = False,
        max_cache_gb:           float = 0.0,
        session_download_max:   float = 0.0,
        parallel_downloads:     int   = 4,
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
        self.max_params_b         = max_params_b
        self.allow_gated          = allow_gated
        self.max_cache_gb         = max_cache_gb
        self.session_download_max = session_download_max
        self.parallel_downloads   = parallel_downloads

    def on_mount(self) -> None:
        rn = f"Run #{self.run_number:03d}"
        self.title     = f"EXPEDITION  {rn}"
        self.sub_title = f"{self.num_chips} chip(s) · {self.arch.upper()}"
        self.push_screen(SetupScreen())
