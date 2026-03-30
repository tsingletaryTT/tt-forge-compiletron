#!/usr/bin/env python3
"""
Worker process for single-chip Forge compilation.
Adapted from tt-forge-creative-demos/forge_worker.py with improvements.
"""

import os
import sys
import time
import random
import signal
import warnings
from pathlib import Path
from typing import Tuple, Optional

# Suppress TensorFlow/XLA warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'       # Suppress TF C++ warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'       # Disable oneDNN (suppresses related warnings)

# Suppress Python warnings aggressively
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*in-place operator.*')
warnings.filterwarnings('ignore', message='.*num_batches_tracked.*')


class FilteredStderr:
    """
    Filter stderr to suppress noisy TF/XLA/CUDA/TVM/ABSL warnings.

    These warnings come from C extensions that write directly to stderr
    before Python's logging system can intercept them. Common sources:
    - XLA: cuFFT/cuDNN/cuBLAS factory registration (harmless duplicate registrations)
    - ABSL: 'All log messages before absl::InitializeLog' preamble
    - computation_placer: 'already registered' (harmless in multi-framework environments)
    - TVM: in-place operator warnings, num_batches_tracked missing
    - Forge: ConstEval debug spam, WARNING/DEBUG level chatter

    Also deduplicates repeated warnings to prevent log flooding.
    """
    def __init__(self, stream):
        self.stream = stream
        self.suppress_patterns = [
            'Unable to register cu',              # cuFFT, cuDNN, cuBLAS duplicate registrations
            'computation placer already registered',
            'All log messages before absl::InitializeLog',
            'In-place operator',                  # TVM in-place operator warnings
            'not found in convert_map',           # TVM missing ops (informational only)
            'Falling back to out-of-place',       # TVM fallback (harmless)
            'num_batches_tracked not found',      # BatchNorm parameter not in Forge params
            'not found in self._parameters',      # Same as above, different message form
            'ConstEval graph:',                   # Forge ConstEval debug spam
            'WARNING',                            # any loguru/logging WARNING line
            '| warning',                          # tt-metal/TTNN lowercase loguru warning
            'warning |',                          # same, less strict leading context
            'DEBUG',                              # any loguru/logging DEBUG line
            '| debug',                            # tt-metal/TTNN lowercase loguru debug
            'Always | ',                          # loguru ALWAYS level (device init noise)
            'E Device',                           # TT device enumeration noise
        ]
        self.seen_warnings = set()
        self.max_seen = 1000  # Cap to prevent memory leak on very long runs

    def write(self, text):
        # Drop lines matching any suppressed pattern
        if any(pattern in text for pattern in self.suppress_patterns):
            return

        # Deduplicate repeated WARNING lines
        text_stripped = text.strip()
        if text_stripped and 'WARNING' in text_stripped:
            if text_stripped in self.seen_warnings:
                return
            if len(self.seen_warnings) < self.max_seen:
                self.seen_warnings.add(text_stripped)

        self.stream.write(text)

    def flush(self):
        self.stream.flush()

    def close(self):
        pass  # Don't close the underlying stream; required for atexit/logging shutdown

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()


# Terminal colors — full Tenstorrent brand palette
GREEN  = '\033[92m'
RED    = '\033[91m'
CYAN   = '\033[96m'
YELLOW = '\033[93m'
BOLD   = '\033[1m'
RESET  = '\033[0m'
PURPLE = '\033[95m'
BLUE   = '\033[94m'
PINK   = '\033[95m'   # same escape as PURPLE

# Rotating per-model color palette (matches original demo)
TT_COLORS = [PURPLE, CYAN, RED, PINK, BLUE]

# Rotating figlet fonts — long names (>25 chars) always use 'small'
CELEBRATION_FONTS = ['small', 'standard', 'slant']


def print_header(chip_id: int, total_models: int):
    """Print demo header — copied directly from demo_compilation_chunked.py"""
    import pyfiglet
    print(f"\n{BOLD}{GREEN}")
    banner = pyfiglet.figlet_format("TT-FORGE", font="banner3")
    print(banner, end='')
    print(f"{RESET}")
    print(f"{BOLD}{CYAN}{'='*80}{RESET}")
    print(f"{BOLD}{CYAN}   COMPILATION SHOWCASE - TENSTORRENT BLACKHOLE{RESET}")
    print(f"{BOLD}{CYAN}   CHIP {chip_id}{RESET}")
    print(f"{BOLD}{CYAN}{'='*80}{RESET}\n")
    print(f"{YELLOW}Hardware:{RESET} 4x P300C Blackhole chips")
    print(f"{YELLOW}Compiler:{RESET} TT-Forge-ONNX (TVM-based MLIR pipeline)")
    print(f"{YELLOW}Models:{RESET} {total_models} in this segment")
    print(f"{YELLOW}Process:{RESET} PyTorch → TVM → Forge IR → MLIR → TT Binary")
    print(f"{BOLD}{CYAN}{'='*80}{RESET}\n")


def generate_fun_prediction(output, display_name, family):
    """
    Generate a factual description of model inference output.
    Copied directly from demo_compilation_chunked.py.
    """
    try:
        if hasattr(output, 'shape') and len(output.shape) >= 2:
            batch_size = output.shape[0]
            num_classes = output.shape[1]

            hash_seed = sum(ord(c) for c in display_name) % 1000
            random.seed(hash_seed)

            messages = [
                f"Inference produced {num_classes}-class distribution (batch size: {batch_size})",
                f"Generated output tensor: {num_classes} classes across {batch_size} sample(s)",
                f"Successfully computed {num_classes}-dimensional output for {batch_size} input(s)",
                f"Model output: {batch_size}×{num_classes} probability distribution",
                f"Inference complete: {num_classes} class predictions generated",
                f"Forward pass produced {num_classes}-class logits (batch: {batch_size})",
            ]

            message = random.choice(messages)
            random.seed()
            return message
        return None
    except Exception:
        random.seed()
        return None


def print_victory_celebration(chip_id: int, successes: int, total: int, successful_models: list):
    """
    Print colorful COMPLETE! celebration organized by model family.
    Copied directly from demo_compilation_chunked.py.
    """
    if successes == 0:
        return

    import pyfiglet
    print(f"\n\n{BOLD}{GREEN}")
    try:
        victory = pyfiglet.figlet_format("COMPLETE!", font="banner3")
        print(victory, end='')
    except Exception:
        print("\n    COMPLETE!\n")
    print(f"{RESET}")

    print(f"{BOLD}{CYAN}   {successes}/{total} MODELS COMPILED SUCCESSFULLY{RESET}\n")

    families: dict = {}
    for model_name in successful_models:
        family = model_name.split('-')[0].lower()
        if family not in families:
            families[family] = []
        families[family].append(model_name)

    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║                    MODELS BY ARCHITECTURE FAMILY                     ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════════╝{RESET}\n")

    sorted_families = sorted(families.items(), key=lambda x: len(x[1]), reverse=True)
    color_rotation = [GREEN, YELLOW, BLUE, PURPLE, PINK, CYAN, RED]
    for idx, (family, models) in enumerate(sorted_families):
        color = color_rotation[idx % len(color_rotation)]
        family_display = family.replace('_', ' ').title()
        print(f"{BOLD}{color}{family_display} Family{RESET} ({len(models)} models)")
        for i in range(0, len(models), 3):
            batch = models[i:i+3]
            names = [m.replace(f"{family}-", "").replace("_", " ") for m in batch]
            print(f"   • " + " • ".join(names))
        print()

    print(f"{BOLD}{GREEN}╔══════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{GREEN}║                         MISSION COMPLETE                             ║{RESET}")
    print(f"{BOLD}{GREEN}║{RESET}     {YELLOW}{successes} models compiled and optimized for Blackhole{RESET}           {BOLD}{GREEN}║{RESET}")
    print(f"{BOLD}{GREEN}║{RESET}     {CYAN}Ready for inference on Tenstorrent hardware{RESET}                {BOLD}{GREEN}║{RESET}")
    print(f"{BOLD}{GREEN}╚══════════════════════════════════════════════════════════════════════╝{RESET}\n")
    print(f"{BOLD}{PURPLE}         TENSTORRENT AI ACCELERATION{RESET}\n")


class TimeoutException(Exception):
    """Exception raised when operation times out"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeouts"""
    raise TimeoutException("Operation timed out")


def import_forge_quietly():
    """
    Import forge while suppressing XLA/CUDA/ABSL noise.

    XLA/ABSL write registration warnings directly to OS fd 2 (not Python's sys.stderr)
    so Python-level filters can't catch them. We temporarily redirect fd 2 to /dev/null
    for the duration of the import, then restore it.

    Returns the forge module.
    Raises ImportError if forge is not available.
    """
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        sys.path.insert(0, os.path.expanduser("~/tt-forge-fe"))
        import forge
        return forge
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


def compile_and_run(model_spec: Tuple, chip_id: int = 0, font_idx: int = 1) -> Tuple[bool, float]:
    """
    Compile model and run inference with timeout and retry logic.

    Args:
        model_spec: Model tuple (display_name, family, loader, input_shape, notes, metadata)
        chip_id: Chip ID for logging
        font_idx: 1-based model counter; drives font and color rotation

    Returns:
        (success, compile_time) tuple
    """
    import torch
    import pyfiglet

    display_name, family, model_loader, input_shape, notes, metadata = model_spec

    # Per-model font and color rotation — matches original demo_compilation_chunked.py
    font  = CELEBRATION_FONTS[font_idx % len(CELEBRATION_FONTS)]
    color = TT_COLORS[font_idx % len(TT_COLORS)]
    # Long names overflow wider fonts — force compact
    if len(display_name) > 25:
        font = 'small'

    # Print model banner with rotating font and brand color
    print(f"\n{BOLD}{color}")
    try:
        banner = pyfiglet.figlet_format(display_name, font=font)
        print(banner, end='')
    except Exception:
        # Try next font in rotation before falling back to plain text
        try:
            fallback_font = CELEBRATION_FONTS[(font_idx + 1) % len(CELEBRATION_FONTS)]
            banner = pyfiglet.figlet_format(display_name, font=fallback_font)
            print(banner, end='')
        except Exception:
            print(f"\n  {display_name}\n")
    print(f"{RESET}")

    print(f"{color}  ★ ═══ {display_name} ═══ ★{RESET}")
    time.sleep(3)  # Celebration pause — let the banner breathe before compilation

    print(f"{BOLD}{BLUE}{'─'*80}{RESET}")
    print(f"  {CYAN}Family:{RESET} {family} | {CYAN}Input:{RESET} {input_shape}")

    start_time = time.time()

    try:
        # forge and logging are already configured in run_worker() before this
        # call, so we just grab the cached module here.
        import forge

        # Load model
        print(f"  {YELLOW}[1/3]{RESET} Loading model architecture...")
        model = model_loader()
        model.eval()

        # Create input
        sample_input = torch.randn(*input_shape)

        # Compile
        print(f"  {YELLOW}[2/3]{RESET} Compiling for TT hardware...")
        compile_start = time.time()
        compiled_model = forge.compile(model, sample_inputs=[sample_input])
        compile_time = time.time() - compile_start

        # Run inference with timeout and retry
        print(f"  {YELLOW}[3/3]{RESET} Running inference on TT device {chip_id}...")

        max_retries = 3
        timeout_seconds = 90
        retry_delay = 1

        output = None
        for attempt in range(1, max_retries + 1):
            try:
                # Set timeout alarm
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)

                if attempt > 1:
                    print(f"    {YELLOW}Retry {attempt}/{max_retries} (waiting {retry_delay}s)...{RESET}")
                    time.sleep(retry_delay)

                # Redirect fd 2 during inference to suppress TTNN C++-level
                # op warnings (op_slicing, conv2d, DRAM layout) that write
                # directly to the OS file descriptor, bypassing FilteredStderr.
                _saved = os.dup(2)
                _null = os.open(os.devnull, os.O_WRONLY)
                os.dup2(_null, 2)
                os.close(_null)
                try:
                    output = compiled_model(sample_input)
                finally:
                    os.dup2(_saved, 2)
                    os.close(_saved)

                # Cancel alarm on success
                signal.alarm(0)
                break

            except TimeoutException:
                signal.alarm(0)
                print(f"    {YELLOW}⚠ Timeout after {timeout_seconds}s{RESET}")

                if attempt < max_retries:
                    retry_delay *= 2
                else:
                    raise TimeoutException(f"Inference timed out after {max_retries} attempts")

            except Exception as e:
                signal.alarm(0)
                raise

        # Handle output
        if isinstance(output, list):
            output = output[0] if output else None

        if output is None:
            raise ValueError("No output from inference")

        total_time = time.time() - start_time

        fun_output = generate_fun_prediction(output, display_name, family)

        print(f"  {BOLD}{GREEN}✓ SUCCESS{RESET}")
        print(f"    Compilation: {compile_time:.2f}s | Total: {total_time:.2f}s")
        print(f"    Output: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        if fun_output:
            print(f"    {CYAN}🎯 {fun_output}{RESET}")

        return True, compile_time

    except TimeoutException as e:
        error_time = time.time() - start_time
        print(f"  {BOLD}{RED}✗ FAILED (TIMEOUT){RESET}")
        print(f"    Error: {str(e)}")
        print(f"    Time: {error_time:.2f}s")
        return False, error_time

    except Exception as e:
        error_time = time.time() - start_time
        print(f"  {BOLD}{RED}✗ FAILED{RESET}")
        print(f"    Error: {type(e).__name__}")
        print(f"    Time: {error_time:.2f}s")
        return False, error_time


def run_worker(chip_id: int, model_indices: list, results_file: Optional[Path] = None):
    """
    Run worker process for given chip.

    Args:
        chip_id: Chip ID (0-based)
        model_indices: List of model indices to compile on this chip
        results_file: Optional CSV file to save results
    """
    # Install FilteredStderr immediately — BEFORE importing torch/forge so
    # CUDA init noise, XLA registrations, and loguru chatter are caught from
    # the very first byte.  Matches forge_worker.py from tt-forge-creative-demos
    # which calls redirect_output_to_log() as the first thing in main().
    if not isinstance(sys.stderr, FilteredStderr):
        sys.stderr = FilteredStderr(sys.stderr)

    # Configure Python logging to CRITICAL for TVM/Forge before any imports
    import logging
    logging.getLogger('tvm.relay.frontend.pytorch').setLevel(logging.CRITICAL)
    logging.getLogger('tvm.relay').setLevel(logging.CRITICAL)
    logging.getLogger('tvm').setLevel(logging.CRITICAL)
    logging.getLogger('forge.tensor').setLevel(logging.INFO)
    logging.getLogger('forge').setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    # Import forge once here (quiet fd-level redirect) so forge module is
    # cached before the model loop — subsequent compile_and_run() calls use
    # the cache and don't trigger fresh XLA/ABSL noise.
    try:
        import_forge_quietly()
    except ImportError:
        pass  # will fail gracefully inside compile_and_run with a clear error

    from lib.models import MODEL_LIST

    # Stagger startup to avoid race conditions
    startup_delay = random.uniform(0, 5)
    print(f"[Chip {chip_id}] Staggered startup delay: {startup_delay:.2f}s")
    time.sleep(startup_delay)

    print(f"[Chip {chip_id}] TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES')}")
    print()

    # Big TT-FORGE startup banner — matches original demo
    print_header(chip_id, len(model_indices))

    # Compile each model
    successes = 0
    failures = 0
    results = []
    total = len(model_indices)

    for idx, model_idx in enumerate(model_indices, 1):
        if model_idx >= len(MODEL_LIST):
            print(f"[Chip {chip_id}] WARNING: Model index {model_idx} out of range")
            continue

        model_spec = MODEL_LIST[model_idx]
        display_name = model_spec[0]

        # Milestone banner every 5 models starting at 6 — matches original
        if idx > 1 and idx % 5 == 1:
            import pyfiglet
            try:
                milestone_banner = pyfiglet.figlet_format(f"Model #{idx}", font="banner")
                print(f"\n{BOLD}{CYAN}{milestone_banner}{RESET}")
            except Exception:
                print(f"\n{BOLD}{CYAN}  ── Model #{idx} ──{RESET}\n")

        # ASCII progress bar — matches original demo_compilation_chunked.py behavior
        bar_length = 40
        filled = int(bar_length * (idx - 1) / total) if total else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        pct = int(100 * (idx - 1) / total) if total else 0
        stats = f"{GREEN}✓{successes}{RESET}/{RED}✗{failures}{RESET}"
        print(f"\n{BOLD}[{bar}] {pct}% ({idx-1}/{total}) {stats}{RESET}")

        # Write machine-readable status for the bottom status pane
        try:
            with open(f'/tmp/compiletron_chip_{chip_id}.status', 'w') as _sf:
                _sf.write(f"chip_id={chip_id}\n")
                _sf.write(f"current={idx - 1}\n")
                _sf.write(f"total={total}\n")
                _sf.write(f"successes={successes}\n")
                _sf.write(f"failures={failures}\n")
                _sf.write(f"model={display_name}\n")
                _sf.write(f"done=0\n")
        except OSError:
            pass

        print(f"\n[Chip {chip_id}] Model {idx}/{total}: {display_name}")

        try:
            success, compile_time = compile_and_run(model_spec, chip_id, font_idx=idx)
            if success:
                successes += 1
            else:
                failures += 1

            results.append({
                'chip_id': chip_id,
                'model': display_name,
                'success': success,
                'compile_time': compile_time,
            })

        except Exception as e:
            print(f"[Chip {chip_id}] ERROR: {e}")
            failures += 1
            results.append({
                'chip_id': chip_id,
                'model': display_name,
                'success': False,
                'compile_time': 0.0,
            })

        # Brief pause between models for readability — matches original
        if idx < total:
            time.sleep(0.3)

    # Victory celebration banner + family breakdown (from original demo)
    successful_models = [r['model'] for r in results if r['success']]
    print_victory_celebration(chip_id, successes, total, successful_models)

    # Per-model ✓/✗ checklist
    print(f"{BOLD}MODELS TESTED ON CHIP {chip_id}:{RESET}\n")
    for r in results:
        mark = "✓" if r['success'] else "✗"
        color = GREEN if r['success'] else RED
        print(f"  {color}{mark}{RESET} {r['model']}")
    print()
    print(f"{BOLD}{CYAN}{'='*80}{RESET}")
    print(f"{CYAN}Check stats pane for overall results{RESET}")
    print(f"{BOLD}{CYAN}{'='*80}{RESET}")

    # Save results if requested
    if results_file:
        import csv
        results_file.parent.mkdir(parents=True, exist_ok=True)

        # Append to CSV
        write_header = not results_file.exists()
        with open(results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['chip_id', 'model', 'success', 'compile_time'])
            if write_header:
                writer.writeheader()
            writer.writerows(results)

    # Update status file to mark this chip done
    try:
        with open(f'/tmp/compiletron_chip_{chip_id}.status', 'w') as _sf:
            _sf.write(f"chip_id={chip_id}\n")
            _sf.write(f"current={total}\n")
            _sf.write(f"total={total}\n")
            _sf.write(f"successes={successes}\n")
            _sf.write(f"failures={failures}\n")
            _sf.write(f"model=DONE\n")
            _sf.write(f"done=1\n")
    except OSError:
        pass

    # Stay alive so the tmux pane keeps showing the checklist.
    # The shell wrapper can omit its own "Press Enter to close" prompt
    # because this loop handles it. Ctrl-C or tmux kill exits cleanly.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    return 0 if failures == 0 else 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='TT-Forge single-chip compilation worker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Round-robin distribution (default):
  Chip 0 compiles models 0, 4, 8, 12, ...
  Chip 1 compiles models 1, 5, 9, 13, ...
  etc.

Example:
  python3 lib/worker.py --chip 0 --stride 4
  python3 lib/worker.py --chip 2 --stride 4 --results /tmp/results.csv
""",
    )
    parser.add_argument('--chip', type=int, default=0,
                        help='Chip ID (also sets start index for round-robin)')
    parser.add_argument('--stride', type=int, default=4,
                        help='Round-robin stride: chip N gets models N, N+stride, N+2*stride, ...')
    parser.add_argument('--results', type=str, default=None,
                        help='Path to CSV file for saving results')
    args = parser.parse_args()

    # Add project root to path so lib.models resolves from any CWD
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib.models import MODEL_LIST

    model_indices = list(range(args.chip, len(MODEL_LIST), args.stride))
    results_file = Path(args.results) if args.results else None

    sys.exit(run_worker(args.chip, model_indices, results_file))
