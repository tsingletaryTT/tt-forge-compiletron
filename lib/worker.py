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

# Suppress warnings aggressively
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Terminal colors
GREEN = '\033[92m'
RED = '\033[91m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'


class TimeoutException(Exception):
    """Exception raised when operation times out"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeouts"""
    raise TimeoutException("Operation timed out")


def compile_and_run(model_spec: Tuple, chip_id: int = 0) -> Tuple[bool, float]:
    """
    Compile model and run inference with timeout and retry logic.

    Args:
        model_spec: Model tuple (display_name, family, loader, input_shape, notes, metadata)
        chip_id: Chip ID for logging

    Returns:
        (success, compile_time) tuple
    """
    import torch
    import pyfiglet

    display_name, family, model_loader, input_shape, notes, metadata = model_spec

    # Color rotation for visual variety
    colors = [CYAN, GREEN, YELLOW]
    color = colors[chip_id % len(colors)]

    # Print model banner
    print(f"\n{BOLD}{color}")
    try:
        banner = pyfiglet.figlet_format(display_name, font='standard')
        print(banner, end='')
    except:
        print(f"\n  {display_name}\n")
    print(f"{RESET}")

    print(f"{color}  ★ {display_name} ★{RESET}")
    time.sleep(1)

    print(f"{BOLD}{CYAN}{'─'*80}{RESET}")
    print(f"  {CYAN}Family:{RESET} {family} | {CYAN}Input:{RESET} {input_shape}")
    print(f"  {CYAN}Expected time:{RESET} {metadata['time']:.1f}s | {CYAN}Complexity:{RESET} {metadata['complexity']}")

    start_time = time.time()

    try:
        # Import forge (do this inside try block)
        sys.path.insert(0, os.path.expanduser("~/tt-forge-fe"))
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

                output = compiled_model(sample_input)

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

        # Success
        print(f"  {BOLD}{GREEN}✓ SUCCESS{RESET}")
        print(f"    Compilation: {compile_time:.2f}s | Total: {total_time:.2f}s")
        if hasattr(output, 'shape'):
            print(f"    Output shape: {output.shape}")

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
        print(f"    Error: {type(e).__name__}: {str(e)[:100]}")
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
    from lib.models import MODEL_LIST

    # Stagger startup to avoid race conditions
    startup_delay = random.uniform(0, 5)
    print(f"[Chip {chip_id}] Staggered startup delay: {startup_delay:.2f}s")
    time.sleep(startup_delay)

    # Set environment for this chip
    print(f"[Chip {chip_id}] Worker started")
    print(f"[Chip {chip_id}] TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES')}")
    print(f"[Chip {chip_id}] Models: {len(model_indices)} total")
    print()

    # Compile each model
    successes = 0
    failures = 0
    results = []

    for idx, model_idx in enumerate(model_indices, 1):
        if model_idx >= len(MODEL_LIST):
            print(f"[Chip {chip_id}] WARNING: Model index {model_idx} out of range")
            continue

        model_spec = MODEL_LIST[model_idx]
        display_name = model_spec[0]

        print(f"\n[Chip {chip_id}] Model {idx}/{len(model_indices)}: {display_name}")

        try:
            success, compile_time = compile_and_run(model_spec, chip_id)
            if success:
                successes += 1
            else:
                failures += 1

            results.append({
                'chip_id': chip_id,
                'model_name': display_name,
                'success': success,
                'compile_time': compile_time,
            })

        except Exception as e:
            print(f"[Chip {chip_id}] ERROR: {e}")
            failures += 1
            results.append({
                'chip_id': chip_id,
                'model_name': display_name,
                'success': False,
                'compile_time': 0.0,
            })

    # Summary
    print()
    print("=" * 80)
    print(f"[Chip {chip_id}] COMPLETE - {successes}/{len(model_indices)} succeeded")
    print("=" * 80)

    # Save results if requested
    if results_file:
        import csv
        results_file.parent.mkdir(parents=True, exist_ok=True)

        # Append to CSV
        write_header = not results_file.exists()
        with open(results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['chip_id', 'model_name', 'success', 'compile_time'])
            if write_header:
                writer.writeheader()
            writer.writerows(results)

    return 0 if failures == 0 else 1


# Example usage
if __name__ == '__main__':
    # Test compilation of ResNet-18
    print("Testing worker with ResNet-18...")
    from lib.models import MODEL_LIST

    # Find ResNet-18
    resnet18 = None
    for model in MODEL_LIST:
        if model[0] == 'ResNet-18':
            resnet18 = model
            break

    if resnet18:
        print(f"\nCompiling {resnet18[0]}...")
        success, time_taken = compile_and_run(resnet18, chip_id=0)
        print(f"\nResult: {'SUCCESS' if success else 'FAILED'} in {time_taken:.2f}s")
    else:
        print("ResNet-18 not found in MODEL_LIST")
