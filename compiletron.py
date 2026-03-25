#!/usr/bin/env python3
"""
TT-Forge Compiletron - Main CLI

Clean, standalone tool for running Forge compilation demos on Tenstorrent hardware.
Supports 1 to 32+ chips with automatic detection and parallel execution.
"""

import os
import sys
import argparse
from pathlib import Path

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from hardware import detect_hardware, print_hardware_info, get_hardware_summary, calculate_model_distribution
from models import (
    MODEL_LIST, get_models, get_quick_test_models, get_slow_models,
    get_model_by_name, estimate_total_time, get_model_stats, get_model_families
)
from cache import ModelCache
from forge_setup import check_forge_environment, check_dependencies, print_environment_status, get_activation_instructions


def cmd_detect(args):
    """Detect hardware command."""
    hw = detect_hardware()

    if 'error' in hw:
        print(f"❌ {hw['error']}")
        return 1

    print_hardware_info(hw)
    print(f"\nSummary: {get_hardware_summary(hw)}")

    # Show round-robin distribution example
    if hw['num_chips'] > 0:
        print(f"\nRound-robin distribution (108 models):")
        dist = calculate_model_distribution(108, hw['num_chips'])
        for chip_id, model_ids in dist:
            print(f"  Chip {chip_id}: {len(model_ids)} models")

    return 0


def cmd_models_list(args):
    """List models command."""
    # Get filtered models
    models = get_models(
        family=args.family,
        batch_size=args.batch_size,
        input_size=args.input_size,
        complexity=args.complexity,
        count=args.count,
        sort_by=args.sort_by
    )

    print(f"Found {len(models)} models")
    if args.family:
        print(f"  Family: {args.family}")
    if args.complexity:
        print(f"  Complexity: {args.complexity}")
    print()

    # Print models
    for model in models:
        name, family, loader, shape, notes, meta = model
        print(f"  {name}")
        print(f"    Family: {family} | Time: {meta['time']:.1f}s | Complexity: {meta['complexity']}")
        print(f"    Params: {meta['params']} | Shape: {shape}")
        print()

    return 0


def cmd_models_families(args):
    """List model families."""
    families = get_model_families()

    print(f"Model Families ({len(families)} total):\n")

    # Sort by count
    sorted_families = sorted(families.items(), key=lambda x: len(x[1]), reverse=True)

    for family, models in sorted_families:
        print(f"  {family}: {len(models)} models")
        if args.verbose:
            for model in models[:3]:  # Show first 3
                print(f"    • {model[0]}")
            if len(models) > 3:
                print(f"    • ... and {len(models) - 3} more")

    return 0


def cmd_models_info(args):
    """Show model info."""
    model = get_model_by_name(args.name)

    if not model:
        print(f"Model not found: {args.name}")
        return 1

    name, family, loader, shape, notes, meta = model

    print(f"Model: {name}")
    print(f"  Family: {family}")
    print(f"  Notes: {notes}")
    print(f"  Input shape: {shape}")
    print(f"  Parameters: {meta['params']}")
    print(f"  Expected compile time: {meta['time']:.1f}s")
    print(f"  Complexity: {meta['complexity']}")
    print(f"  Success rate: {meta['success']*100:.0f}%")

    return 0


def cmd_models_quick(args):
    """Show quick test models."""
    models = get_quick_test_models(args.count)

    print(f"Quick Test Models (fastest {args.count}):\n")
    for model in models:
        name, family, loader, shape, notes, meta = model
        print(f"  {name}: {meta['time']:.1f}s")

    return 0


def cmd_models_stress(args):
    """Show stress test models."""
    models = get_slow_models(args.count)

    print(f"Stress Test Models (slowest {args.count}):\n")
    for model in models:
        name, family, loader, shape, notes, meta = model
        print(f"  {name}: {meta['time']:.1f}s")

    return 0


def cmd_models_stats(args):
    """Show model statistics."""
    stats = get_model_stats()

    print("Model Library Statistics:")
    print(f"  Total models: {stats['total_models']}")
    print(f"  Families: {stats['families']}")
    print(f"  Average compile time: {stats['avg_compile_time']:.1f}s")
    print(f"  Success rate: {stats['success_rate']*100:.1f}%")

    print(f"\nBy complexity:")
    for comp, count in stats['by_complexity'].items():
        print(f"  {comp}: {count} models")

    print(f"\nTop families:")
    sorted_families = sorted(stats['by_family'].items(), key=lambda x: x[1], reverse=True)[:10]
    for family, count in sorted_families:
        print(f"  {family}: {count} models")

    return 0


def cmd_models_estimate(args):
    """Estimate compilation time."""
    # Get models
    models = get_models(
        family=args.family,
        complexity=args.complexity,
        count=args.count
    )

    # Get hardware
    hw = detect_hardware()
    num_chips = args.chips if args.chips else hw.get('num_chips', 1)

    # Estimate
    total_time = estimate_total_time(models, num_chips)

    print(f"Estimate for {len(models)} models on {num_chips} chip(s):")
    print(f"  Total time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print(f"  Parallel execution: {len(models)} models / {num_chips} chips")
    print(f"  Average per chip: {total_time/60:.1f} minutes")

    return 0


def cmd_cache_status(args):
    """Show cache status."""
    cache = ModelCache()
    stats = cache.get_cache_stats()

    print("Model Cache Status:")
    print(f"  PyTorch cache: {stats['total_size_mb']:.1f} MB ({stats['num_files']} files)")
    print(f"  Location: ~/.cache/torch/hub/checkpoints/")

    return 0


def cmd_cache_clear(args):
    """Clear cache."""
    cache = ModelCache()

    if args.yes:
        cache.clear_cache(confirm=True)
        print("✓ Cache cleared")
    else:
        cache.clear_cache(confirm=False)
        print("\nTo actually clear cache, run with --yes flag")

    return 0


def cmd_setup_check(args):
    """Check environment setup."""
    print_environment_status()
    print()

    # Check if ready to run
    forge_env = check_forge_environment()
    deps = check_dependencies()

    ready = (
        forge_env['forge_installed'] and
        deps['python_ok'] and
        deps['tt_metal_ok'] and
        deps['pytorch_installed']
    )

    if ready:
        print("✓ Environment ready!")
        if not forge_env['env_activated']:
            print(f"\n⚠️  Remember to activate Forge environment:")
            print(f"    {get_activation_instructions()}")
    else:
        print("✗ Environment not ready. Missing:")
        if not forge_env['forge_installed']:
            print("  • tt-forge-fe (run: compiletron setup --install-forge)")
        if not deps['python_ok']:
            print(f"  • Python >= 3.12 (current: {deps['python_version']})")
        if not deps['tt_metal_ok']:
            print("  • tt-metal (install from tenstorrent/tt-metal)")
        if not deps['pytorch_installed']:
            print("  • PyTorch (pip install torch)")

    return 0 if ready else 1


def cmd_setup_install_forge(args):
    """Install Forge."""
    from forge_setup import install_forge

    print("Installing tt-forge-fe...")
    print("This will take 45-60 minutes.")
    print()

    if not args.yes:
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled")
            return 1

    success, message = install_forge()
    print(message)

    if success:
        print("\n✓ Installation complete!")
        print(f"\nActivate environment with:")
        print(f"    {get_activation_instructions()}")
        return 0
    else:
        return 1


def cmd_run(args):
    """Run model compilations."""
    import subprocess
    from pathlib import Path
    import csv
    import datetime

    # Get hardware
    hw = detect_hardware()

    print(f"🎰 TT-Forge Compiletron - Compilation Run")
    print(f"==========================================")

    if 'error' in hw:
        print(f"⚠️  Hardware detection failed: {hw['error']}")
        print(f"    Proceeding with default settings (single chip)")
        # Set default hardware config
        hw = {
            'num_chips': 1,
            'board_type': 'unknown',
            'architecture': 'unknown',
            'devices': []
        }
    else:
        print(f"Hardware: {get_hardware_summary(hw)}")

    # Get models
    if args.quick:
        models = get_quick_test_models(5)
        print(f"Mode: Quick test (5 fastest models)")
    elif args.stress:
        models = get_slow_models(5)
        print(f"Mode: Stress test (5 slowest models)")
    else:
        models = get_models(
            family=args.family,
            complexity=args.complexity,
            count=args.count
        )
        print(f"Mode: Standard ({len(models)} models)")

    # Show models
    print(f"\nModels to compile:")
    for i, model in enumerate(models[:10], 1):
        print(f"  {i}. {model[0]} ({model[5]['time']:.1f}s)")
    if len(models) > 10:
        print(f"  ... and {len(models) - 10} more")

    # Estimate time
    num_chips = hw['num_chips'] if args.parallel and hw['num_chips'] > 0 else 1
    total_time = estimate_total_time(models, num_chips)

    print(f"\nEstimate:")
    print(f"  Chips: {num_chips}")
    print(f"  Total time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print()

    # Check if Forge is available
    forge_env = os.environ.get('TTFORGE_TOOLCHAIN_DIR') or os.environ.get('TTMLIR_TOOLCHAIN_DIR')

    if not forge_env:
        print(f"⚠️  Forge environment not activated!")
        print(f"\nTo run compilations:")
        print(f"  1. Activate Forge: {get_activation_instructions()}")
        print(f"  2. Run this command again")
        print(f"\nOr run in Docker:")
        print(f"  ./docker-run.sh compile --quick")
        return 1

    print(f"✓ Forge environment detected")

    # Check if forge module can be imported
    try:
        import forge
        print(f"✓ Forge module available")
    except ImportError:
        print(f"❌ Cannot import forge module")
        print(f"   Forge may not be properly installed")
        print(f"   Run: python3 compiletron.py setup check")
        return 1

    # Single chip or parallel?
    if args.parallel and num_chips > 1:
        print(f"\n🚀 Parallel Mode - {num_chips} chips")
        print(f"\nFor parallel execution, use the orchestrator script:")
        print(f"  cd ~/code/tt-forge-compiletron")
        print(f"  bash scripts/run_parallel.sh")
        print(f"\nOr run in tmux:")
        print(f"  bash scripts/view_logs.sh")
        print()
        return 0

    # Single chip compilation
    print(f"\n🚀 Starting single-chip compilation...")
    print(f"   Chip: {args.chip if args.chip is not None else 0}")
    print(f"   Models: {len(models)}")
    print()

    # Import worker functionality
    sys.path.insert(0, str(Path(__file__).parent / 'lib'))
    from worker import compile_and_run

    # Results tracking
    results = []
    results_file = Path('results') / f'results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results_file.parent.mkdir(exist_ok=True)

    successful = 0
    failed = 0
    total_compilation_time = 0

    # Process each model
    for i, model_spec in enumerate(models, 1):
        model_name = model_spec[0]
        expected_time = model_spec[5]['time']

        print(f"\n[{i}/{len(models)}] {model_name} (expected: {expected_time:.1f}s)")
        print(f"-" * 60)

        try:
            success, compile_time = compile_and_run(model_spec, args.chip or 0)

            if success:
                print(f"✅ SUCCESS - {compile_time:.1f}s")
                successful += 1
                total_compilation_time += compile_time
                results.append({
                    'model': model_name,
                    'status': 'success',
                    'time': compile_time,
                    'expected': expected_time
                })
            else:
                print(f"❌ FAILED")
                failed += 1
                results.append({
                    'model': model_name,
                    'status': 'failed',
                    'time': 0,
                    'expected': expected_time
                })

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
            results.append({
                'model': model_name,
                'status': 'error',
                'time': 0,
                'expected': expected_time
            })

    # Save results
    if results:
        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model', 'status', 'time', 'expected'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n📊 Results saved to: {results_file}")

    # Summary
    print(f"\n" + "=" * 60)
    print(f"COMPILATION SUMMARY")
    print(f"=" * 60)
    print(f"  Total models: {len(models)}")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  Success rate: {successful/len(models)*100:.1f}%")
    if successful > 0:
        print(f"  Total time: {total_compilation_time:.1f}s ({total_compilation_time/60:.1f} min)")
        print(f"  Average time: {total_compilation_time/successful:.1f}s per model")
    print()

    return 0 if failed == 0 else 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='TT-Forge Compiletron - Model compilation for Tenstorrent hardware',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # ========== detect command ==========
    parser_detect = subparsers.add_parser('detect', help='Detect hardware')

    # ========== models command ==========
    parser_models = subparsers.add_parser('models', help='Model library commands')
    models_sub = parser_models.add_subparsers(dest='subcommand')

    # models list
    models_list = models_sub.add_parser('list', help='List models')
    models_list.add_argument('--family', help='Filter by family')
    models_list.add_argument('--batch-size', type=int, help='Filter by batch size')
    models_list.add_argument('--input-size', type=int, help='Filter by input size')
    models_list.add_argument('--complexity', choices=['low', 'medium', 'high'], help='Filter by complexity')
    models_list.add_argument('--count', type=int, help='Limit number of models')
    models_list.add_argument('--sort-by', default='compile_time', choices=['compile_time', 'name', 'family', 'size'])

    # models families
    models_families = models_sub.add_parser('families', help='List model families')
    models_families.add_argument('-v', '--verbose', action='store_true', help='Show model names')

    # models info
    models_info = models_sub.add_parser('info', help='Show model info')
    models_info.add_argument('name', help='Model name')

    # models quick
    models_quick = models_sub.add_parser('quick', help='Show quick test models')
    models_quick.add_argument('--count', type=int, default=5, help='Number of models')

    # models stress
    models_stress = models_sub.add_parser('stress', help='Show stress test models')
    models_stress.add_argument('--count', type=int, default=5, help='Number of models')

    # models stats
    models_stats = models_sub.add_parser('stats', help='Show model statistics')

    # models estimate
    models_estimate = models_sub.add_parser('estimate', help='Estimate compilation time')
    models_estimate.add_argument('--family', help='Filter by family')
    models_estimate.add_argument('--complexity', help='Filter by complexity')
    models_estimate.add_argument('--count', type=int, help='Number of models')
    models_estimate.add_argument('--chips', type=int, help='Number of chips (default: auto-detect)')

    # ========== cache command ==========
    parser_cache = subparsers.add_parser('cache', help='Cache management')
    cache_sub = parser_cache.add_subparsers(dest='subcommand')

    cache_status = cache_sub.add_parser('status', help='Show cache status')
    cache_clear = cache_sub.add_parser('clear', help='Clear cache')
    cache_clear.add_argument('--yes', action='store_true', help='Confirm deletion')

    # ========== setup command ==========
    parser_setup = subparsers.add_parser('setup', help='Setup and installation')
    setup_sub = parser_setup.add_subparsers(dest='subcommand')

    setup_check = setup_sub.add_parser('check', help='Check environment')
    setup_install = setup_sub.add_parser('install-forge', help='Install tt-forge-fe')
    setup_install.add_argument('--yes', action='store_true', help='Skip confirmation')

    # ========== run command (simplified) ==========
    parser_run = subparsers.add_parser('run', help='Run compilation (simplified)')
    parser_run.add_argument('--parallel', action='store_true', help='Run on all chips')
    parser_run.add_argument('--chip', type=int, help='Specific chip ID')
    parser_run.add_argument('--family', help='Model family')
    parser_run.add_argument('--complexity', help='Model complexity')
    parser_run.add_argument('--count', type=int, default=10, help='Number of models')
    parser_run.add_argument('--quick', action='store_true', help='Quick test (5 fastest)')
    parser_run.add_argument('--stress', action='store_true', help='Stress test (5 slowest)')

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Route to command handlers
    if args.command == 'detect':
        return cmd_detect(args)

    elif args.command == 'models':
        if not args.subcommand:
            print("Usage: compiletron models {list,families,info,quick,stress,stats,estimate}")
            return 1
        elif args.subcommand == 'list':
            return cmd_models_list(args)
        elif args.subcommand == 'families':
            return cmd_models_families(args)
        elif args.subcommand == 'info':
            return cmd_models_info(args)
        elif args.subcommand == 'quick':
            return cmd_models_quick(args)
        elif args.subcommand == 'stress':
            return cmd_models_stress(args)
        elif args.subcommand == 'stats':
            return cmd_models_stats(args)
        elif args.subcommand == 'estimate':
            return cmd_models_estimate(args)

    elif args.command == 'cache':
        if not args.subcommand:
            print("Usage: compiletron cache {status,clear}")
            return 1
        elif args.subcommand == 'status':
            return cmd_cache_status(args)
        elif args.subcommand == 'clear':
            return cmd_cache_clear(args)

    elif args.command == 'setup':
        if not args.subcommand:
            print("Usage: compiletron setup {check,install-forge}")
            return 1
        elif args.subcommand == 'check':
            return cmd_setup_check(args)
        elif args.subcommand == 'install-forge':
            return cmd_setup_install_forge(args)

    elif args.command == 'run':
        return cmd_run(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
