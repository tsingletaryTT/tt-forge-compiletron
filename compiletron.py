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

# Suppress TF/XLA/CUDA/ABSL stderr noise BEFORE any imports that pull in JAX/TF
# (huggingface_hub, transformers, etc. import TF at load time and print to raw stderr)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class _FilteredStderr:
    """Drop noisy C-extension stderr lines from XLA/CUDA/ABSL/TVM at import time."""
    _suppress = [
        'Unable to register cu',
        'computation placer already registered',
        'All log messages before absl::InitializeLog',
        'In-place operator',
        'not found in convert_map',
        'Falling back to out-of-place',
        'num_batches_tracked not found',
        'not found in self._parameters',
        'ConstEval graph:',
        'WARNING  |',
        'DEBUG    |',
    ]
    def __init__(self, stream): self._s = stream; self._seen = set()
    def write(self, t):
        if any(p in t for p in self._suppress): return
        ts = t.strip()
        if ts and 'WARNING' in ts:
            if ts in self._seen: return
            if len(self._seen) < 1000: self._seen.add(ts)
        self._s.write(t)
    def flush(self): self._s.flush()
    def close(self): pass  # Don't close the underlying stream; needed for atexit/logging shutdown
    def isatty(self): return self._s.isatty()
    def fileno(self): return self._s.fileno()

sys.stderr = _FilteredStderr(sys.stderr)

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from hardware import detect_hardware, print_hardware_info, get_hardware_summary, calculate_model_distribution
from models import (
    MODEL_LIST, get_models, get_quick_test_models, get_slow_models,
    get_model_by_name, estimate_total_time, get_model_stats, get_model_families
)
from cache import ModelCache
from forge_setup import check_forge_environment, check_dependencies, print_environment_status, get_activation_instructions
from discovery import (
    discover_forge_models, discover_huggingface_models, deduplicate_models,
    filter_by_family, save_discovered_models, load_discovered_models
)


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


def cmd_test(args):
    """Test single chip with a quick model."""
    import sys
    from pathlib import Path

    # Get hardware
    hw = detect_hardware()

    print(f"🧪 TT-Forge Compiletron - Single Chip Test")
    print(f"==========================================\n")

    if 'error' in hw:
        print(f"❌ Hardware detection failed: {hw['error']}")
        return 1

    # Determine chip to test
    chip_id = args.chip if args.chip is not None else 0

    if chip_id >= hw['num_chips']:
        print(f"❌ Chip {chip_id} not available")
        print(f"   Available chips: 0-{hw['num_chips']-1}")
        return 1

    print(f"Hardware: {get_hardware_summary(hw)}")
    print(f"Testing chip: {chip_id}")
    print()

    # Check if Forge is available
    forge_env = os.environ.get('TTFORGE_TOOLCHAIN_DIR') or os.environ.get('TTMLIR_TOOLCHAIN_DIR')

    if not forge_env:
        print(f"⚠️  Forge environment not activated!")
        print(f"\nTo run test:")
        print(f"  1. Activate Forge: {get_activation_instructions()}")
        print(f"  2. Run this command again")
        print(f"\nOr run in Docker:")
        print(f"  ./docker-run.sh test")
        return 1

    print(f"✓ Forge environment detected")

    # Check if forge module can be imported (quiet import suppresses XLA/ABSL noise)
    sys.path.insert(0, str(Path(__file__).parent / 'lib'))
    from worker import import_forge_quietly
    try:
        import_forge_quietly()
        print(f"✓ Forge module available")
    except ImportError:
        print(f"❌ Cannot import forge module")
        print(f"   Forge may not be properly installed")
        print(f"   Run: python3 compiletron.py setup check")
        return 1

    # Get ResNet-18 for testing (fast, reliable)
    test_model = get_model_by_name("ResNet-18")

    if not test_model:
        print(f"❌ Test model (ResNet-18) not found")
        return 1

    print(f"\nTest model: {test_model[0]}")
    print(f"  Expected time: {test_model[5]['time']:.1f}s")
    print(f"  Complexity: {test_model[5]['complexity']}")
    print()

    # Import worker functionality
    sys.path.insert(0, str(Path(__file__).parent / 'lib'))
    from worker import compile_and_run

    print(f"🚀 Starting test compilation...\n")
    print(f"-" * 60)

    try:
        success, compile_time = compile_and_run(test_model, chip_id)

        if success:
            print(f"-" * 60)
            print(f"\n✅ TEST PASSED")
            print(f"   Chip {chip_id} is working correctly")
            print(f"   Compilation time: {compile_time:.1f}s")
            print(f"   Expected time: {test_model[5]['time']:.1f}s")

            if compile_time < test_model[5]['time'] * 1.5:
                print(f"   ✓ Performance within expected range")
            else:
                print(f"   ⚠️  Slower than expected (may be first run)")

            return 0
        else:
            print(f"-" * 60)
            print(f"\n❌ TEST FAILED")
            print(f"   Compilation failed on chip {chip_id}")
            print(f"   Time taken: {compile_time:.1f}s")
            print(f"\n   Troubleshooting:")
            print(f"   1. Check Forge installation: compiletron setup check")
            print(f"   2. Check device status: tt-smi")
            print(f"   3. Try resetting device: tt-smi -r")
            return 1

    except Exception as e:
        print(f"-" * 60)
        print(f"\n❌ TEST ERROR")
        print(f"   {str(e)}")
        return 1


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


def cmd_results(args):
    """View compilation results."""
    from pathlib import Path
    import csv

    results_dir = Path('results')

    if not results_dir.exists():
        print("❌ No results directory found")
        print("   Run compilations first: compiletron run --quick")
        return 1

    # Find latest results file
    result_files = sorted(results_dir.glob('results_*.csv'), reverse=True)

    if not result_files:
        print("❌ No results files found")
        print("   Run compilations first: compiletron run --quick")
        return 1

    latest_file = result_files[0]

    print(f"📊 Compilation Results")
    print(f"=" * 60)
    print(f"File: {latest_file.name}")
    print()

    # Read and parse results
    with open(latest_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("❌ Results file is empty")
        return 1

    successful = sum(1 for r in rows if r['success'] == 'True')
    failed = sum(1 for r in rows if r['success'] == 'False')
    total = len(rows)

    compile_times = [float(r['compile_time']) for r in rows if r['success'] == 'True']
    total_time = sum(compile_times)
    avg_time = total_time / len(compile_times) if compile_times else 0
    min_time = min(compile_times) if compile_times else 0
    max_time = max(compile_times) if compile_times else 0

    # Summary
    print(f"Summary:")
    print(f"  Total models: {total}")
    print(f"  ✅ Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"  ❌ Failed: {failed} ({failed/total*100:.1f}%)")
    print()

    if compile_times:
        print(f"Compilation Times:")
        print(f"  Total: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Average: {avg_time:.1f}s")
        print(f"  Range: {min_time:.1f}s - {max_time:.1f}s")
        print()

    # Top 5 fastest
    if args.verbose and compile_times:
        successful_rows = [r for r in rows if r['success'] == 'True']
        fastest = sorted(successful_rows, key=lambda r: float(r['compile_time']))[:5]
        slowest = sorted(successful_rows, key=lambda r: float(r['compile_time']), reverse=True)[:5]

        print(f"Fastest 5:")
        for i, row in enumerate(fastest, 1):
            print(f"  {i}. {row['model']} - {float(row['compile_time']):.1f}s")
        print()

        print(f"Slowest 5:")
        for i, row in enumerate(slowest, 1):
            print(f"  {i}. {row['model']} - {float(row['compile_time']):.1f}s")
        print()

    # List all files
    if len(result_files) > 1:
        print(f"Other result files ({len(result_files)-1}):")
        for f in result_files[1:6]:  # Show up to 5 more
            print(f"  {f.name}")
        if len(result_files) > 6:
            print(f"  ... and {len(result_files)-6} more")

    return 0


def cmd_results_report(args):
    """Generate markdown report."""
    from pathlib import Path
    import csv
    import datetime

    results_dir = Path('results')
    result_files = sorted(results_dir.glob('results_*.csv'), reverse=True)

    if not result_files:
        print("❌ No results files found")
        return 1

    latest_file = result_files[0]
    output_file = Path(args.output) if args.output else Path('report.md')

    print(f"Generating report from {latest_file.name}...")

    # Read results
    with open(latest_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Generate markdown
    report = []
    report.append("# TT-Forge Compilation Report\n")
    report.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Source:** {latest_file.name}\n\n")

    # Summary
    successful = sum(1 for r in rows if r['success'] == 'True')
    failed = sum(1 for r in rows if r['success'] == 'False')
    total = len(rows)

    compile_times = [float(r['compile_time']) for r in rows if r['success'] == 'True']
    total_time = sum(compile_times)
    avg_time = total_time / len(compile_times) if compile_times else 0

    report.append("## Summary\n\n")
    report.append(f"- **Total Models:** {total}\n")
    report.append(f"- **Successful:** {successful} ({successful/total*100:.1f}%)\n")
    report.append(f"- **Failed:** {failed} ({failed/total*100:.1f}%)\n")
    report.append(f"- **Total Time:** {total_time:.1f}s ({total_time/60:.1f} min)\n")
    report.append(f"- **Average Time:** {avg_time:.1f}s\n\n")

    # Successful models
    report.append("## Successful Compilations\n\n")
    report.append("| Model | Time (s) | Chip |\n")
    report.append("|-------|----------|------|\n")

    for row in rows:
        if row['success'] == 'True':
            report.append(f"| {row['model']} | {float(row['compile_time']):.1f} | {row.get('chip', 0)} |\n")

    # Failed models
    if failed > 0:
        report.append("\n## Failed Compilations\n\n")
        report.append("| Model | Chip |\n")
        report.append("|-------|------|\n")

        for row in rows:
            if row['success'] == 'False':
                report.append(f"| {row['model']} | {row.get('chip', 0)} |\n")

    # Write report
    with open(output_file, 'w') as f:
        f.writelines(report)

    print(f"✓ Report saved to: {output_file}")

    return 0


def cmd_results_export(args):
    """Export results to CSV."""
    from pathlib import Path
    import shutil

    results_dir = Path('results')
    result_files = sorted(results_dir.glob('results_*.csv'), reverse=True)

    if not result_files:
        print("❌ No results files found")
        return 1

    latest_file = result_files[0]
    output_file = Path(args.output) if args.output else Path('results.csv')

    print(f"Exporting {latest_file.name} to {output_file}...")

    shutil.copy(latest_file, output_file)

    print(f"✓ Results exported to: {output_file}")

    return 0


def cmd_discover_forge(args):
    """Discover models from Forge repositories."""
    print("🔍 Scanning TT-Forge repositories for models...")
    print()

    models = discover_forge_models(args.forge_path)

    if not models:
        print("❌ No models found")
        print("   Make sure tt-forge-fe is installed at ~/tt-forge-fe")
        return 1

    # Deduplicate
    models = deduplicate_models(models)

    print(f"✓ Found {len(models)} unique models\n")

    # Group by family
    families = {}
    for model in models:
        families.setdefault(model.family, []).append(model)

    # Show by family
    for family, family_models in sorted(families.items()):
        print(f"{family}: {len(family_models)} models")
        if args.verbose:
            for m in family_models[:3]:
                print(f"  • {m.name} ({m.source}) - confidence: {m.confidence:.1f}")
            if len(family_models) > 3:
                print(f"  • ... and {len(family_models) - 3} more")

    # Save option
    if args.save:
        output_file = args.save
        save_discovered_models(models, output_file)
        print(f"\n✓ Saved to {output_file}")

    return 0


def cmd_discover_huggingface(args):
    """Discover models from HuggingFace."""
    print(f"🔍 Searching HuggingFace for models...")
    if args.family:
        print(f"   Family: {args.family}")
    if args.task:
        print(f"   Task: {args.task}")
    print(f"   Limit: {args.limit}")
    print()

    models = discover_huggingface_models(
        family=args.family,
        task=args.task,
        limit=args.limit
    )

    if not models:
        print("❌ No models found")
        return 1

    print(f"✓ Found {len(models)} models\n")

    # Show models
    for i, model in enumerate(models, 1):
        downloads = model.metadata.get('downloads', 0)
        print(f"{i}. {model.name}")
        print(f"   Family: {model.family} | Downloads: {downloads:,} | Confidence: {model.confidence:.1f}")
        if args.verbose and model.metadata.get('tags'):
            tags = model.metadata['tags'][:5]
            print(f"   Tags: {', '.join(tags)}")

    # Save option
    if args.save:
        output_file = args.save
        save_discovered_models(models, output_file)
        print(f"\n✓ Saved to {output_file}")

    return 0


def cmd_discover_test(args):
    """Test discovered models by actually compiling them."""
    import json
    import csv
    import datetime
    from pathlib import Path

    # Load discovered models
    if not Path(args.file).exists():
        print(f"❌ File not found: {args.file}")
        print(f"   Run 'compiletron discover huggingface --save' first")
        return 1

    print(f"📋 Loading discovered models from {args.file}...")
    models = load_discovered_models(args.file)

    # Filter by confidence if requested
    if args.min_confidence:
        models = [m for m in models if m.confidence >= args.min_confidence]
        print(f"   Filtered to {len(models)} models with confidence >= {args.min_confidence}")

    # Limit count
    if args.count:
        models = models[:args.count]

    print(f"\n🧪 Testing {len(models)} discovered models")
    print(f"=" * 60)

    # Check hardware
    hw = detect_hardware()
    if 'error' in hw:
        print(f"⚠️  Hardware detection failed, using chip 0")
        chip_id = 0
    else:
        chip_id = args.chip if args.chip is not None else 0

    # Check Forge
    forge_env = os.environ.get('TTFORGE_TOOLCHAIN_DIR') or os.environ.get('TTMLIR_TOOLCHAIN_DIR')
    if not forge_env:
        print(f"\n❌ Forge environment not activated!")
        print(f"   Activate with: {get_activation_instructions()}")
        return 1

    # Import worker first so we can use import_forge_quietly for the availability check
    sys.path.insert(0, str(Path(__file__).parent / 'lib'))
    from worker import compile_and_run, import_forge_quietly

    # Check forge module (use quiet import to suppress XLA/ABSL fd-level noise)
    try:
        import_forge_quietly()
        print(f"✓ Forge module available\n")
    except ImportError:
        print(f"❌ Cannot import forge module")
        print(f"   Forge may not be properly installed")
        return 1

    # Results tracking
    results = []
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f'discovered_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    successful = 0
    failed = 0
    total_compile_time = 0

    # Test each model
    for i, discovered_model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {discovered_model.name}")
        print(f"   Source: {discovered_model.source} | Family: {discovered_model.family}")
        print(f"   Confidence: {discovered_model.confidence:.1f}")
        print(f"-" * 60)

        try:
            # Create model loader based on source
            model_spec = _create_model_spec_from_discovered(discovered_model)

            if not model_spec:
                print(f"⚠️  Skipping: Unable to create loader for {discovered_model.source} source")
                results.append({
                    'model': discovered_model.name,
                    'family': discovered_model.family,
                    'source': discovered_model.source,
                    'success': False,
                    'compile_time': 0,
                    'confidence': discovered_model.confidence,
                    'chip': chip_id
                })
                failed += 1
                continue

            # Try to compile
            success, compile_time = compile_and_run(model_spec, chip_id)

            if success:
                print(f"✅ SUCCESS - {compile_time:.1f}s")
                successful += 1
                total_compile_time += compile_time
            else:
                print(f"❌ FAILED")
                failed += 1

            results.append({
                'model': discovered_model.name,
                'family': discovered_model.family,
                'source': discovered_model.source,
                'success': success,
                'compile_time': compile_time,
                'confidence': discovered_model.confidence,
                'chip': chip_id
            })

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
            results.append({
                'model': discovered_model.name,
                'family': discovered_model.family,
                'source': discovered_model.source,
                'success': False,
                'compile_time': 0,
                'confidence': discovered_model.confidence,
                'chip': chip_id
            })

    # Save results
    if results:
        with open(results_file, 'w', newline='') as f:
            fieldnames = ['model', 'family', 'source', 'success', 'compile_time', 'confidence', 'chip']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n📊 Results saved to: {results_file}")

    # Summary
    print(f"\n" + "=" * 60)
    print(f"DISCOVERY TEST SUMMARY")
    print(f"=" * 60)
    print(f"  Total models: {len(models)}")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    if successful > 0:
        print(f"  Success rate: {successful/len(models)*100:.1f}%")
        print(f"  Total time: {total_compile_time:.1f}s")
        print(f"  Average time: {total_compile_time/successful:.1f}s")
    print()

    return 0 if failed == 0 else 1


def _create_model_spec_from_discovered(discovered_model):
    """
    Create a model_spec tuple from a DiscoveredModel.

    Returns:
        Tuple of (name, family, loader_fn, input_shape, notes, metadata) or None if can't create
    """
    import torch

    name = discovered_model.name
    family = discovered_model.family
    source = discovered_model.source
    location = discovered_model.location

    # Default input shape (most vision models use this)
    input_shape = (1, 3, 224, 224)

    # Create loader function based on source and tags
    loader_fn = None

    # Check if this is a timm model (even if discovered from HuggingFace)
    tags = discovered_model.metadata.get('tags', [])
    is_timm = 'timm' in tags or location.startswith('timm/')

    if is_timm:
        # TIMM models (may be hosted on HuggingFace)
        model_name = location.split('/')[-1] if '/' in location else location
        def loader():
            import timm
            model = timm.create_model(model_name, pretrained=True)
            return model
        loader_fn = loader
        actual_source = 'timm'

    elif source == 'huggingface':
        # HuggingFace models via transformers
        def loader():
            from transformers import AutoModel
            model = AutoModel.from_pretrained(location, trust_remote_code=True)
            return model
        loader_fn = loader
        actual_source = 'huggingface'

    elif source == 'timm':
        # Direct TIMM models
        def loader():
            import timm
            model = timm.create_model(location, pretrained=True)
            return model
        loader_fn = loader
        actual_source = 'timm'

    elif source == 'forge' and 'torchvision.models' in discovered_model.metadata.get('pattern', ''):
        # Torchvision models
        def loader():
            import torchvision.models as models
            # Extract model name from location (file path)
            model_fn = getattr(models, name.lower().replace('-', ''), None)
            if model_fn:
                return model_fn(pretrained=True)
            else:
                raise ValueError(f"Model {name} not found in torchvision.models")
        loader_fn = loader
        actual_source = 'torchvision'

    if not loader_fn:
        return None

    # Create metadata
    metadata = {
        'time': 15.0,  # Default estimate
        'params': 'unknown',
        'complexity': 'medium',
        'success': discovered_model.confidence,
        'discovered': True,
        'source': actual_source
    }

    notes = f"Discovered from {source} (loaded via {actual_source})"

    return (name, family, loader_fn, input_shape, notes, metadata)


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

    # Check if forge module can be imported (quiet import suppresses XLA/ABSL noise)
    sys.path.insert(0, str(Path(__file__).parent / 'lib'))
    from worker import import_forge_quietly
    try:
        import_forge_quietly()
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

    # ========== test command ==========
    parser_test = subparsers.add_parser('test', help='Test single chip')
    parser_test.add_argument('--chip', type=int, default=0, help='Chip ID to test (default: 0)')

    # ========== results command ==========
    parser_results = subparsers.add_parser('results', help='View compilation results')
    results_sub = parser_results.add_subparsers(dest='subcommand')

    results_view = results_sub.add_parser('view', help='View results summary')
    results_view.add_argument('-v', '--verbose', action='store_true', help='Show detailed stats')

    results_report = results_sub.add_parser('report', help='Generate markdown report')
    results_report.add_argument('--output', help='Output file (default: report.md)')

    results_export = results_sub.add_parser('export', help='Export to CSV')
    results_export.add_argument('--output', help='Output file (default: results.csv)')

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

    # ========== discover command ==========
    parser_discover = subparsers.add_parser('discover', help='Discover new models')
    discover_sub = parser_discover.add_subparsers(dest='subcommand')

    # discover forge
    discover_forge = discover_sub.add_parser('forge', help='Scan Forge repositories')
    discover_forge.add_argument('--forge-path', help='Path to tt-forge-fe (default: ~/tt-forge-fe)')
    discover_forge.add_argument('-v', '--verbose', action='store_true', help='Show model details')
    discover_forge.add_argument('--save', help='Save results to JSON file')

    # discover huggingface
    discover_hf = discover_sub.add_parser('huggingface', help='Search HuggingFace model hub')
    discover_hf.add_argument('--family', help='Model family (e.g., resnet, bert)')
    discover_hf.add_argument('--task', help='Task type (e.g., image-classification)')
    discover_hf.add_argument('--limit', type=int, default=20, help='Max models to return')
    discover_hf.add_argument('-v', '--verbose', action='store_true', help='Show model details')
    discover_hf.add_argument('--save', help='Save results to JSON file')

    # discover test
    discover_test = discover_sub.add_parser('test', help='Test discovered models')
    discover_test.add_argument('file', help='JSON file with discovered models')
    discover_test.add_argument('--chip', type=int, help='Chip to test on (default: 0)')
    discover_test.add_argument('--count', type=int, help='Limit number to test')
    discover_test.add_argument('--min-confidence', type=float, help='Minimum confidence (0.0-1.0)')

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

    elif args.command == 'test':
        return cmd_test(args)

    elif args.command == 'results':
        if not args.subcommand:
            # Default to view
            args.verbose = False
            return cmd_results(args)
        elif args.subcommand == 'view':
            return cmd_results(args)
        elif args.subcommand == 'report':
            return cmd_results_report(args)
        elif args.subcommand == 'export':
            return cmd_results_export(args)

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

    elif args.command == 'discover':
        if not args.subcommand:
            print("Usage: compiletron discover {forge,huggingface,test}")
            return 1
        elif args.subcommand == 'forge':
            return cmd_discover_forge(args)
        elif args.subcommand == 'huggingface':
            return cmd_discover_huggingface(args)
        elif args.subcommand == 'test':
            return cmd_discover_test(args)

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
