#!/bin/bash
# Example: Resume - Continue interrupted compilation
#
# This demonstrates how to resume a compilation run that was interrupted.
# The script checks for existing results and skips already-compiled models.
#
# Expected time: Depends on how many models remain
# Hardware required: Any Tenstorrent device

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Example: Resume Interrupted Compilation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Check if Forge environment is activated
if [ -z "$TTFORGE_TOOLCHAIN_DIR" ] && [ -z "$TTMLIR_TOOLCHAIN_DIR" ]; then
    echo "⚠️  Forge environment not activated!"
    echo
    echo "Please activate Forge first:"
    echo "  source ~/tt-forge-fe/env/activate"
    echo
    exit 1
fi

echo "✓ Forge environment detected"
echo

# Check for existing results
if [ ! -d "results" ] || [ -z "$(ls -A results)" ]; then
    echo "❌ No previous results found"
    echo
    echo "Start a new compilation run first:"
    echo "  ./scripts/examples/example-quick-test.sh"
    echo "  ./scripts/examples/example-parallel-sweep.sh"
    echo
    exit 1
fi

# Show existing results
echo "Previous results:"
python3 compiletron.py results
echo

echo "Resume Strategies:"
echo
echo "1. Manual Resume (recommended):"
echo "   - Review which models failed"
echo "   - Re-run specific families or complexity levels"
echo "   - Example: python3 compiletron.py run --family resnet"
echo
echo "2. Skip Completed Models:"
echo "   - Get list of successful models from results CSV"
echo "   - Filter them out using Python script"
echo "   - Run remaining models"
echo
echo "3. Re-run Failed Models Only:"
echo "   - Export results: python3 compiletron.py results export"
echo "   - Parse CSV for failed models"
echo "   - Create custom model list and compile"
echo

# Example: Show failed models
echo "Failed models from last run:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

LATEST_RESULTS=$(ls -t results/results_*.csv 2>/dev/null | head -1)

if [ -n "$LATEST_RESULTS" ]; then
    echo "File: $LATEST_RESULTS"
    echo

    # Extract failed models (where success = False)
    FAILED_COUNT=$(grep ",False," "$LATEST_RESULTS" 2>/dev/null | wc -l)

    if [ "$FAILED_COUNT" -gt 0 ]; then
        echo "Failed models ($FAILED_COUNT):"
        grep ",False," "$LATEST_RESULTS" | cut -d',' -f1 | head -10

        if [ "$FAILED_COUNT" -gt 10 ]; then
            echo "... and $(($FAILED_COUNT - 10)) more"
        fi
    else
        echo "✓ No failed models! All compilations successful."
    fi
else
    echo "No results CSV found"
fi

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "To retry failed models manually:"
echo "  1. Note failed model names from above"
echo "  2. Run: python3 compiletron.py models info <model-name>"
echo "  3. Check family and complexity"
echo "  4. Retry: python3 compiletron.py run --family <family>"
echo
