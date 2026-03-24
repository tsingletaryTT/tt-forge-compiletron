#!/bin/bash
# Test runner for tt-forge-compiletron
# Runs all tests with appropriate pytest flags

cd "$(dirname "$0")"

echo "🧪 Running TT-Forge Compiletron Tests"
echo "======================================"
echo ""

# Run tests with asyncio plugin disabled (compatibility fix)
python3 -m pytest tests/ -v -p no:asyncio "$@"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed (exit code: $exit_code)"
fi

exit $exit_code
