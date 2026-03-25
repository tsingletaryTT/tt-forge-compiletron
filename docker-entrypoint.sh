#!/bin/bash
# Docker entrypoint for TT-Forge Compiletron

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# Help text
show_help() {
    cat <<EOF
TT-Forge Compiletron Docker Container

USAGE:
    docker run [options] tt-forge-compiletron <command>

COMMANDS:
    detect              Detect Tenstorrent hardware
    models              Model library commands
    test                Run test suite
    compile             Run model compilations
    shell               Interactive shell
    help                Show this help

MODELS SUBCOMMANDS:
    models list         List all models
    models stats        Show model statistics
    models quick        Show fastest models
    models families     Show model families
    models estimate     Estimate compilation time

COMPILE OPTIONS:
    compile --quick             Compile 5 fastest models
    compile --stress            Compile 5 slowest models
    compile --count N           Compile N models
    compile --parallel          Use all chips (parallel)

EXAMPLES:
    # Detect hardware
    docker run --device=/dev/tenstorrent tt-forge-compiletron detect

    # Show model stats
    docker run tt-forge-compiletron models stats

    # Run tests
    docker run tt-forge-compiletron test

    # Quick compilation (5 fast models)
    docker run --device=/dev/tenstorrent \\
        -v ~/tt-metal:/tt-metal:ro \\
        -v ~/tt-forge-fe:/tt-forge-fe:ro \\
        -v compiletron-cache:/cache \\
        tt-forge-compiletron compile --quick

    # Parallel compilation on all chips
    docker run --device=/dev/tenstorrent \\
        -v ~/tt-metal:/tt-metal:ro \\
        -v ~/tt-forge-fe:/tt-forge-fe:ro \\
        -v compiletron-cache:/cache \\
        -v compiletron-results:/results \\
        tt-forge-compiletron compile --parallel --count 50

    # Interactive shell
    docker run -it --device=/dev/tenstorrent \\
        -v ~/tt-metal:/tt-metal:ro \\
        -v ~/tt-forge-fe:/tt-forge-fe:ro \\
        tt-forge-compiletron shell

VOLUMES:
    /tt-metal           Mount tt-metal installation (read-only)
    /tt-forge-fe        Mount tt-forge-fe installation (read-only)
    /cache              PyTorch model cache (persistent)
    /results            Compilation results (persistent)
    /models             Custom model directory (optional)

DEVICES:
    --device=/dev/tenstorrent    Access to Tenstorrent hardware

ENVIRONMENT VARIABLES:
    TT_METAL_HOME       Path to tt-metal (default: /tt-metal)
    FORGE_HOME          Path to tt-forge-fe (default: /tt-forge-fe)
    CACHE_DIR           Cache directory (default: /cache)
    RESULTS_DIR         Results directory (default: /results)

EOF
}

# Main entrypoint logic
main() {
    local cmd="${1:-help}"
    shift || true

    case "$cmd" in
        help|--help|-h)
            show_help
            ;;

        detect)
            info "Detecting Tenstorrent hardware..."
            python3 compiletron.py detect
            ;;

        models)
            python3 compiletron.py models "$@"
            ;;

        test)
            info "Running test suite..."
            ./run_tests.sh "$@"
            ;;

        compile|run)
            info "Starting compilation..."

            # Check if Forge is available
            if [ ! -d "/tt-forge-fe/env" ]; then
                error "tt-forge-fe not found. Mount it with: -v ~/tt-forge-fe:/tt-forge-fe:ro"
                exit 1
            fi

            # Activate Forge environment
            info "Activating Forge environment..."
            source /tt-forge-fe/env/activate

            # Verify environment is set
            if [ -z "$TTFORGE_TOOLCHAIN_DIR" ] && [ -z "$TTMLIR_TOOLCHAIN_DIR" ]; then
                error "Forge environment not properly activated"
                exit 1
            fi

            # Add forge module to Python path
            export PYTHONPATH="/tt-forge-fe/forge:$PYTHONPATH"

            # Use Forge venv Python if available, otherwise system Python
            if [ -f "/opt/ttforge-toolchain/venv/bin/python3" ]; then
                PYTHON_CMD="/opt/ttforge-toolchain/venv/bin/python3"
            else
                PYTHON_CMD="python3"
            fi

            # Run compilation
            $PYTHON_CMD compiletron.py run "$@"
            ;;

        shell|bash|sh)
            info "Starting interactive shell..."
            exec /bin/bash
            ;;

        *)
            # Try to run as compiletron command
            python3 compiletron.py "$cmd" "$@"
            ;;
    esac
}

# Run main function with all arguments
main "$@"
