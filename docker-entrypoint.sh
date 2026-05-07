#!/bin/bash
# Docker entrypoint for TT-Forge Compiletron

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${BLUE}ℹ${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
error()   { echo -e "${RED}✗${NC} $1"; }

show_help() {
    cat <<EOF
TT-Forge Compiletron Docker Container

USAGE:
    docker run [options] tt-forge-compiletron <command> [args...]

COMMANDS:
    run     Launch expedition (accepts all expedition.py run flags)
    test    Run test suite
    shell   Interactive shell
    help    Show this help

RUN EXAMPLES:
    # TUI on 4 chips
    docker run --device=/dev/tenstorrent -v data:/app/data \\
        tt-forge-compiletron run --tui --chips 4

    # CLI, 20 models, auto backend (intelligent dispatch)
    docker run --device=/dev/tenstorrent -v data:/app/data \\
        tt-forge-compiletron run --chips 4 --limit 20

    # Seed models only (tt-forge-models zoo)
    docker run --device=/dev/tenstorrent -v data:/app/data \\
        tt-forge-compiletron run --seed-only --limit 10

    # XLA backend
    docker run --device=/dev/tenstorrent -v data:/app/data \\
        tt-forge-compiletron run --backend xla --chips 2

VOLUMES:
    /app/data                  Bestiary and run journals (persist this)
    /root/.cache/huggingface   HuggingFace model cache  (persist for speed)

DEVICES:
    --device=/dev/tenstorrent  Tenstorrent hardware access

EOF
}

activate_forge() {
    if [ ! -d "/tt-forge-fe/env" ]; then
        error "tt-forge-fe not found at /tt-forge-fe"
        exit 1
    fi
    info "Activating Forge environment..."
    source /tt-forge-fe/env/activate
    export PYTHONPATH="/tt-forge-fe/forge:$PYTHONPATH"
}

main() {
    local cmd="${1:-help}"
    shift || true

    case "$cmd" in
        help|--help|-h)
            show_help
            ;;

        run)
            activate_forge
            if [ -f "/opt/ttforge-toolchain/venv/bin/python3" ]; then
                PYTHON_CMD="/opt/ttforge-toolchain/venv/bin/python3"
            else
                PYTHON_CMD="python3"
            fi
            exec $PYTHON_CMD expedition.py run "$@"
            ;;

        test)
            info "Running test suite..."
            exec ./run_tests.sh "$@"
            ;;

        shell|bash|sh)
            exec /bin/bash
            ;;

        *)
            error "Unknown command: $cmd"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
