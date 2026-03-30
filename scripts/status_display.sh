#!/usr/bin/env bash
# Renders per-chip ASCII progress bars for the bottom status strip.
# Called by `watch -n1` from run_4way_tmux.sh.
#
# Reads /tmp/compiletron_chip_N.status files written by lib/worker.py.
# Format per file:
#   chip_id=N  current=N  total=N  successes=N  failures=N  model=NAME  done=0|1

# ANSI colors (no tput needed — these are constant)
BOLD=$'\033[1m'
RESET=$'\033[0m'
GREEN=$'\033[32m'
RED=$'\033[31m'
CYAN=$'\033[36m'
YELLOW=$'\033[33m'
DIM=$'\033[2m'

BAR_LEN=30

render_bar() {
    local current=$1 total=$2 successes=$3 failures=$4 model=$5 done=$6

    if [[ "$total" -le 0 ]]; then
        printf "%s[%s] waiting...%s\n" "$DIM" "$(printf '░%.0s' $(seq 1 $BAR_LEN))" "$RESET"
        return
    fi

    local filled=$(( BAR_LEN * current / total ))
    local empty=$(( BAR_LEN - filled ))
    local pct=$(( 100 * current / total ))

    local bar
    bar="$(printf '█%.0s' $(seq 1 $filled 2>/dev/null))$(printf '░%.0s' $(seq 1 $empty 2>/dev/null))"

    local stats="${GREEN}✓${successes}${RESET}/${RED}✗${failures}${RESET}"

    # Trim model name to fit — status strip is narrow
    local label="${model:0:22}"

    if [[ "$done" == "1" ]]; then
        printf "%s[%s]%s %3d%% (%d/%d) %s  %s✓ DONE%s\n" \
            "$BOLD" "$bar" "$RESET" "$pct" "$current" "$total" "$stats" "$GREEN" "$RESET"
    else
        printf "%s[%s]%s %3d%% (%d/%d) %s  %s%s%s\n" \
            "$BOLD" "$bar" "$RESET" "$pct" "$current" "$total" "$stats" "$CYAN" "$label" "$RESET"
    fi
}

# Print one line per chip
for chip in 0 1 2 3; do
    file="/tmp/compiletron_chip_${chip}.status"
    if [[ -f "$file" ]]; then
        # Parse key=value pairs
        chip_id=$(grep '^chip_id=' "$file" | cut -d= -f2)
        current=$(grep '^current=' "$file"  | cut -d= -f2)
        total=$(grep '^total='   "$file"    | cut -d= -f2)
        succ=$(grep '^successes=' "$file"   | cut -d= -f2)
        fail=$(grep '^failures='  "$file"   | cut -d= -f2)
        model=$(grep '^model='    "$file"   | cut -d= -f2-)
        done=$(grep '^done='      "$file"   | cut -d= -f2)

        printf "${BOLD}${YELLOW}Chip %d${RESET} " "$chip"
        render_bar "$current" "$total" "$succ" "$fail" "$model" "$done"
    else
        printf "${BOLD}${YELLOW}Chip %d${RESET} ${DIM}[%-${BAR_LEN}s] waiting for worker...${RESET}\n" \
            "$chip" ""
    fi
done
