#!/usr/bin/env bash
# Renders per-chip ASCII progress bars for the bottom status strip.
# Called by `watch -n1` from run_4way_tmux.sh or run_expedition.sh.
#
# Classic mode: reads /tmp/compiletron_chip_N.status
# Expedition mode (--expedition): reads /tmp/expedition_chip_N.status
#
# Classic format:  chip_id= current= total= successes= failures= model= done=
# Expedition adds: pts= streak= best_streak=

BOLD=$'\033[1m'
RESET=$'\033[0m'
GREEN=$'\033[32m'
RED=$'\033[31m'
CYAN=$'\033[36m'
YELLOW=$'\033[33m'
PURPLE=$'\033[35m'
GOLD=$'\033[33m'
DIM=$'\033[2m'

BAR_LEN=24
MODE="classic"
[[ "$1" == "--expedition" ]] && MODE="expedition"

render_bar_classic() {
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
    local label="${model:0:22}"

    if [[ "$done" == "1" ]]; then
        printf "%s[%s]%s %3d%% %s  %s✓ DONE%s\n" \
            "$BOLD" "$bar" "$RESET" "$pct" "$stats" "$GREEN" "$RESET"
    else
        printf "%s[%s]%s %3d%% %s  %s%s%s\n" \
            "$BOLD" "$bar" "$RESET" "$pct" "$stats" "$CYAN" "$label" "$RESET"
    fi
}

render_bar_expedition() {
    local current=$1 total=$2 successes=$3 failures=$4 model=$5 done=$6 pts=$7 streak=$8

    if [[ "$total" -le 0 ]]; then
        printf "%s[%s] waiting...%s\n" "$DIM" "$(printf '░%.0s' $(seq 1 $BAR_LEN))" "$RESET"
        return
    fi

    local filled=$(( BAR_LEN * current / total ))
    local empty=$(( BAR_LEN - filled ))
    local pct=$(( 100 * current / total ))
    local bar
    bar="$(printf '█%.0s' $(seq 1 $filled 2>/dev/null))$(printf '░%.0s' $(seq 1 $empty 2>/dev/null))"

    local label="${model:0:18}"
    local streak_str=""
    if [[ "$streak" -ge 2 ]]; then
        streak_str=" 🔥×${streak}"
    fi

    if [[ "$done" == "1" ]]; then
        printf "%s[%s]%s %3d%% ${GREEN}✓${successes}${RESET}/${RED}✗${failures}${RESET}  pts:${GOLD}%s${RESET}  ${GREEN}✓ DONE${RESET}\n" \
            "$BOLD" "$bar" "$RESET" "$pct" "$pts"
    else
        printf "%s[%s]%s %3d%% ${GREEN}✓${successes}${RESET}/${RED}✗${failures}${RESET}  pts:${GOLD}%s${RESET}%s  ${CYAN}%s${RESET}\n" \
            "$BOLD" "$bar" "$RESET" "$pct" "$pts" "$streak_str" "$label"
    fi
}

# Print one line per chip
for chip in 0 1 2 3; do
    if [[ "$MODE" == "expedition" ]]; then
        file="/tmp/expedition_chip_${chip}.status"
    else
        file="/tmp/compiletron_chip_${chip}.status"
    fi

    if [[ -f "$file" ]]; then
        # Parse all fields in a single read pass — avoids launching 9 grep
        # subprocesses per chip per refresh cycle.
        chip_id="" current="" total="" succ="" fail="" model="" done_val="" pts="" streak=""
        while IFS= read -r line; do
            key="${line%%=*}"
            val="${line#*=}"
            case "$key" in
                chip_id)   chip_id="$val" ;;
                current)   current="$val" ;;
                total)     total="$val" ;;
                successes) succ="$val" ;;
                failures)  fail="$val" ;;
                model)     model="$val" ;;
                done)      done_val="$val" ;;
                pts)       pts="$val" ;;
                streak)    streak="$val" ;;
            esac
        done < "$file"
        done="$done_val"

        printf "${BOLD}${YELLOW}C%d${RESET} " "$chip"
        if [[ "$MODE" == "expedition" ]]; then
            render_bar_expedition "$current" "$total" "$succ" "$fail" "$model" "$done" "$pts" "$streak"
        else
            render_bar_classic "$current" "$total" "$succ" "$fail" "$model" "$done"
        fi
    else
        printf "${BOLD}${YELLOW}C%d${RESET} ${DIM}[%-${BAR_LEN}s] waiting for worker...${RESET}\n" \
            "$chip" ""
    fi
done
