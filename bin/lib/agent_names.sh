#!/usr/bin/env bash
# Generate unique agent names of the form <star>_<greek> (e.g., vega_alpha).
# usage: bin/lib/agent_names.sh N [EXCLUDE_FILE]
# EXCLUDE_FILE — names that must not repeat: a name per line or TSV
# where the name is the second field (e.g., "swarm-1<TAB>vega_alpha").
set -euo pipefail

N="${1:-}"
EXCLUDE="${2:-}"
case "$N" in ''|*[!0-9]*) echo "usage: bin/lib/agent_names.sh N [EXCLUDE_FILE]" >&2; exit 1 ;; esac
[ "$N" -ge 1 ] || { echo "N должно быть >= 1" >&2; exit 1; }

WORDS_A=(
  vega rigel sirius altair deneb arcturus capella castor pollux antares
  spica procyon bellatrix betelgeuse aldebaran canopus achernar fomalhaut
  mimosa regulus dubhe alkaid mirfak alderamin
)
WORDS_B=(
  alpha beta gamma delta epsilon zeta eta theta iota kappa
  lambda mu nu xi omicron pi rho sigma tau upsilon
  phi chi psi omega
)

declare -A USED=()
if [ -n "$EXCLUDE" ] && [ -f "$EXCLUDE" ]; then
  while IFS=$'\t ' read -r first second _rest; do
    if [ -n "${second:-}" ]; then name="$second"; else name="$first"; fi
    [ -n "${name:-}" ] && USED["$name"]=1
  done < "$EXCLUDE"
fi

combos=()
for a in "${WORDS_A[@]}"; do
  for b in "${WORDS_B[@]}"; do
    combos+=("${a}_${b}")
  done
done

for ((i = ${#combos[@]} - 1; i > 0; i--)); do
  j=$((RANDOM % (i + 1)))
  tmp="${combos[i]}"
  combos[i]="${combos[j]}"
  combos[j]="$tmp"
done

printed=0
for name in "${combos[@]}"; do
  [ -n "${USED[$name]:-}" ] && continue
  printf '%s\n' "$name"
  printed=$((printed + 1))
  [ "$printed" -ge "$N" ] && break
done
if [ "$printed" -lt "$N" ]; then
  echo "не хватает уникальных имён: нужно $N, доступно $printed" >&2
  exit 1
fi
