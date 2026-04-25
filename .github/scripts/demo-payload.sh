#!/usr/bin/env bash
# Inner payload for the demo recording: simulate a user typing 'camas all' at
# a shell prompt, then exec the real command. Invoked via `asciinema rec -c`
# so every keystroke and the command output end up in the cast.

set -euo pipefail

PROMPT='\033[1;32m❯\033[0m '
CMD='camas all'
TYPE_DELAY="${TYPE_DELAY:-0.08}"

printf "%b" "$PROMPT"
sleep 0.4
for ((i = 0; i < ${#CMD}; i++)); do
	printf "%s" "${CMD:$i:1}"
	sleep "$TYPE_DELAY"
done
sleep 0.4
printf '\n'
sleep 0.2

exec camas all
