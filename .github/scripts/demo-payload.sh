#!/usr/bin/env bash
# Inner payload for the demo recording: simulate a user typing 'camas all' at
# a shell prompt, then exec the real command. Invoked via `asciinema rec -c`
# so every keystroke and the command output end up in the cast.

set -euo pipefail

PROMPT='\033[1;32m❯\033[0m '
CMD=(camas all)
TYPE_DELAY="${TYPE_DELAY:-0.08}"

# Type then run the same argv, so the keystrokes shown can never drift from
# what's executed.
typed="${CMD[*]}"
printf "%b" "$PROMPT"
sleep 0.4
for ((i = 0; i < ${#typed}; i++)); do
	printf "%s" "${typed:$i:1}"
	sleep "$TYPE_DELAY"
done
sleep 0.4
printf '\n'
sleep 0.2

# Record the live Termtree TUI the demo exists to showcase. Under CI,
# GITHUB_ACTIONS=true makes camas auto-select Status(output_mode="github") as the
# default effect, which renders collapsed workflow groups instead of the
# animation — so scrub it for this one process to get the local experience a user
# typing `camas all` would actually see.
exec env -u GITHUB_ACTIONS "${CMD[@]}"
