# Ctrl-C escalation playground

Interactive fixture for camas's four-press Ctrl-C handling ([#16](https://github.com/JPHutchins/camas/issues/16)).
Five subprocesses, each with a different SIGINT response, so one run reproduces
real-world variance — some die politely, some need a nudge, some ignore signals
outright, one leaves camas hanging.

```sh
cd tests/fixtures/ctrl-c
camas            # runs `demo`: all five behaviors in parallel
camas polite     # or drive one behavior in isolation
```

Then press **Ctrl-C** and watch the rows:

| Press | camas does | what you see |
|---|---|---|
| 1st | forward `SIGINT` to every running leaf | `polite` settles `STOP`; others flip to `^C` |
| 2nd | forward `SIGINT` again | `stubborn` settles `STOP`; survivors read `^C^C`; `cleanup` finishes its ~2s exit |
| 3rd | `SIGKILL` the survivors | `ignore` reads `KILL` then settles `STOP` |
| 4th | gracefully cancel the run | `leaker` had left camas hanging on a held-open pipe — this unwinds it; exit `130` |

Every interrupted exit prints a white `Ctrl-C (N) received - exiting` final line (N = total presses). `STOP` is a dark red, `KILL` a bold yellow.

Run `demo` to the end and you exercise all four presses: the `leaker` keeps camas
alive after the kill (a detached grandchild holds its stdout open, so the read
never EOFs), and only the 4th press cancels the run. See
[`sigint_app.py`](sigint_app.py) for exactly what each behavior does.

POSIX-oriented — this is the platform where camas runs the full escalation.
