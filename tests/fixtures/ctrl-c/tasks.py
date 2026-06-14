# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Interactive playground for camas's four-press Ctrl-C escalation (issue #16).

    cd tests/fixtures/ctrl-c && camas        # runs `demo`: all five behaviors at once
    camas polite                             # or run one behavior in isolation

Then press Ctrl-C and watch the rows: 1st/2nd forward SIGINT (rows read ^C, then
^C^C), 3rd force-kills survivors (KILL, then STOP as they die), 4th cancels if
camas is left hanging on a leaked pipe. Each exit prints a white
``Ctrl-C (N) received - exiting``. See sigint_app.py for what each behavior does.
"""

from pathlib import Path

from camas import Config, Parallel, Task

_APP = str(Path(__file__).parent / "sigint_app.py")


def _behavior(name: str) -> Task:
	return Task(("python", _APP, name), name=name)


polite = _behavior("polite")
cleanup = _behavior("cleanup")
stubborn = _behavior("stubborn")
ignore = _behavior("ignore")
leaker = _behavior("leaker")

demo = Parallel(polite, cleanup, stubborn, ignore, leaker, name="ctrl-c-demo")

_ = Config(default_task=demo)
