"""camas tasks for the playwright-matrix-override fixture.

A 2-axis matrix (BROWSER x VIEWPORT = 6 combinations) is the natural shape
of cross-browser end-to-end testing. The matrix override CLI lets you scope
a run to a slice without editing tasks.py.
"""

from camas import Config, Parallel, Sequential, Task

install = Task("npx playwright install --with-deps")

e2e = Parallel(
	Task("npx playwright test --project={BROWSER}-{VIEWPORT}"),
	matrix={
		"BROWSER": ("chromium", "firefox", "webkit"),
		"VIEWPORT": ("desktop", "mobile"),
	},
)

smoke = Parallel(
	Task("npx playwright test --grep @smoke --project={BROWSER}-{VIEWPORT}"),
	matrix={
		"BROWSER": ("chromium", "firefox", "webkit"),
		"VIEWPORT": ("desktop", "mobile"),
	},
)

ci = Sequential(install, e2e)

_ = Config(default_task=ci)
