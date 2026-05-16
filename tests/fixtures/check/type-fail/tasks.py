from camas import Parallel, Task

# A matrix axis maps an axis name to a tuple of values; a bare string is a
# common mistake (forgetting the trailing comma to make a 1-tuple).
build = Parallel(
	Task("build --target={TARGET}"),
	matrix={"TARGET": "x86_64"},
)
