from camas import Parallel, Task

build = Parallel(
	Task("build --target={TARGET}"),
	matrix={"TARGET": "x86_64"},
)
