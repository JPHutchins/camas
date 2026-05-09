# effect-plugin

Demonstrates an Effects plugin: user-defined Effect classes live inline
in `tasks.py`, inherit from `Effect[T]` for static type safety, and are
discovered automatically from the file's scope — usable by name from
`--effects` and listed by `camas --effects`.
