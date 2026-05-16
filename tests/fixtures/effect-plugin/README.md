# effect-plugin

Demonstrates an Effects plugin: user-defined Effect classes inherit from
`Effect[T]` for static type safety, and are discovered automatically from
the tasks file's scope — usable by name from `--effects` and listed by
`camas --effects`. `FileLog` lives inline in `tasks.py`; `Tail` is defined
in `tail.py` and imported, to show that user effects can be organized
across modules.
