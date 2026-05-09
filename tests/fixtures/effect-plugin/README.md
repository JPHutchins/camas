# effect-plugin

Demonstrates an Effects plugin: a user-defined Effect lives inline in
`tasks.py`, satisfies the protocol structurally (no inheritance), and is
discovered automatically from the file's scope — usable by name from
`--effects` and listed by `camas --effects`.
