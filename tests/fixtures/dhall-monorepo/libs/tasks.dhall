-- Child project of the dhall-monorepo fixture. Its own Config; camas mounts its tasks under
-- `libs.*` and composes each parent Config field from the child's matching field.
let camas = ../../../../src/camas/data/prelude.dhall

let lint = camas.task "ruff check ."

let test = camas.task "pytest libs {paths}"

let check = camas.parallel [ lint, test ]

in  camas.export
      { tasks = toMap { lint, test, check }
      , config = camas.Config::{ default_task = "check", github_task = "check" }
      }
