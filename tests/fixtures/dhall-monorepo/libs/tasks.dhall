-- Child project of the dhall-monorepo fixture. Its own Config; camas mounts its tasks under
-- `libs.*` and composes each parent Config field from the child's matching field.
let camas = ../camas.dhall

let lint = camas.leaf "ruff check ."

let test = camas.leaf "pytest libs {paths}"

let check = camas.parallel [ lint, test ]

in  camas.export
      { tasks = toMap { lint, test, check }
      , config = camas.Config::{ default_task = "check", github_task = "check" }
      }
