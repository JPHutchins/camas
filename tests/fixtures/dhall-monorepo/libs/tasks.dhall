-- Child project of the dhall-monorepo fixture. Its own Config; camas mounts its tasks under
-- `libs.*` and composes each parent Config field from the child's matching field.
let camas = ../camas.dhall

let lint = camas.Task::{ cmd = "ruff check {paths}", paths = "." }

let test = camas.Task::{ cmd = "pytest libs {paths}", paths = "." }

let check = camas.Parallel::{ refs = [ "lint", "test" ] }

in  { tasks = { lint, test, check }
    , config = camas.Config::{ default_task = "check", github_task = "check" }
    }
