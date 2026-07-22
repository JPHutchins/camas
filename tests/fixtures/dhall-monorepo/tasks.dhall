-- Root tasks.dhall for the dhall-monorepo fixture. Exercises leaves, both group kinds,
-- nested groups (by name), matrix expansion, a Config with a Claude agent, and a Project child.
let camas = ./camas.dhall

let format = camas.Task::{ cmd = "ruff format {paths}", mutates = True, paths = "." }

let format_check = camas.Task::{ cmd = "ruff format --check {paths}", paths = "." }

let lint = camas.Task::{ cmd = "ruff check {paths}", paths = "." }

let lint_fix = camas.Task::{ cmd = "ruff check --fix {paths}", mutates = True, paths = "." }

let fix = camas.Sequential::{ refs = [ "lint_fix", "format" ] }

let mypy = camas.Task::{ cmd = "mypy ." }

let pyright = camas.Task::{ cmd = "pyright src tests" }

let typecheck = camas.Parallel::{ refs = [ "mypy", "pyright" ] }

let test = camas.Task::{ cmd = "pytest -m 'not slow'" }

let coverage = camas.Task::{ cmd = "pytest --cov" }

let check = camas.Parallel::{ refs = [ "format_check", "lint", "typecheck", "test" ] }

let gate = camas.Parallel::{ refs = [ "format_check", "lint", "typecheck", "coverage" ] }

let all = camas.Sequential::{ refs = [ "fix", "gate" ] }

let matrix =
      camas.Sequential::{
      , refs = [ "check" ]
      , env = toMap { UV_PROJECT_ENVIRONMENT = ".camas/.venv-{PY}", UV_PYTHON = "{PY}" }
      , matrix = toMap { PY = [ "3.13", "3.14", "3.15" ] }
      }

let libs = camas.Project "./libs"

in  { tasks =
        { format
        , format_check
        , lint
        , lint_fix
        , fix
        , mypy
        , pyright
        , typecheck
        , test
        , coverage
        , check
        , gate
        , all
        , matrix
        , libs
        }
    , config = camas.Config::{
      , default_task = "all"
      , github_task = "check"
      , agent = Some camas.Claude::{ fix = "fix", check = "gate" }
      }
    }
