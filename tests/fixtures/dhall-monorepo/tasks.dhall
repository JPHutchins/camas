-- Root tasks.dhall for the dhall-monorepo fixture, in the inline (value-composed) style.
-- Groups nest real values (no string refs); `export` renders the document to JSON. The test
-- imports the shipped prelude by relative path so it stays offline; real projects import the
-- hosted prelude by URL.
let camas = ../../../src/camas/data/prelude.dhall

let format = camas.taskWith camas.Task::{ cmd = "ruff format {paths}", mutates = True, paths = "." }

let lint = camas.task "ruff check ."

let mypy = camas.task "mypy ."

let typecheck = camas.parallel [ mypy, camas.task "pyright src tests" ]

let test = camas.task "pytest -m 'not slow'"

let coverage = camas.task "pytest --cov"

let check = camas.parallel [ lint, typecheck, test ]

let gate = camas.parallel [ lint, typecheck, coverage ]

let fix = camas.sequential [ camas.task "ruff check --fix .", format ]

let all = camas.sequential [ fix, gate ]

let matrix =
      camas.parallelWith
        camas.Group::{
        , children = [ check ]
        , env = toMap { UV_PROJECT_ENVIRONMENT = ".camas/.venv-{PY}", UV_PYTHON = "{PY}" }
        , matrix = toMap { PY = [ "3.13", "3.14", "3.15" ] }
        }

let libs = camas.project "./libs"

in  camas.export
      { tasks =
          toMap
            { format, lint, mypy, typecheck, test, coverage, check, gate, fix, all, matrix, libs }
      , config = camas.Config::{
        , default_task = "all"
        , github_task = "check"
        , agent = Some camas.Claude::{ fix = "fix", check = "gate" }
        }
      }
