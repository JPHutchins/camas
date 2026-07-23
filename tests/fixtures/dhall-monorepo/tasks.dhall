-- Root tasks.dhall for the dhall-monorepo fixture, in the inline (value-composed) style.
-- Groups nest real values (no string refs); `export` renders the document to JSON. The fixture
-- imports a local prelude copy so the test is hermetic; real projects import the hosted prelude.
let camas = ./camas.dhall

let format = camas.task camas.Task::{ cmd = "ruff format {paths}", mutates = True, paths = "." }

let lint = camas.leaf "ruff check ."

let mypy = camas.leaf "mypy ."

let typecheck = camas.parallel [ mypy, camas.leaf "pyright src tests" ]

let test = camas.leaf "pytest -m 'not slow'"

let coverage = camas.leaf "pytest --cov"

let check = camas.parallel [ lint, typecheck, test ]

let gate = camas.parallel [ lint, typecheck, coverage ]

let fix = camas.sequential [ camas.leaf "ruff check --fix .", format ]

let all = camas.sequential [ fix, gate ]

let matrix =
      camas.par
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
