-- camas Dhall prelude: constructor schemas mirroring the camas public API.
--
-- Dhall has no recursive types, so a group names its children (`refs : List Text`) rather than
-- nesting them inline — the same by-name model as `[tool.camas.tasks]` in pyproject.toml. camas
-- resolves those names to the tasks bound in the same file. Empty-string / empty-list defaults are
-- the "unset" sentinels (camas reads `""` as absent), keeping the surface close to Python kwargs.
--
-- Author a `tasks.dhall` beside this file (or copy it) and evaluate to
-- `{ tasks : <record of nodes>, config : Optional Config }`. See the README for a worked example.

let Map = \(V : Type) -> List { mapKey : Text, mapValue : V }

let AgentFormat =
      { Type = { args : Text, kind : Text, limit : Natural }
      , default = { limit = 8000 }
      }

let Task =
      { Type =
          { kind : Text
          , cmd : Text
          , name : Text
          , env : Map Text
          , cwd : Text
          , help : Text
          , mutates : Bool
          , paths : Text
          , when : List Text
          , agent_format : Optional AgentFormat.Type
          }
      , default =
        { kind = "task"
        , name = ""
        , env = [] : Map Text
        , cwd = ""
        , help = ""
        , mutates = False
        , paths = ""
        , when = [] : List Text
        , agent_format = None AgentFormat.Type
        }
      }

let GroupFields =
      { kind : Text
      , refs : List Text
      , name : Text
      , matrix : Map (List Text)
      , env : Map Text
      , cwd : Text
      , help : Text
      , paths : Text
      , when : List Text
      }

let groupDefaults =
      { name = ""
      , matrix = [] : Map (List Text)
      , env = [] : Map Text
      , cwd = ""
      , help = ""
      , paths = ""
      , when = [] : List Text
      }

let Sequential =
      { Type = GroupFields, default = groupDefaults // { kind = "sequential" } }

let Parallel =
      { Type = GroupFields, default = groupDefaults // { kind = "parallel" } }

let Project = \(path : Text) -> { kind = "project", path }

let Claude =
      { Type = { fix : Text, check : Text, default : Text }
      , default = { check = "", default = "" }
      }

let Config =
      { Type =
          { default_task : Text
          , github_task : Text
          , camas_dir : Text
          , agent : Optional Claude.Type
          }
      , default =
        { default_task = ""
        , github_task = ""
        , camas_dir = ".camas"
        , agent = None Claude.Type
        }
      }

in  { Map
    , AgentFormat
    , Task
    , Sequential
    , Parallel
    , Project
    , Claude
    , Config
    }
