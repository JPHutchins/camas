-- camas Dhall prelude — self-contained (no imports). Constructors build the canonical camas
-- task JSON; `export` renders the whole document to the JSON string camas parses. Users compose
-- with real values (inline nesting), not string references.

let OutputKind = < sarif | rdjson | lsp | junit | tap | raw >

let StrMap = List { mapKey : Text, mapValue : Text }

let StrsMap = List { mapKey : Text, mapValue : List Text }

let esc =
      \(s : Text) ->
        let e = Text/replace "\\" "\\\\" s
        let e = Text/replace "\"" "\\\"" e
        let e = Text/replace "${"\n"}" "\\n" e
        let e = Text/replace "${"\r"}" "\\r" e
        let e = Text/replace "${"\t"}" "\\t" e
        in  "\"${e}\""

let joinMap =
      \(f : Text -> Text) ->
      \(xs : List Text) ->
        ( List/fold
            Text
            xs
            { first : Bool, out : Text }
            ( \(x : Text) ->
              \(a : { first : Bool, out : Text }) ->
                { first = False, out = if a.first then f x else "${f x},${a.out}" }
            )
            { first = True, out = "" }
        ).out

let id = \(x : Text) -> x

let arr = \(items : List Text) -> "[${joinMap id items}]"

let obj = \(fields : List Text) -> "{${joinMap id fields}}"

let strArr = \(xs : List Text) -> "[${joinMap esc xs}]"

let field = \(k : Text) -> \(v : Text) -> "${esc k}:${v}"

let jbool = \(b : Bool) -> if b then "true" else "false"

let jstrObj =
      \(m : StrMap) ->
        obj
          ( List/fold
              { mapKey : Text, mapValue : Text }
              m
              (List Text)
              ( \(p : { mapKey : Text, mapValue : Text }) ->
                \(acc : List Text) ->
                  [ field p.mapKey (esc p.mapValue) ] # acc
              )
              ([] : List Text)
          )

let matrixObj =
      \(m : StrsMap) ->
        obj
          ( List/fold
              { mapKey : Text, mapValue : List Text }
              m
              (List Text)
              ( \(p : { mapKey : Text, mapValue : List Text }) ->
                \(acc : List Text) ->
                  [ field p.mapKey (strArr p.mapValue) ] # acc
              )
              ([] : List Text)
          )

let AgentFormat =
      { Type = { args : Text, kind : OutputKind, limit : Natural }
      , default.limit = 8000
      }

let formatJSON =
      \(f : Optional AgentFormat.Type) ->
        merge
          { None = "null"
          , Some =
              \(a : AgentFormat.Type) ->
                obj
                  [ field "args" (esc a.args)
                  , field
                      "kind"
                      ( esc
                          ( merge
                              { sarif = "sarif"
                              , rdjson = "rdjson"
                              , lsp = "lsp"
                              , junit = "junit"
                              , tap = "tap"
                              , raw = "raw"
                              }
                              a.kind
                          )
                      )
                  , field "limit" (Natural/show a.limit)
                  ]
          }
          f

let Task =
      { Type =
          { cmd : Text
          , name : Text
          , env : StrMap
          , cwd : Text
          , help : Text
          , mutates : Bool
          , paths : Text
          , when : List Text
          , format : Optional AgentFormat.Type
          }
      , default =
        { name = ""
        , env = [] : StrMap
        , cwd = ""
        , help = ""
        , mutates = False
        , paths = ""
        , when = [] : List Text
        , format = None AgentFormat.Type
        }
      }

let taskWith =
      \(o : Task.Type) ->
        obj
          [ field "kind" (esc "task")
          , field "cmd" (esc o.cmd)
          , field "name" (esc o.name)
          , field "env" (jstrObj o.env)
          , field "cwd" (esc o.cwd)
          , field "help" (esc o.help)
          , field "mutates" (jbool o.mutates)
          , field "paths" (esc o.paths)
          , field "when" (strArr o.when)
          , field "agent_format" (formatJSON o.format)
          ]

let task = \(cmd : Text) -> taskWith Task::{ cmd }

let Group =
      { Type =
          { children : List Text
          , name : Text
          , env : StrMap
          , cwd : Text
          , help : Text
          , paths : Text
          , when : List Text
          , matrix : StrsMap
          }
      , default =
        { name = ""
        , env = [] : StrMap
        , cwd = ""
        , help = ""
        , paths = ""
        , when = [] : List Text
        , matrix = [] : StrsMap
        }
      }

let groupObj =
      \(kind : Text) ->
      \(o : Group.Type) ->
        obj
          [ field "kind" (esc kind)
          , field "children" (arr o.children)
          , field "name" (esc o.name)
          , field "env" (jstrObj o.env)
          , field "cwd" (esc o.cwd)
          , field "help" (esc o.help)
          , field "paths" (esc o.paths)
          , field "when" (strArr o.when)
          , field "matrix" (matrixObj o.matrix)
          ]

let parallelWith = groupObj "parallel"

let sequentialWith = groupObj "sequential"

let parallel = \(children : List Text) -> parallelWith Group::{ children }

let sequential = \(children : List Text) -> sequentialWith Group::{ children }

let project = \(path : Text) -> obj [ field "kind" (esc "project"), field "path" (esc path) ]

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

let configJSON =
      \(c : Config.Type) ->
        obj
          [ field "default_task" (esc c.default_task)
          , field "github_task" (esc c.github_task)
          , field "camas_dir" (esc c.camas_dir)
          , field
              "agent"
              ( merge
                  { None = "null"
                  , Some =
                      \(a : Claude.Type) ->
                        obj
                          [ field "fix" (esc a.fix)
                          , field "check" (esc a.check)
                          , field "default" (esc a.default)
                          ]
                  }
                  c.agent
              )
          ]

let export =
      \(x : { tasks : StrMap, config : Config.Type }) ->
        obj
          [ field
              "tasks"
              ( obj
                  ( List/fold
                      { mapKey : Text, mapValue : Text }
                      x.tasks
                      (List Text)
                      ( \(p : { mapKey : Text, mapValue : Text }) ->
                        \(acc : List Text) ->
                          [ field p.mapKey p.mapValue ] # acc
                      )
                      ([] : List Text)
                  )
              )
          , field "config" (configJSON x.config)
          ]

in  { OutputKind
    , AgentFormat
    , Task
    , Group
    , Claude
    , Config
    , task
    , taskWith
    , parallel
    , parallelWith
    , sequential
    , sequentialWith
    , project
    , export
    }
