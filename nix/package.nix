# Canonical camas Nix derivation. Consumed by:
#   1. this repo's flake.nix (passes `src = ./.` and a git-rev-based version)
#   2. (future) nixpkgs `pkgs/by-name/ca/camas/package.nix`, which inlines this
#      file's body with `src = fetchFromGitHub {...}` and a hardcoded `version`
#      matching the released tag.
#
# optional-dependencies mirrors pyproject.toml's [project.optional-dependencies]
# by reading it from `src` at eval time, so the extras never drift. That read is
# a plain path lookup here (src = ./.); a nixpkgs port whose src is a fetcher
# would turn it into import-from-derivation, so when porting, inline
# optional-dependencies as a literal alongside a hardcoded `src` and `version`.
{
  lib,
  python3Packages,
  src,
  version,
  extras ? [ ],
  withMypyC ? true,
}:

let
  pname = "camas";

  pyprojectExtras = (lib.importTOML (src + "/pyproject.toml")).project.optional-dependencies;

  parseSpec =
    spec:
    if lib.hasPrefix "${pname}[" spec then
      {
        self = lib.splitString "," (lib.removeSuffix "]" (lib.removePrefix "${pname}[" spec));
      }
    else
      { pkg = builtins.head (builtins.match "([A-Za-z0-9._-]+).*" spec); };

  resolveExtra =
    extra:
    lib.unique (
      lib.concatMap (
        spec:
        let
          parsed = parseSpec spec;
        in
        if parsed ? self then lib.concatMap resolveExtra parsed.self else [ python3Packages.${parsed.pkg} ]
      ) pyprojectExtras.${extra}
    );

  optional-dependencies = lib.mapAttrs (extra: _: resolveExtra extra) pyprojectExtras;
in
python3Packages.buildPythonApplication {
  inherit pname version src;
  pyproject = true;

  env = {
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CAMAS = version;
  }
  // lib.optionalAttrs withMypyC {
    CAMAS_USE_MYPYC = "1";
  };

  build-system = with python3Packages; [
    setuptools
    setuptools-scm
    mypy
  ];

  dependencies = lib.concatMap (extra: optional-dependencies.${extra}) extras;

  inherit optional-dependencies;

  nativeCheckInputs =
    (with python3Packages; [
      pytestCheckHook
      pytest-asyncio
      cyclopts
    ])
    ++ optional-dependencies.all;

  disabledTestMarks = [ "slow" ];

  pythonImportsCheck = [
    "camas"
    "camas.main.dispatch"
    "camas.main.entrypoint"
    "camas.effect.summary"
    "camas.effect.termtree"
  ];

  meta = {
    description = "Task runner with parallel execution, matrix expansion, and pluggable output effects";
    homepage = "https://github.com/JPHutchins/camas";
    changelog = "https://github.com/JPHutchins/camas/releases";
    license = lib.licenses.mit;
    mainProgram = "camas";
    platforms = lib.platforms.unix;
    # maintainers = [ lib.maintainers.jphutchins ];
    # ^ uncomment after PR'ing JP to nixpkgs lib/maintainers/maintainer-list.nix
  };
}
