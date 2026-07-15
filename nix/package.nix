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

  resolver = import ./resolve-extras.nix { inherit lib pname; };

  optional-dependencies = lib.mapAttrs (
    extra: _: resolver.mkResolveExtra { inherit python3Packages pyprojectExtras; } extra
  ) pyprojectExtras;
in
python3Packages.buildPythonApplication {
  inherit pname version src;
  pyproject = true;

  # An ambient PYTHONPATH (e.g. a consumer mkShell's python hook) can shadow the
  # wrapped app's own site-packages with wrong-ABI modules.
  makeWrapperArgs = [ "--unset PYTHONPATH" ];

  # An application is a leaf: its deps are baked into the wrapper, so propagating
  # them (python3 included) would stuff a consumer mkShell's PYTHONPATH with this
  # app's entire python closure.
  postFixup = ''
    : > "$out/nix-support/propagated-build-inputs"
  '';

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
      jsonschema
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
