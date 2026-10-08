# Canonical camas Nix derivation. Consumed by:
#   1. this repo's flake.nix (passes `src = ./.` and a git-rev-based version)
#   2. (future) nixpkgs `pkgs/by-name/ca/camas/package.nix`, which inlines this
#      file's body with `src = fetchFromGitHub {...}` and a hardcoded `version`
#      matching the released tag.
#
# optional-dependencies mirrors pyproject.toml's [project.optional-dependencies],
# and nativeCheckInputs its [dependency-groups].test, by reading it from `src` at
# eval time, so neither drifts. That read is a plain path lookup here (src = ./.);
# a nixpkgs port whose src is a fetcher would turn it into import-from-derivation,
# so when porting, inline both as literals alongside a hardcoded `src` and `version`.
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

  pyproject = lib.importTOML (src + "/pyproject.toml");

  pyprojectExtras = pyproject.project.optional-dependencies;

  resolver = import ./resolve-extras.nix { inherit lib pname; };

  resolverArgs = {
    inherit python3Packages pyprojectExtras;
    pyprojectGroups = pyproject.dependency-groups;
  };

  optional-dependencies = lib.mapAttrs (
    extra: _: resolver.mkResolveExtra resolverArgs extra
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

  nativeCheckInputs = [
    python3Packages.pytestCheckHook
  ]
  ++ resolver.mkResolveGroup resolverArgs "test"
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
