# Canonical camas Nix derivation. Consumed by:
#   1. this repo's flake.nix (passes `src = ../.` and a git-rev-based version)
#   2. (future) nixpkgs `pkgs/by-name/ca/camas/package.nix`, which inlines this
#      file's body with `src = fetchFromGitHub {...}` and a hardcoded `version`
#      matching the released tag.
#
# When publishing to nixpkgs, copy this file's body and replace the two args
# below with let-bindings. Keep the rest identical.
{
  lib,
  python3Packages,
  src,
  version,
  extras ? [ ],
  withMypyC ? true,
}:

python3Packages.buildPythonApplication {
  pname = "camas";
  inherit version src;
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

  dependencies = lib.optionals (lib.elem "github_checks" extras) [
    python3Packages.httpx
  ];

  nativeCheckInputs = with python3Packages; [
    pytestCheckHook
    pytest-asyncio
    cyclopts
    httpx
  ];

  disabledTestMarks = [ "slow" ];

  pythonImportsCheck = [
    "camas"
    "camas.main.dispatch"
    "camas.main.entrypoint"
    "camas.effect.summary"
    "camas.effect.termtree"
  ];

  passthru.optional-dependencies = {
    github_checks = [ python3Packages.httpx ];
  };

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
