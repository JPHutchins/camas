# argtree — JP Hutchins's typed/declarative argparse layer that camas depends on
# at runtime (see camas.main.parser). Not yet in nixpkgs (first released 2026-05),
# so the flake packages it out of tree and injects it into python3Packages via an
# overlay (see flake.nix). Drop this file and the overlay once argtree is in nixpkgs.
{
  lib,
  buildPythonPackage,
  fetchPypi,
  hatchling,
  hatch-vcs,
}:

buildPythonPackage rec {
  pname = "argtree";
  version = "0.1.2";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-qRu68JO/Jt0D86h7aTIXWw1vj9AuBUVMbqoIuvLxgNE=";
  };

  # hatch-vcs reads the version from the sdist's baked PKG-INFO (no .git needed).
  build-system = [
    hatchling
    hatch-vcs
  ];

  # No runtime dependencies. Upstream's test suite dev-depends on camas — which
  # depends on argtree — so running it here would be a build cycle; skip it and
  # rely on the import check.
  doCheck = false;

  pythonImportsCheck = [ "argtree" ];

  meta = {
    description = "Typed, declarative, faithful argparse: a command is a dataclass/NamedTuple tree";
    homepage = "https://github.com/JPHutchins/argtree";
    changelog = "https://github.com/JPHutchins/argtree/releases";
    license = lib.licenses.mit;
  };
}
