# Pure extras resolver extracted from package.nix so its guards are exercisable
# from the flake `extras-resolver` check. It maps each pyproject
# [project.optional-dependencies] spec to a nixpkgs python3Packages derivation,
# following `camas[...]` self-references, and throws a readable message on the
# four edges that would otherwise fail with an opaque eval trace or loop forever:
#   1. a PyPI name whose nixpkgs attr differs (map it in pypiToNixAttr),
#   2. a package absent from python3Packages,
#   3. a mutually self-referential extra pair (cycle), and
#   4. a spec that is not a parseable PEP 508 name.
{
  lib,
  pname,
}:
let
  parseSpec =
    spec:
    if lib.hasPrefix "${pname}[" spec then
      {
        self = lib.splitString "," (lib.removeSuffix "]" (lib.removePrefix "${pname}[" spec));
      }
    else
      let
        matched = builtins.match "([A-Za-z0-9._-]+).*" spec;
      in
      if matched == null then
        throw "camas resolve-extras: cannot parse requirement ${builtins.toJSON spec}; expected a spec beginning with a PEP 508 name ([A-Za-z0-9._-])"
      else
        { pkg = builtins.head matched; };

  mkResolveExtra =
    {
      python3Packages,
      pyprojectExtras,
      pypiToNixAttr ? { },
    }:
    let
      resolvePkg =
        extra: name:
        let
          attr = pypiToNixAttr.${name} or name;
        in
        if python3Packages ? ${attr} then
          python3Packages.${attr}
        else
          throw "camas resolve-extras: extra '${extra}' requires Python package '${name}' (nixpkgs attr '${attr}') which is absent from python3Packages; add it to nixpkgs or map it in pypiToNixAttr";

      resolve =
        seen: extra:
        if builtins.elem extra seen then
          throw "camas resolve-extras: cyclic self-reference in optional-dependencies: ${
            lib.concatStringsSep " -> " (seen ++ [ extra ])
          }"
        else
          lib.unique (
            lib.concatMap (
              spec:
              let
                parsed = parseSpec spec;
              in
              if parsed ? self then
                lib.concatMap (resolve (seen ++ [ extra ])) parsed.self
              else
                [ (resolvePkg extra parsed.pkg) ]
            ) pyprojectExtras.${extra}
          );
    in
    resolve [ ];
in
{
  inherit parseSpec mkResolveExtra;
}
