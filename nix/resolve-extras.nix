# Pure extras resolver extracted from package.nix so its guards are exercisable
# from the flake `extras-resolver` check. It maps each pyproject
# [project.optional-dependencies] spec and [dependency-groups] entry to a nixpkgs
# python3Packages derivation, following `camas[...]` self-references and
# `{ include-group = ... }` tables, and throws a readable message on the five
# edges that would otherwise fail with an opaque eval trace or loop forever:
#   1. a PyPI name whose nixpkgs attr differs (map it in pypiToNixAttr),
#   2. a package absent from python3Packages,
#   3. a mutually self-referential extra or group pair (cycle),
#   4. a spec that is neither a parseable PEP 508 name nor an include-group
#      table holding only that key, and
#   5. an include-group naming no [dependency-groups] entry.
# It resolves names only: version specifiers and environment markers are not
# evaluated, so a pin is whatever nixpkgs ships.
{
  lib,
  pname,
}:
let
  parseSpec =
    spec:
    if builtins.isAttrs spec && builtins.attrNames spec == [ "include-group" ] then
      { group = spec.include-group; }
    else if !builtins.isString spec then
      throw "camas resolve-extras: cannot parse requirement ${builtins.toJSON spec}; expected a PEP 508 string or a table holding only include-group"
    else if lib.hasPrefix "${pname}[" spec then
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

  mkResolver =
    {
      python3Packages,
      pyprojectExtras,
      pyprojectGroups ? { },
      pypiToNixAttr ? { },
    }:
    let
      resolvePkg =
        origin: name:
        let
          attr = pypiToNixAttr.${name} or name;
        in
        if python3Packages ? ${attr} then
          python3Packages.${attr}
        else
          throw "camas resolve-extras: ${origin} requires Python package '${name}' (nixpkgs attr '${attr}') which is absent from python3Packages; add it to nixpkgs or map it in pypiToNixAttr";

      resolve =
        seen: origin: specs:
        if builtins.elem origin seen then
          throw "camas resolve-extras: cyclic reference: ${lib.concatStringsSep " -> " (seen ++ [ origin ])}"
        else
          lib.unique (
            lib.concatMap (
              spec:
              let
                parsed = parseSpec spec;
              in
              if parsed ? self then
                lib.concatMap (resolveExtra (seen ++ [ origin ])) parsed.self
              else if parsed ? group then
                resolveGroup (seen ++ [ origin ]) parsed.group
              else
                [ (resolvePkg origin parsed.pkg) ]
            ) specs
          );

      resolveExtra = seen: extra: resolve seen "extra '${extra}'" pyprojectExtras.${extra};

      resolveGroup =
        seen: group:
        if pyprojectGroups ? ${group} then
          resolve seen "dependency group '${group}'" pyprojectGroups.${group}
        else
          throw "camas resolve-extras: dependency group '${group}' is not in [dependency-groups]";
    in
    {
      extra = resolveExtra [ ];
      group = resolveGroup [ ];
    };
in
{
  inherit parseSpec;
  mkResolveExtra = args: (mkResolver args).extra;
  mkResolveGroup = args: (mkResolver args).group;
}
