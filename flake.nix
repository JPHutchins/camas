{
  description = "camas — task runner with parallel execution, matrix expansion, and pluggable output effects";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;

      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = lib.genAttrs systems;

      extraNames = builtins.attrNames (lib.importTOML ./pyproject.toml).project.optional-dependencies;
      perExtra = builtins.filter (extra: extra != "all") extraNames;
      withExtraName = extra: "with-${lib.replaceStrings [ "_" ] [ "-" ] extra}";

      # VERSION holds the released version; CI's version-gate asserts it matches
      # the git tag before publish. A clean checkout (a tagged fetch included)
      # reports it bare; a dirty tree appends the rev.
      baseVersion = lib.fileContents ./VERSION;
      version =
        if self ? shortRev then baseVersion else "${baseVersion}+${self.dirtyShortRev or "unknown"}";

      mkCamas =
        pkgs: args:
        pkgs.callPackage ./nix/package.nix (
          {
            inherit version;
            src = ./.;
          }
          // args
        );
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = mkCamas pkgs { };
          all = mkCamas pkgs { extras = [ "all" ]; };
          interpreted = mkCamas pkgs { withMypyC = false; };
        }
        // lib.listToAttrs (
          map (extra: lib.nameValuePair (withExtraName extra) (mkCamas pkgs { extras = [ extra ]; })) perExtra
        )
      );

      apps = forAllSystems (
        system:
        lib.mapAttrs (_: pkg: {
          type = "app";
          program = "${pkg}/bin/camas";
        }) self.packages.${system}
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.uv
              pkgs.python3
            ];

            env.UV_PYTHON_DOWNLOADS = "never";

            shellHook = ''
              unset PYTHONPATH
            '';
          };
        }
      );

      checks = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          nixfmt =
            pkgs.runCommand "nixfmt-check"
              {
                nativeBuildInputs = [ pkgs.nixfmt ];
              }
              ''
                nixfmt --check ${./flake.nix} ${./nix/package.nix} ${./nix/resolve-extras.nix}
                touch $out
              '';

          extras-resolver =
            let
              resolver = import ./nix/resolve-extras.nix {
                inherit lib;
                pname = "camas";
              };
              pyproject = lib.importTOML ./pyproject.toml;
              realExtras = pyproject.project.optional-dependencies;
              realGroups = pyproject.dependency-groups;
              resolverArgs =
                {
                  extras ? realExtras,
                  groups ? realGroups,
                  packages ? pkgs.python3Packages,
                }:
                {
                  python3Packages = packages;
                  pyprojectExtras = extras;
                  pyprojectGroups = groups;
                };
              names = map (drv: drv.name);
              resolveNames = args: extra: names (resolver.mkResolveExtra (resolverArgs args) extra);
              resolveGroupNames = args: group: names (resolver.mkResolveGroup (resolverArgs args) group);
              forced = value: builtins.tryEval (builtins.deepSeq value value);
              fails = value: !(forced value).success;
              fakePackages = {
                httpx = {
                  name = "httpx";
                };
                msgspec = {
                  name = "msgspec";
                };
              };
              realResolves = !fails (builtins.mapAttrs (extra: _: resolveNames { } extra) realExtras);
              realTestGroupResolves =
                builtins.length (resolveGroupNames { } "test") == builtins.length realGroups.test;
              includeGroupFollowed =
                resolveGroupNames {
                  groups = {
                    a = [
                      { include-group = "b"; }
                      "httpx"
                    ];
                    b = [ "msgspec" ];
                  };
                  packages = fakePackages;
                } "a" == [
                  "msgspec"
                  "httpx"
                ];
              groupReachesExtras =
                resolveGroupNames {
                  extras.x = [ "msgspec" ];
                  groups = {
                    g = [
                      { include-group = "h"; }
                      "camas[x]"
                    ];
                    h = [ "httpx" ];
                  };
                  packages = fakePackages;
                } "g" == [
                  "httpx"
                  "msgspec"
                ];
              groupCyclicFails = fails (
                resolveGroupNames {
                  groups = {
                    a = [ { include-group = "b"; } ];
                    b = [ { include-group = "a"; } ];
                  };
                  packages = fakePackages;
                } "a"
              );
              missingGroupFails = fails (
                resolveGroupNames {
                  groups.a = [ { include-group = "nope"; } ];
                  packages = fakePackages;
                } "a"
              );
              tableFails = fails (
                resolveGroupNames {
                  groups.t = [ { path = "x"; } ];
                  packages = fakePackages;
                } "t"
              );
              mixedTableFails = fails (
                resolveGroupNames {
                  groups = {
                    t = [
                      {
                        include-group = "u";
                        path = "x";
                      }
                    ];
                    u = [ "httpx" ];
                  };
                  packages = fakePackages;
                } "t"
              );
              cyclicFails = fails (
                resolveNames {
                  extras = {
                    a = [ "camas[b]" ];
                    b = [ "camas[a]" ];
                  };
                  packages = fakePackages;
                } "a"
              );
              missingFails = fails (
                resolveNames {
                  extras.x = [ "definitely-absent-pkg" ];
                  packages = fakePackages;
                } "x"
              );
              unparseableFails = fails (
                resolveNames {
                  extras.y = [ "@vcs+https://example/x" ];
                  packages = fakePackages;
                } "y"
              );
            in
            assert realResolves;
            assert realTestGroupResolves;
            assert includeGroupFollowed;
            assert groupReachesExtras;
            assert groupCyclicFails;
            assert missingGroupFails;
            assert tableFails;
            assert mixedTableFails;
            assert cyclicFails;
            assert missingFails;
            assert unparseableFails;
            pkgs.runCommand "extras-resolver-check" { } ''
              touch $out
            '';
        }
        // lib.mapAttrs' (
          name: pkg: lib.nameValuePair ("package" + lib.optionalString (name != "default") "-${name}") pkg
        ) self.packages.${system}
      );

      formatter = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        pkgs.writeShellApplication {
          name = "camas-nix-formatter";
          runtimeInputs = [ pkgs.nixfmt ];
          text = ''
            shopt -s globstar nullglob
            files=( ./**/*.nix )
            exec nixfmt "$@" "''${files[@]}"
          '';
        }
      );
    };
}
