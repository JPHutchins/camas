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
      # `dhall`'s only binding (s-zeng/dhall-python) is absent from nixpkgs' python3Packages, so it
      # can't be resolved by nix/resolve-extras.nix; drop it from the nix extra matrix (it is not in
      # `all` either). The camas[dhall] path is exercised on CPython, where the wheel/sdist builds.
      nixExtras = builtins.filter (extra: extra != "dhall") extraNames;
      perExtra = builtins.filter (extra: extra != "all") nixExtras;
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
              realExtras = (lib.importTOML ./pyproject.toml).project.optional-dependencies;
              resolveNames =
                pyprojectExtras: python3Packages: extra:
                map (drv: drv.name) (resolver.mkResolveExtra { inherit python3Packages pyprojectExtras; } extra);
              forced = value: builtins.tryEval (builtins.deepSeq value value);
              fakePackages = {
                httpx = {
                  name = "httpx";
                };
                msgspec = {
                  name = "msgspec";
                };
              };
              # Resolve every nix-buildable extra (nixExtras drops `dhall`, which has no
              # python3Packages attr — see the nixExtras definition above).
              realResolved = lib.genAttrs nixExtras (extra: resolveNames realExtras pkgs.python3Packages extra);
              realResolves = (forced realResolved).success;
              cyclicFails =
                !(forced (
                  resolveNames {
                    a = [ "camas[b]" ];
                    b = [ "camas[a]" ];
                  } fakePackages "a"
                )).success;
              missingFails =
                !(forced (
                  resolveNames {
                    x = [ "definitely-absent-pkg" ];
                  } fakePackages "x"
                )).success;
              unparseableFails =
                !(forced (
                  resolveNames {
                    y = [ "@vcs+https://example/x" ];
                  } fakePackages "y"
                )).success;
            in
            assert realResolves;
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
