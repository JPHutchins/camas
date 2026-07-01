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

      version = "0.0.0+${self.shortRev or self.dirtyShortRev or "unknown"}";

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
                nixfmt --check ${./flake.nix} ${./nix/package.nix}
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
