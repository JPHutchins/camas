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

      version = "0.0.0+${self.shortRev or self.dirtyShortRev or "unknown"}";

      # argtree isn't in nixpkgs yet; add it to every python package set so
      # nix/package.nix can depend on it as plain `python3Packages.argtree`
      # (the form a future nixpkgs build will use). Drop once it lands upstream.
      argtreeOverlay = final: prev: {
        pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
          (pyfinal: pyprev: {
            argtree = pyfinal.callPackage ./nix/argtree.nix { };
          })
        ];
      };

      pkgsFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [ argtreeOverlay ];
        };

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
          pkgs = pkgsFor system;
        in
        {
          default = mkCamas pkgs { };
          with-github-checks = mkCamas pkgs { extras = [ "github_checks" ]; };
          with-check = mkCamas pkgs { extras = [ "check" ]; };
          all = mkCamas pkgs {
            extras = [
              "github_checks"
              "check"
            ];
          };
          interpreted = mkCamas pkgs { withMypyC = false; };
        }
      );

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/camas";
        };
        with-github-checks = {
          type = "app";
          program = "${self.packages.${system}.with-github-checks}/bin/camas";
        };
        with-check = {
          type = "app";
          program = "${self.packages.${system}.with-check}/bin/camas";
        };
        all = {
          type = "app";
          program = "${self.packages.${system}.all}/bin/camas";
        };
        interpreted = {
          type = "app";
          program = "${self.packages.${system}.interpreted}/bin/camas";
        };
      });

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
                nixfmt --check ${./flake.nix} ${./nix/package.nix} ${./nix/argtree.nix}
                touch $out
              '';

          package = self.packages.${system}.default;
          package-with-github-checks = self.packages.${system}.with-github-checks;
          package-with-check = self.packages.${system}.with-check;
          package-all = self.packages.${system}.all;
          package-interpreted = self.packages.${system}.interpreted;
        }
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
