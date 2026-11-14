
{
  description = "Python Magpylib development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };

          python = pkgs.python312;
          pythonPackages = pkgs.python312Packages;
        in
        {
          default = pkgs.mkShell {
            venvDir = ".venv";

            packages = with pkgs; [
              python
              pythonPackages.venvShellHook

              # Python development
              ruff
              basedpyright

              # Native dependencies
              stdenv.cc.cc.lib
              zlib
              libGL
              glib
            ];

            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
              pip install -r requirements.txt
            '';

            postShellHook = ''
              unset SOURCE_DATE_EPOCH

              export LD_LIBRARY_PATH="${
                pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc.lib
                  pkgs.zlib
                  pkgs.libGL
                  pkgs.glib
                ]
              }:$LD_LIBRARY_PATH"

              echo "Python development environment ready!"
              echo "Python: $(python --version)"
              echo "Venv: $VIRTUAL_ENV"
            '';
          };
        });
    };
}
