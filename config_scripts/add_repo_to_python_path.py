# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

# Makes this repository importable in the Python environment that runs the script:
# a one-line .pth file holding the repository root goes into the environment's site-packages,
# so scripts and Jupyter kernels of that environment import quapopt from this repository.
# Run it once per pixi environment, and again after the environment is reinstalled.

import os
import sysconfig

pth_file_name = "quapopt_repo.pth"

project_root_abs = os.path.dirname(os.path.abspath(__file__))
project_root_abs = os.path.dirname(project_root_abs)

site_packages = sysconfig.get_paths()["purelib"]
pth_path = os.path.join(site_packages, pth_file_name)

with open(pth_path, "w") as f:
    f.write(f"{project_root_abs}\n")

print(f"✅ Success! {project_root_abs} is now on the Python path of this environment.")
print(f"This has been saved to the '{pth_path}' file.")
