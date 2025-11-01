# # Copyright 2025 USRA
# # Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import glob
import os
import platform
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy, sysconfig, sys
# --- Configuration ---
# The name of the main package directory, relative to the *parent* of setup.py
PACKAGE_NAME = "quapopt"

# --- Calculate Paths ---
# Get the absolute path to the directory containing this setup.py file
PROJECT_ROOT = os.path.dirname(__file__)
PKG_DIR = os.path.join(PROJECT_ROOT, PACKAGE_NAME)



print(f"Project Root: {PROJECT_ROOT}")
print(f"Package Directory: {PKG_DIR}")

# --- Find Cython files ---
# Search recursively for .pyx files starting from the package directory
pyx_files = glob.glob(os.path.join(PKG_DIR, "**", "*.pyx"), recursive=True)





print("________________________________________")
print(f"Searching for .pyx files")
print(f"Found {len(pyx_files)} .pyx files:")
for f in pyx_files:
    # Print path relative to project root for clarity
    print(f"- {os.path.relpath(f, PROJECT_ROOT)}")
print("________________________________________")

if not pyx_files:
    print("Warning: No .pyx files found. Check the PACKAGE_NAME and file locations.")

# --- Platform-specific compiler configuration ---
system = platform.system()
machine = platform.machine()
print(f"\nDetected platform: {system} ({machine})")

# Configure compiler flags based on platform
extra_compile_args = []
extra_link_args = []

if system.lower() == "darwin":  # macOS
    # macOS uses clang, which needs specific C++ standard flags
    extra_compile_args = [
        "-stdlib=libc++",       # Use libc++ standard library (clang default)
    ]
    extra_link_args = [
        "-stdlib=libc++",       # Link with libc++
    ]

    # For Apple Silicon (M1/M2/M3), we might need additional flags
    if machine == "arm64":
        print("Detected Apple Silicon (ARM64)")
        # Ensure we're building for the native architecture
        extra_compile_args.append("-arch")
        extra_compile_args.append("arm64")
        extra_link_args.append("-arch")
        extra_link_args.append("arm64")
        # Set minimum macOS version for Apple Silicon
        extra_compile_args.append("-mmacosx-version-min=11.0")
    else:
        # Intel Mac
        extra_compile_args.append("-mmacosx-version-min=10.9")

    print("Using macOS-specific compiler flags (clang with libc++)")
else:
    print(f"Using default compiler flags")



# --- Create Extension objects ---
extensions = []
for path_abs in pyx_files:
    # 1. Get the path relative to the *package* directory
    rel_path = os.path.relpath(path_abs, PROJECT_ROOT).replace(os.sep, "/")

    # 2. Remove the file extension (.pyx)
    module_path = os.path.splitext(rel_path)[0]
    module_name = module_path.replace("/", ".")

    print(f"Creating Extension for path: {rel_path} -> module name: {module_name}")

    extensions.append(
        Extension(
            module_name,
            [rel_path],
            include_dirs=[numpy.get_include(),
                          sysconfig.get_paths()["include"]
                          ],
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    )






print("\n==============Created extensions, running setup...==============\n")
# --- Run setup ---
setup(
    name=PACKAGE_NAME,
    packages=find_packages(include=[PACKAGE_NAME, f"{PACKAGE_NAME}.*"]),
    package_data = {f"{PACKAGE_NAME}.additional_packages.ancillary_functions_usra": ["*.py"]},
    include_package_data=True,
    version="0.2.0",
    # Include the compiled Cython extensions
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"},
    ),
)

print("___________________")
print("\nBasic setup script finished.")












