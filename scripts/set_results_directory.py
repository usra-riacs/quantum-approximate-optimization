# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import os
# The file to store the environment variable
env_file = ".env"
var_name = "DEFAULT_STORAGE_DIRECTORY"

# Suggest a default value
project_root_abs = os.path.dirname(os.path.abspath(__file__))
project_root_abs = os.path.dirname(project_root_abs)
default_path = os.path.join(project_root_abs, "output")

print(f"The current default results folder will be set to: {default_path}")
user_input = input(f"Press ENTER to accept or specify a new path: ").strip()

# Use the user's input or the default
final_path = user_input if user_input else default_path

# Create the directory if it doesn't exist
os.makedirs(final_path, exist_ok=True)

# Write the variable to the .env file
with open(env_file, "w") as f:
    f.write(f"{var_name}={final_path}\n")

print(f"✅ Success! Results folder set to: {final_path}")
print(f"This has been saved to the '{env_file}' file.")
