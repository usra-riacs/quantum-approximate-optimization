# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

_storage_directory_configured = os.getenv('DEFAULT_STORAGE_DIRECTORY')

if _storage_directory_configured is None or not _storage_directory_configured.strip():
    # Nothing configured: one "output" directory for the whole installation, at the package
    # root -- the directory that holds `quapopt`, which is the repository root for a source
    # checkout. Following the current working directory instead would scatter a new store
    # under every folder a notebook or script happens to be started from.
    DEFAULT_STORAGE_DIRECTORY = Path(__file__).resolve().parents[4] / "output"
else:
    DEFAULT_STORAGE_DIRECTORY = Path(_storage_directory_configured)

DEFAULT_STORAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)