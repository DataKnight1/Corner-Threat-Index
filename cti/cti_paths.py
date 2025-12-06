"""
Author: Tiago
Date: 2025-12-04
Description: Centralized paths for the CTI project. Resolves repository root and commonly used directories for data, outputs, and assets to keep imports consistent.
"""

from __future__ import annotations

from pathlib import Path

# Resolve repository root (…/twelve-deep-learning). This file lives at
# Final_Project/cti/cti_paths.py ⇒ parents[2] is the repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Key dirs
FINAL_PROJECT_DIR: Path = REPO_ROOT / "Final_Project"
DATA_2024: Path = REPO_ROOT / "PremierLeague_data" / "2024"

# Common subdirs
DYNAMIC_DIR: Path = DATA_2024 / "dynamic"
TRACKING_DIR: Path = DATA_2024 / "tracking"
META_DIR: Path = DATA_2024 / "meta"

OUTPUT_DIR: Path = FINAL_PROJECT_DIR / "cti_outputs"
DATA_OUT_DIR: Path = FINAL_PROJECT_DIR / "cti_data"

ASSETS_DIR: Path = FINAL_PROJECT_DIR / "assets"
