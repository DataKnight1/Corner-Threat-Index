"""
Author: Tiago
Date: 2025-12-04
Description: Regenerate the team_top_feature table/PNG using team names and logos.
"""

from pathlib import Path
import polars as pl
from cti_team_mapping import build_team_name_map
from cti_nmf_routines import save_team_top_feature_table

from cti_paths import FINAL_PROJECT_DIR, DATA_2024

# Paths
DATA_DIR = DATA_2024
OUTPUT_IMG_DIR = FINAL_PROJECT_DIR / "cti_outputs"
DATA_OUT_DIR = FINAL_PROJECT_DIR / "cti_data"

# Load existing CSV
csv_path = DATA_OUT_DIR / "team_top_feature.csv"
if not csv_path.exists():
    print(f"Error: {csv_path} not found")
    print("Please run the full pipeline first to generate the CSV")
    exit(1)

# Read the CSV
table_df = pl.read_csv(csv_path)
print(f"Loaded table with {table_df.height} teams")
print(f"Columns: {table_df.columns}")

# Build team name map
meta_dir = DATA_DIR / 'meta'
team_name_map = build_team_name_map(meta_dir, use_fallback=True)

# Update team column with actual names if it has team IDs
if 'team' in table_df.columns:
    team_values = table_df['team'].to_list()

    # Check if we need to convert IDs to names (first value is numeric)
    first_value = team_values[0]
    needs_conversion = False
    try:
        int(first_value)
        needs_conversion = True
    except (ValueError, TypeError):
        # Already has team names
        print("Table already has team names, no conversion needed")

    if needs_conversion:
        # Convert team IDs to names
        team_names = [team_name_map.get(int(tid), str(tid)) for tid in team_values]
        table_df = table_df.with_columns([
            pl.Series('team', team_names)
        ])
        print("\nUpdated team names:")
        for tid, name in zip(team_values, team_names):
            print(f"  {tid} -> {name}")
else:
    print("Warning: 'team' column not found in CSV")

# Set logo directory
from cti_paths import ASSETS_DIR as _ASSETS
logo_dir = _ASSETS
if not logo_dir.exists():
    print(f"Warning: Logo directory not found: {logo_dir}")
    logo_dir = None

# Save updated table
save_team_top_feature_table(
    table_df,
    DATA_OUT_DIR / "team_top_feature.csv",
    OUTPUT_IMG_DIR / "team_top_feature.png",
    logo_dir=logo_dir
)

print(f"\nOK Updated table saved to:")
print(f"  CSV: {DATA_OUT_DIR / 'team_top_feature.csv'}")
print(f"  PNG: {OUTPUT_IMG_DIR / 'team_top_feature.png'}")
