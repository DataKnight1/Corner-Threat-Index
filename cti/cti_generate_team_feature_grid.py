"""
Author: Tiago
Date: 2025-12-04
Description: Generate a grid of the 20 teams showing, for each team, its crest and the visualization of its top NMF feature (topic) on a half pitch.
"""

from pathlib import Path
from cti_paths import FINAL_PROJECT_DIR
import pickle
import polars as pl

from cti_nmf_routines import (
    load_nmf_model,
    visualize_team_top_features_grid,
)


OUTPUT_DIR = FINAL_PROJECT_DIR / "cti_outputs"
DATA_DIR = FINAL_PROJECT_DIR / "cti_data"
ASSETS_DIR = FINAL_PROJECT_DIR / "assets"


def main():
    csv_path = DATA_DIR / "team_top_feature.csv"
    nmf_path = DATA_DIR / "nmf_model.pkl"
    zones_path = DATA_DIR / "gmm_zones.pkl"

    if not csv_path.exists():
        raise SystemExit(f"Missing table: {csv_path}")
    if not nmf_path.exists():
        raise SystemExit(f"Missing NMF model: {nmf_path}")
    if not zones_path.exists():
        raise SystemExit(f"Missing GMM zones: {zones_path}")

    table_df = pl.read_csv(csv_path)
    nmf_model = load_nmf_model(nmf_path)
    with open(zones_path, 'rb') as f:
        zone_models = pickle.load(f)

    fig = visualize_team_top_features_grid(
        table_df,
        nmf_model,
        zone_models,
        logo_dir=ASSETS_DIR,
        rows=4,
        cols=5,
        title='Top Corner Feature per Team'
    )

    out_img = OUTPUT_DIR / "team_top_features_grid.png"
    fig.savefig(out_img, dpi=170, bbox_inches='tight')
    print(f"OK Saved: {out_img}")


if __name__ == "__main__":
    main()
