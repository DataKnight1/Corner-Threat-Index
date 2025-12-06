"""
Author: Tiago
Date: 2025-12-04
Description: Run NMF topic modelling on 42-d run vectors and emit figures like Paper §3.2 (features grid) and Figure 4 (top corners for a feature).
"""

from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from cti_corner_extraction import load_events_basic, load_tracking_full
from cti_gmm_zones import load_zone_models, encode_all_corners
from cti_nmf_routines import (
    fit_nmf_routines,
    save_nmf_model,
    build_corner_samples_for_visuals,
    visualize_feature_grid,
    visualize_top_corners_for_feature,
)


ROOT = Path(__file__).parent
OUT = ROOT / "cti_outputs"  # figures
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "cti_data"     # data artifacts
DATA.mkdir(parents=True, exist_ok=True)


def main(feature: int = 12):
    corners_path = DATA / "corners_dataset.parquet"
    zones_path = DATA / "gmm_zones.pkl"
    runs_path = DATA / "run_vectors.npy"

    if not corners_path.exists():
        print(f"Error: {corners_path} not found. Run extraction first.")
        return

    corners_df = pl.read_parquet(corners_path)
    print(f"OK Loaded {corners_df.height} corners")

    if not zones_path.exists():
        print(f"Error: {zones_path} not found. Fit GMM zones first.")
        return
    zones = load_zone_models(zones_path)

    # Load or build run vectors
    if runs_path.exists():
        runs = np.load(runs_path)
        print(f"OK Loaded run vectors: {runs.shape}")
    else:
        print("Computing run vectors...")
        match_ids = corners_df["match_id"].unique().to_list()
        events_dict = {}
        tracking_dict = {}
        for mid in match_ids:
            try:
                events_dict[mid] = load_events_basic(mid)
                tracking_dict[mid] = load_tracking_full(mid, sort_rows=False)
            except Exception as exc:
                print(f"  Warning: failed to load match {mid}: {exc}")
        runs = encode_all_corners(corners_df, tracking_dict, events_dict, zones, verbose=True)
        np.save(runs_path, runs)
        print(f"OK Saved run vectors to {runs_path}")

    # Fit NMF with up to 30 topics (bounded by sample size)
    n_components = min(30, runs.shape[0], runs.shape[1])
    nmf = fit_nmf_routines(runs, n_components=n_components)
    save_nmf_model(nmf, DATA / "nmf_model.pkl")

    # Build corner samples for visualization
    print("Building corner samples for visuals...")
    match_ids = corners_df["match_id"].unique().to_list()
    events_dict = {mid: load_events_basic(mid) for mid in match_ids}
    tracking_dict = {mid: load_tracking_full(mid, sort_rows=False) for mid in match_ids}
    samples = build_corner_samples_for_visuals(corners_df, events_dict, tracking_dict)
    print(f"OK Prepared {len(samples)} corner samples")

    # Figure 3: feature grid
    fig_grid = visualize_feature_grid(nmf, zones)
    fig_grid.savefig(OUT / "nmf_features_grid.png", dpi=180, bbox_inches='tight')
    plt.close(fig_grid)
    print(f"OK Saved {OUT / 'nmf_features_grid.png'}")

    # Figure 4: top corners for a feature (default 12)
    fid = max(0, min(feature - 1, nmf.n_topics - 1))  # 1-based -> 0-based
    fig_feat = visualize_top_corners_for_feature(fid, nmf, samples, zones, top_n=10)
    fig_feat.savefig(OUT / f"feature_{feature}_top_corners.png", dpi=180, bbox_inches='tight')
    plt.close(fig_feat)
    print(f"OK Saved {OUT / f'feature_{feature}_top_corners.png'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", type=int, default=12)
    args = ap.parse_args()
    main(feature=args.feature)
