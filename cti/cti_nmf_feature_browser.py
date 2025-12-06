"""
Author: Tiago
Date: 2025-12-04
Description: NMF feature browser wrapper. Preserves CLI compatibility and delegates to cti_nmf_routines.
"""

from pathlib import Path
import argparse
import numpy as np
import polars as pl
from cti_nmf_routines import (
    load_run_vectors,
    run_nmf_feature_browser,
    write_top_corners_per_feature,
    plot_feature_runs,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-vectors", type=Path, default=Path("Final_Project/cti_data/run_vectors.npy"))
    ap.add_argument("--corners", type=Path, default=Path("Final_Project/cti_data/corners_dataset.parquet"))
    ap.add_argument("--outdir", type=Path, default=Path("Final_Project/cti_data/nmf_features"),
                    help="Directory for NMF arrays and CSV (data artifacts)")
    ap.add_argument("--plots-outdir", type=Path, default=Path("Final_Project/cti_outputs/nmf_features"),
                    help="Directory for PNG plots (figures)")
    ap.add_argument("--components", type=int, default=30)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plots", action="store_true", help="emit per-feature bar plots for top run types")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.plots_outdir.mkdir(parents=True, exist_ok=True)
    runs = load_run_vectors(args.run_vectors)
    X = np.maximum(runs.T.copy(), 0.0)
    W, H, err = run_nmf_feature_browser(X, k=args.components, seed=args.seed)
    np.save(args.outdir / "nmf_W_42xK.npy", W)
    np.save(args.outdir / "nmf_H_KxN.npy", H)
    (args.outdir / "nmf_reconstruction_err.txt").write_text(f"{err:.6f}\n", encoding="utf-8")

    corners_df = pl.read_parquet(args.corners)
    write_top_corners_per_feature(H, corners_df, args.outdir / "top_corners_per_feature.csv", top=args.top)

    if args.plots:
        plot_feature_runs(W, args.plots_outdir, top_m=8)

    print("Complete.")


if __name__ == "__main__":
    main()
