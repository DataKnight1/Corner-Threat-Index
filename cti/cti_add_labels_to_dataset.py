"""
Author: Tiago
Date: 2025-12-04
Description: Add labels (y1–y5) to the corners dataset using tracking and events data. Uses the improved, domain-driven labeling approach to produce interpretable targets for each corner.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from cti_labels_improved import (
    build_corner_xthreat_model,
    compute_improved_labels
)
from cti_corner_extraction import load_events_basic, load_tracking_full
from cti_gmm_zones import build_player_team_map


def add_labels_to_corners_dataset(
    corners_df: pl.DataFrame,
    events_dict: Dict[int, pl.DataFrame],
    tracking_dict: Dict[int, pl.DataFrame],
    xt_surface: np.ndarray,
    use_improved_labels: bool = True,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Add target labels (y1–y5) to the corners dataset using the improved labeling strategy.

    :param corners_df: Corners metadata.
    :param events_dict: Mapping match_id -> events DataFrame.
    :param tracking_dict: Mapping match_id -> tracking DataFrame.
    :param xt_surface: Expected threat surface (12x8 grid).
    :param use_improved_labels: If True, use domain-driven improved labels; otherwise fallback.
    :param verbose: If True, print progress.
    :return: DataFrame with appended label columns (y1_label..y5_label).
    """
    if verbose:
        print(f"\n[IMPROVED LABELS] Computing labels for {corners_df.height} corners...")
        print(f"  Strategy: {'Domain-driven (IMPROVED)' if use_improved_labels else 'Original xG-based'}")

    if use_improved_labels:
        if verbose:
            print("\n[Phase 1/2] Building historical corner xThreat model...")
        xthreat_model = build_corner_xthreat_model(corners_df, events_dict)
    else:
        xthreat_model = {}

    if verbose:
        print(f"\n[Phase 2/2] Computing labels for each corner...")

    results = []

    for corner in tqdm(corners_df.iter_rows(named=True), total=corners_df.height,
                       desc="Computing labels", disable=not verbose):
        match_id = corner["match_id"]

        if match_id not in events_dict or match_id not in tracking_dict:
            results.append({
                **corner,
                "y1_label": np.nan,
                "y2_label": np.nan,
                "y3_label": np.nan,
                "y4_label": np.nan,
                "y5_label": np.nan
            })
            continue

        events_df = events_dict[match_id]
        tracking_df = tracking_dict[match_id]

        # Convert frame_start to timestamp (assume 25 fps)
        FPS = 25.0
        frame_start = corner['frame_start']
        period = corner['period']

        # Estimate timestamp from frame number
        # Each period is ~45 minutes, frames reset per period
        timestamp = frame_start / FPS  # Convert frames to seconds

        # Try to find matching corner event for more accurate timestamp
        # Corner events are identified by start_type_id (11=corner_reception, 12=corner_interception)
        corner_events = events_df.filter(
            (pl.col('start_type_id').is_not_null()) &
            (pl.col('start_type_id').is_in([11, 12])) &
            (pl.col('period') == period) &
            (pl.col('frame_start') == frame_start)  # Match exact frame
        )

        if len(corner_events) > 0:
            # Use event timestamp if found (more accurate)
            event_timestamp = corner_events.row(0, named=True).get('time_start', timestamp)
            if isinstance(event_timestamp, (int, float)):
                timestamp = float(event_timestamp)
            elif isinstance(event_timestamp, str):
                # Parse time string "MM:SS.ms" to seconds
                try:
                    parts = event_timestamp.split(':')
                    if len(parts) == 2:
                        minutes = float(parts[0])
                        seconds = float(parts[1])
                        timestamp = minutes * 60.0 + seconds
                except:
                    pass  # Keep frame-based timestamp

        corner_event = {
            "frame_start": corner["frame_start"],
            "period": corner["period"],
            "match_id": match_id,
            "team_id": corner["team_id"],
            "x_start": corner.get("x_start", 0.0),
            "y_start": corner.get("y_start", 0.0),
            "timestamp": timestamp
        }

        try:
            if use_improved_labels:
                # Use improved labeling strategy
                targets = compute_improved_labels(
                    corner_event,
                    events_df,
                    tracking_df,
                    xt_surface,
                    xthreat_model
                )
            else:
                # Fallback to original method (not recommended)
                from cti_integration import extract_targets
                team_maps = {match_id: build_player_team_map(events_df)}
                targets = extract_targets(
                    corner_event,
                    events_df,
                    xt_surface,
                    tracking_df=tracking_df,
                    player_team_map=team_maps.get(match_id, None),
                    use_tracking_counter=True
                )

            results.append({
                **corner,
                "y1_label": targets["y1"],
                "y2_label": targets["y2"],
                "y3_label": targets["y3"],
                "y4_label": targets["y4"],
                "y5_label": targets["y5"]
            })

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to compute labels for corner {corner.get('corner_id')}: {e}")
            results.append({
                **corner,
                "y1_label": np.nan,
                "y2_label": np.nan,
                "y3_label": np.nan,
                "y4_label": np.nan,
                "y5_label": np.nan
            })

    # Create new DataFrame with labels
    labeled_df = pl.from_dicts(results)

    if verbose:
        # Print statistics
        n_valid = labeled_df.filter(pl.col("y1_label").is_not_nan()).height
        print(f"\nOK Computed labels for {n_valid}/{corners_df.height} corners")

        if n_valid > 0:
            print("\nLabel Statistics:")
            for col in ["y1_label", "y2_label", "y3_label", "y4_label", "y5_label"]:
                values = labeled_df[col].drop_nans()
                if len(values) > 0:
                    print(f"  {col}: mean={float(values.mean()):.4f}, "
                          f"min={float(values.min()):.4f}, max={float(values.max()):.4f}, "
                          f"nonzero={int((values > 0).sum())}/{len(values)}")

    return labeled_df


def main():
    """
    Standalone entry point to add labels to an existing corners dataset.
    Loads corners, xT surface, events, and tracking, computes labels, and writes
    a labeled parquet file (and updates the main dataset).
    """
    import argparse
    import pickle
    from cti_paths import FINAL_PROJECT_DIR, DATA_2024

    parser = argparse.ArgumentParser(description="Add labels to corners dataset")
    parser.add_argument("--input", type=Path,
                       default=FINAL_PROJECT_DIR / "cti_data" / "corners_dataset.parquet",
                       help="Input corners dataset (without labels)")
    parser.add_argument("--output", type=Path,
                       default=FINAL_PROJECT_DIR / "cti_data" / "corners_dataset_labeled.parquet",
                       help="Output path for labeled dataset")
    parser.add_argument("--xt-surface", type=Path,
                       default=FINAL_PROJECT_DIR / "cti_data" / "xt_surface.pkl",
                       help="Path to xT surface pickle")
    parser.add_argument("--no-tracking-counter", action="store_true",
                       help="Disable tracking-based counter risk (use events only)")
    args = parser.parse_args()

    # Load corners dataset
    print(f"Loading corners from {args.input}...")
    corners_df = pl.read_parquet(args.input)
    print(f"  Loaded {corners_df.height} corners from {corners_df['match_id'].n_unique()} matches")

    # Load xT surface
    print(f"\nLoading xT surface from {args.xt_surface}...")
    with open(args.xt_surface, "rb") as f:
        xt_surface = pickle.load(f)
    print(f"  xT surface shape: {xt_surface.shape}")

    # Get unique match IDs
    match_ids = corners_df["match_id"].unique().to_list()

    # Load events and tracking
    print(f"\nLoading events and tracking for {len(match_ids)} matches...")
    events_dict = {}
    tracking_dict = {}

    for match_id in tqdm(match_ids):
        try:
            events = load_events_basic(match_id)
            tracking = load_tracking_full(match_id, sort_rows=False)

            if events.height > 0 and tracking.height > 0:
                events_dict[match_id] = events
                tracking_dict[match_id] = tracking
        except Exception as e:
            print(f"  Warning: Failed to load match {match_id}: {e}")

    print(f"OK Loaded {len(events_dict)} matches")

    # Add labels
    labeled_df = add_labels_to_corners_dataset(
        corners_df,
        events_dict,
        tracking_dict,
        xt_surface,
        use_tracking_counter=not args.no_tracking_counter,
        verbose=True
    )

    # Save
    print(f"\nSaving labeled dataset to {args.output}...")
    labeled_df.write_parquet(args.output)
    print(f"OK Saved {labeled_df.height} corners with labels")

    # Also update the main dataset file
    main_dataset = FINAL_PROJECT_DIR / "cti_data" / "corners_dataset.parquet"
    if args.output != main_dataset:
        print(f"\nUpdating main dataset: {main_dataset}")
        labeled_df.write_parquet(main_dataset)
        print("OK Updated")


if __name__ == "__main__":
    main()
