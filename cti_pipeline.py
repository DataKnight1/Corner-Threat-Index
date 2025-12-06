"""
Author: Tiago Monteiro
Created: 2025-11-30
Last Modified: 2025-12-04
Description:
    End-to-end pipeline for the Corner Threat Index (CTI) system. Supports training,
    evaluation, and report generation. The default flow trains, runs inference,
    writes CSV/PNG outputs, and builds core visualizations.
"""

import argparse
import sys
import subprocess
from pathlib import Path
import numpy as np
import polars as pl
import numpy as np
from typing import Dict
import torch
from tqdm import tqdm

# Import all CTI modules
import sys as _sys_path_helper
_THIS_DIR = Path(__file__).parent
_CTI_DIR = _THIS_DIR / "cti"
if str(_CTI_DIR) not in _sys_path_helper.path:
    _sys_path_helper.path.insert(0, str(_CTI_DIR))

from cti_paths import REPO_ROOT, FINAL_PROJECT_DIR, DATA_2024
from cti_corner_extraction import (
    detect_corners,
    extract_all_corners_with_windows,
    save_corners_dataset
)
from cti_gmm_zones import (
    extract_initial_positions,
    extract_target_positions,
    fit_gmm_zones,
    build_player_team_map,
    encode_all_corners,
    save_zone_models,
    visualize_zones,
    audit_canonicalization_coverage,
    audit_right_half_occupancy,
    audit_takepoint_cluster
)
from cti_nmf_routines import (
    fit_nmf_routines,
    save_nmf_model,
    generate_team_routine_report,
    build_corner_samples_for_visuals,
    visualize_feature_grid,
    visualize_top_corners_for_feature,
    compute_team_top_feature_table,
    save_team_top_feature_table,
    save_team_cti_table,
)
from cti_team_mapping import build_team_name_map, PREMIER_LEAGUE_2024_TEAMS
from cti_add_labels_to_dataset import add_labels_to_corners_dataset
# Defensive roles stage intentionally disabled in this setup
# Use the half-pitch xT implementation (renamed module)
from cti_xt_surface_half_pitch import (
    classify_event_action,
    extract_action_sequences,
    discretize_position,
    build_action_counts,
    build_xT_matrices,
    value_iteration_vectorized as value_iteration_with_absorption,
    visualize_xt_surface as visualize_xt_surface_half
)


def build_xt_surface(events_df: pl.DataFrame, grid_shape=(12, 8)):
    """
    Build an xT grid using the half-pitch implementation.

    :param events_df: Events dataframe.
    :param grid_shape: Grid shape for xT computation.
    :return: xt_grid, n_iter, converged, n_actions.
    """
    # Extract action sequences from events
    actions = extract_action_sequences(events_df)

    # Build counts and matrices
    counts_shot, xg_sum, counts_loss, counts_move = build_action_counts(actions)
    P_shot, R_shot, P_loss, P_move = build_xT_matrices(
        counts_shot, xg_sum, counts_loss, counts_move
    )

    # Run value iteration
    xt_grid, n_iter, converged = value_iteration_with_absorption(
        P_shot, R_shot, P_loss, P_move
    )

    n_actions = int(actions.height) if hasattr(actions, 'height') else 0
    return xt_grid, n_iter, converged, n_actions


def save_xt_surface(xt_grid: np.ndarray, path: Path):
    """
    Persist the xT grid to disk.

    :param xt_grid: xT grid array.
    :param path: Output pickle path.
    """
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(xt_grid, f)
from cti_integration import (
    CornerGraphDataset,
    CTIMultiTaskModel,
    CTILightningModule,
    learn_lambda_gamma,
    compute_cti,
    compute_evaluation_metrics
)

# Configuration
ROOT = REPO_ROOT
# Raw input data (Premier League dataset)
RAW_DATA_DIR = DATA_2024
# PNG figures only
OUTPUT_DIR = FINAL_PROJECT_DIR / "cti_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
# Data artifacts (CSV, Parquet, PKL, NPY, TXT)
DATA_OUT_DIR = FINAL_PROJECT_DIR / "cti_data"
DATA_OUT_DIR.mkdir(exist_ok=True)


def generate_team_cti_table_v2(data_dir: Path, output_dir: Path, assets_dir: Path):
    """
    Build a goal-weighted CTI table (v2) from the existing team_cti_detailed.csv.

    :param data_dir: Directory containing team_cti_detailed.csv.
    :param output_dir: Directory to write the PNG table.
    :param assets_dir: Directory containing club logos.
    """
    table_path = data_dir / "team_cti_detailed.csv"
    if not table_path.exists():
        print(f"[POST] Skipping v2 CTI table (missing {table_path})")
        return

    df = pl.read_csv(table_path)

    if "cti_goal_weighted" in df.columns:
        cti_col = "cti_goal_weighted"
        title = "Team CTI (goal-weighted) v2"
    else:
        cti_col = "cti_avg"
        title = "Team CTI (baseline) v2"

    table = (
        df.select(
            [
                pl.col("team"),
                pl.col(cti_col).alias("cti_avg"),
                pl.col("y1_avg").alias("p_shot"),
                pl.col("counter_risk"),
                pl.col("y5_avg").alias("delta_xt"),
                pl.col("n_corners"),
            ]
        )
        .sort("cti_avg", descending=True)
    )

    out_csv = data_dir / "team_cti_v2.csv"
    # Save as the primary table filename (v2 content)
    out_png = output_dir / "team_cti_table.png"
    table.write_csv(out_csv)

    save_team_cti_table(
        table,
        out_csv=out_csv,
        out_png=out_png,
        title=title,
        logo_dir=assets_dir,
    )

    print(f"[POST] Wrote goal-weighted CTI table v2: {out_png}")


def _run_subprocess_py(script_path: Path, args: list[str]):
    """
    Run a Python script as a subprocess with the given arguments.

    :param script_path: Path to the Python script to run.
    :param args: List of command-line arguments to pass to the script.
    """
    cmd = [sys.executable, str(script_path), *args]
    print(f"\n[POST] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: post-processing step failed: {e}")


def load_data_for_matches(match_ids: list, verbose: bool = True):
    """
    Load events and tracking data for multiple matches.

    :param match_ids: List of match IDs to load.
    :param verbose: Whether to print progress and error messages.
    :return: Tuple of (events_dict, tracking_dict) mapping match IDs to DataFrames.
    """
    from cti_corner_extraction import load_events_basic, load_tracking_full

    events_dict = {}
    tracking_dict = {}

    if verbose:
        print(f"Loading data for {len(match_ids)} matches...")

    failed_matches = []
    for match_id in tqdm(match_ids, disable=not verbose):
        try:
            events = load_events_basic(match_id)
            tracking = load_tracking_full(match_id, sort_rows=False)

            if events.height > 0 and tracking.height > 0:
                events_dict[match_id] = events
                tracking_dict[match_id] = tracking
            else:
                failed_matches.append((match_id, "Empty data"))
        except Exception as e:
            failed_matches.append((match_id, str(e)))

    if verbose and len(failed_matches) > 0:
        print(f"\nFailed to load {len(failed_matches)} matches:")
        for match_id, error in failed_matches[:5]:  # Show first 5
            print(f"  Match {match_id}: {error}")
        if len(failed_matches) > 5:
            print(f"  ... and {len(failed_matches) - 5} more")

    if verbose:
        print(f"OK Loaded {len(events_dict)} matches successfully")

    return events_dict, tracking_dict


def phase1_extract_corners(match_ids: list, output_path: Path):
    """
    Phase 1: Extract all corners from matches and save to a dataset.

    :param match_ids: List of match IDs to process.
    :param output_path: Path to save the extracted corners dataset (parquet).
    :return: Tuple of (corners_df, events_dict, tracking_dict).
    """
    print("\n" + "="*60)
    print("PHASE 1: Corner Extraction & Windowing")
    print("="*60)

    events_dict, tracking_dict = load_data_for_matches(match_ids)

    all_corners = []

    for match_id in tqdm(match_ids, desc="Extracting corners"):
        if match_id not in events_dict:
            continue

        corners = extract_all_corners_with_windows(
            match_id,
            events_dict[match_id],
            tracking_dict[match_id],
            apply_gates=True,
            verbose=False
        )

        if corners.height > 0:
            all_corners.append(corners)

    # Combine all corners
    if len(all_corners) == 0:
        raise ValueError(
            "No corners detected in any matches! "
            "Check that your events data has 'start_type_id' column with values 11 or 12."
        )

    corners_df = pl.concat(all_corners, how="vertical_relaxed")

    # Save
    save_corners_dataset(corners_df, output_path)

    return corners_df, events_dict, tracking_dict


def phase2_fit_features(corners_df, events_dict, tracking_dict, output_dir: Path):
    """
    Phase 2: Perform feature engineering and model fitting (GMM, NMF, xT).

    :param corners_df: DataFrame containing corner events.
    :param events_dict: Dictionary mapping match IDs to event DataFrames.
    :param tracking_dict: Dictionary mapping match IDs to tracking DataFrames.
    :param output_dir: Directory to save fitted models and features.
    :return: Tuple of (corners_df, zone_models, nmf_model, xt_grid).
    """
    print("\n" + "="*60)
    print("PHASE 2: Feature Engineering & Model Fitting")
    print("="*60)

    # Step 1: GMM Zones
    print("\n[1/4] Fitting GMM zones...")
    corner_samples = []
    team_maps: Dict[int, Dict[int, int]] = {}
    new_rows = []

    for corner in tqdm(corners_df.iter_rows(named=True), total=corners_df.height,
                       desc="Extracting positions"):
        match_id = corner["match_id"]
        if match_id not in tracking_dict:
            continue

        events_df = events_dict[match_id]
        tracking_df = tracking_dict[match_id]
        team_map = team_maps.setdefault(match_id, build_player_team_map(events_df))

        corner_payload = dict(corner)
        corner_payload['player_team_map'] = team_map
        corner_payload['attacking_team_id'] = corner.get('team_id')
        corner_payload['taker_player_id'] = corner.get('player_in_possession_id')

        new_row = corner.copy()
        try:
            init_pos = extract_initial_positions(corner_payload, tracking_df, events_df)
            target_pos = extract_target_positions(corner_payload, tracking_df, events_df)
            
            if init_pos.height > 0:
                att_pos = init_pos.filter(pl.col('is_attacking'))
                def_pos = init_pos.filter(~pl.col('is_attacking'))
                new_row['att_x'] = att_pos['x_m'].to_list()
                new_row['att_y'] = att_pos['y_m'].to_list()
                new_row['def_x'] = def_pos['x_m'].to_list()
                new_row['def_y'] = def_pos['y_m'].to_list()
            else:
                new_row['att_x'] = []
                new_row['att_y'] = []
                new_row['def_x'] = []
                new_row['def_y'] = []

        except Exception:
            init_pos = pl.DataFrame()
            target_pos = pl.DataFrame()
            new_row['att_x'] = []
            new_row['att_y'] = []
            new_row['def_x'] = []
            new_row['def_y'] = []


        new_rows.append(new_row)

        if init_pos.height == 0 or target_pos.height == 0:
            continue

        corner_samples.append({'initial': init_pos.filter(pl.col('is_attacking')), 'target': target_pos})
    
    corners_df = pl.from_dicts(new_rows)

    if not corner_samples:
        raise ValueError('No corner samples available for GMM fitting')

    all_init = np.vstack([sample['initial'].select(["x_m", "y_m"]).to_numpy() for sample in corner_samples])
    all_target = np.vstack([sample['target'].select(["x_m", "y_m"]).to_numpy() for sample in corner_samples])

    zone_models = fit_gmm_zones(corner_samples, restrict_targets_to_pa=True)
    save_zone_models(zone_models, DATA_OUT_DIR / "gmm_zones.pkl")

    # Run canonicalization audits
    print("\n=== Canonicalization Audit ===")

    # Audit 1: Metadata coverage
    coverage = audit_canonicalization_coverage(corners_df)
    print(f"Corners with canonicalization metadata: {coverage['has_both']}/{coverage['total_corners']} ({coverage['coverage_pct']:.1f}%)")

    # Audit 2: Right-half occupancy
    init_occupancy = audit_right_half_occupancy(all_init)
    target_occupancy = audit_right_half_occupancy(all_target)
    print(f"Right-half occupancy - Initial: {init_occupancy:.1%}, Target: {target_occupancy:.1%} (expect >60%)")

    # Audit 3: Takepoint cluster
    takepoint_stats = audit_takepoint_cluster(corners_df)
    if "error" not in takepoint_stats:
        print(f"Takepoint cluster - X median: {takepoint_stats['x_median']:.1f}m (expect ~105), Y std: {takepoint_stats['y_std']:.1f}m (expect <5)")

    print("=" * 50 + "\n")

    # Visualize zones
    fig = visualize_zones(zone_models, all_init[:1000], all_target[:1000])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "gmm_zones.png", dpi=150, bbox_inches='tight')

    # Step 2: Encode run vectors
    print("\n[2/4] Encoding 42-d run vectors...")
    run_vectors = encode_all_corners(
        corners_df,
        tracking_dict,
        events_dict,
        zone_models,
        verbose=True
    )
    np.save(DATA_OUT_DIR / "run_vectors.npy", run_vectors)

    # Use only non-zero run vectors for NMF
    nz_idx = np.where(run_vectors.sum(axis=1) > 1e-8)[0]
    if nz_idx.size == 0:
        raise ValueError("All run vectors are zero; check GMM encoding thresholds and inputs.")
    run_vectors_nz = run_vectors[nz_idx]
    # Robust selection in Polars versions without DataFrame.take
    corners_df_nz = (
        corners_df
        .with_row_count('_rn')
        .filter(pl.col('_rn').is_in(nz_idx.tolist()))
        .drop('_rn')
    )

    # Step 3: NMF Routines
    print("\n[3/4] Fitting NMF for routine discovery...")
    # NMF requires n_components <= min(n_samples, n_features)
    n_corners = run_vectors_nz.shape[0]
    n_features = run_vectors_nz.shape[1]
    n_components = min(30, n_corners, n_features)

    if n_components < 30:
        print(f"  Note: Using {n_components} components (reduced from 30 due to sample size)")

    nmf_model = fit_nmf_routines(run_vectors_nz, n_components=n_components)
    save_nmf_model(nmf_model, DATA_OUT_DIR / "nmf_model.pkl")

    # Calculate feature importance (sum of W matrix columns)
    feature_importance = nmf_model.W.sum(axis=0)
    # Get order of features by importance (descending)
    feature_order = np.argsort(feature_importance)[::-1]

    # Visualize NMF features (Figure 3 style) and an example feature (Figure 4 style)
    corner_samples_for_viz = build_corner_samples_for_visuals(corners_df, events_dict, tracking_dict)
    fig_feat_grid = visualize_feature_grid(nmf_model, zone_models, feature_order=feature_order, feature_importance=feature_importance)
    fig_feat_grid.savefig(output_dir / "nmf_features_grid.png", dpi=170, bbox_inches='tight')
    # Example: feature 12 top corners (1-based index -> 0-based id 11 if exists)
    if nmf_model.n_topics >= 12:
        fig_top = visualize_top_corners_for_feature(11, nmf_model, corner_samples_for_viz, zone_models, top_n=10, corner_index_map=nz_idx.tolist())
        fig_top.savefig(output_dir / "feature_12_top_corners.png", dpi=170, bbox_inches='tight')

    # Team → top feature table and figure
    # Build team id -> name map from meta files (includes predefined fallback)
    meta_dir = RAW_DATA_DIR / 'meta'
    team_name_map = build_team_name_map(meta_dir, use_fallback=True)

    # Compute simple per-team xT from events (avg over corners)
    extra_metrics: Dict[int, Dict[str, float]] = {}
    for idx, corner in enumerate(corners_df_nz.iter_rows(named=True)):
        mid = corner['match_id']
        team_id = corner.get('team_id')
        if mid not in events_dict:
            continue
        ev = events_dict[mid]
        # Outcome window
        frame_start = corner['frame_start']
        period = corner['period']
        fs = int(frame_start)
        fe = int(frame_start + 10 * 25)
        mask = (
            (pl.col('period') == period) &
            (pl.col('frame_start') >= fs) & (pl.col('frame_start') <= fe)
        )
        if 'team_id' in ev.columns and team_id is not None:
            mask &= (pl.col('team_id') == team_id)
        ev_w = ev.filter(mask)
        xt_val = 0.0
        if 'xthreat' in ev_w.columns and ev_w.height > 0:
            try:
                xt_val = float(ev_w.select(pl.col('xthreat').sum()).item())
            except Exception:
                xt_val = 0.0
        if team_id is not None:
            dm = extra_metrics.setdefault(team_id, {'xt_total': 0.0, 'n_corners': 0})
            dm['xt_total'] += xt_val
            dm['n_corners'] += 1
    # finalize averages
    for k, v in extra_metrics.items():
        n = max(1, v.get('n_corners', 1))
        v['xt_avg'] = v.get('xt_total', 0.0) / n

    team_table = compute_team_top_feature_table(corners_df_nz, nmf_model, team_name_map=team_name_map, extra_metrics=extra_metrics)
    # Logo directory: Final_Project/assets (manual naming expected)
    logo_dir = (ROOT / 'Final_Project' / 'assets')
    if not logo_dir.exists():
        logo_dir = None
    save_team_top_feature_table(team_table, DATA_OUT_DIR / "team_top_feature.csv", output_dir / "team_top_feature.png", logo_dir=logo_dir)

    # Step 4: xT Surface
    print("\n[4/4] Building xT surface...")
    all_events = pl.concat(list(events_dict.values()), how="vertical_relaxed")
    xt_grid, n_iter, converged, n_actions = build_xt_surface(all_events, grid_shape=(12, 8))
    save_xt_surface(xt_grid, DATA_OUT_DIR / "xt_surface.pkl")

    # Visualize xT using the half-pitch visualizer (saves directly)
    visualize_xt_surface_half(
        xt_grid,
        converged,
        n_iter,
        OUTPUT_DIR / "xt_surface.png",
        n_actions
    )

    print("\nOK Phase 2 complete: All features saved")

    return corners_df, zone_models, nmf_model, xt_grid


def phase2b_add_labels(corners_df, events_dict, tracking_dict, xt_surface,
                       dataset_path: Path):
    """
    Phase 2b: Add target labels (y1-y5) to the corners dataset using the improved strategy.

    :param corners_df: DataFrame containing corner events.
    :param events_dict: Dictionary mapping match IDs to event DataFrames.
    :param tracking_dict: Dictionary mapping match IDs to tracking DataFrames.
    :param xt_surface: xT surface grid for y5 calculation.
    :param dataset_path: Path to save the labeled dataset (parquet).
    :return: DataFrame with added label columns.
    """
    print("\n" + "="*60)
    print("PHASE 2b: Computing Target Labels (IMPROVED STRATEGY)")
    print("="*60)
    print("\nUsing domain-driven approach:")
    print("  - y2: Historical corner xThreat by delivery zone")
    print("  - y3: Rule-based counter-attack detection")
    print("  - y4: Conditional counter danger (xT-based)")
    print("  - y1, y5: Original definitions (working well)\n")

    # Add labels using IMPROVED strategy
    labeled_df = add_labels_to_corners_dataset(
        corners_df,
        events_dict,
        tracking_dict,
        xt_surface,
        use_improved_labels=True,  # USE IMPROVED STRATEGY
        verbose=True
    )

    # Print label statistics
    print("\n" + "="*60)
    print("Label Statistics (IMPROVED)")
    print("="*60)
    for label in ['y1_label', 'y2_label', 'y3_label', 'y4_label', 'y5_label']:
        if label in labeled_df.columns:
            values = labeled_df[label].drop_nulls()
            mean_val = values.mean()
            nonzero_pct = (values > 0.01).sum() / len(values) * 100
            print(f"  {label}: mean={mean_val:.4f}, non-zero={nonzero_pct:.1f}%")

    # Save updated dataset
    print(f"\nSaving labeled dataset to {dataset_path}...")
    labeled_df.write_parquet(dataset_path)
    print(f"OK Saved {labeled_df.height} corners with IMPROVED labels")

    return labeled_df


def phase2c_visualize_targets(corners_df, events_dict, tracking_dict, output_dir: Path,
                               n_examples: int = 3):
    """
    Phase 2c: Create visualizations explaining what each target (y1-y5) measures.

    :param corners_df: DataFrame containing corner events.
    :param events_dict: Dictionary mapping match IDs to event DataFrames.
    :param tracking_dict: Dictionary mapping match IDs to tracking DataFrames.
    :param output_dir: Directory to save visualization outputs.
    :param n_examples: Number of example corners to visualize.
    :return: Path to the directory containing visualizations.
    """
    print("\n" + "="*60)
    print("PHASE 2c: Target Variable Visualizations")
    print("="*60)
    print(f"\nCreating visualizations for {n_examples} example corners...")
    print("  - Static images showing what each y1-y5 measures")
    print("  - Animated GIFs showing tracking data over time")
    print()

    from cti_visualize_targets import visualize_all_targets_for_corner

    viz_output_dir = output_dir / "target_visualizations"
    viz_output_dir.mkdir(parents=True, exist_ok=True)

    # Select diverse examples (different y values)
    examples = []

    # Try to find corners with different characteristics
    for i in range(min(n_examples, corners_df.height)):
        corner = corners_df.row(i * (corners_df.height // n_examples), named=True)
        match_id = corner['match_id']

        if match_id in events_dict and match_id in tracking_dict:
            examples.append((corner, events_dict[match_id], tracking_dict[match_id]))

        if len(examples) >= n_examples:
            break

    # Create visualizations for each example
    for idx, (corner, events_df, tracking_df) in enumerate(examples):
        try:
            visualize_all_targets_for_corner(
                corner, tracking_df, events_df, viz_output_dir, corner_idx=idx
            )
        except Exception as e:
            print(f"  Warning: Failed to visualize corner {idx}: {e}")
            continue

    print(f"\nOK Phase 2c complete: Visualizations saved to {viz_output_dir}")
    print(f"   Created {len(examples)} examples × 5 targets × 2 formats (static + GIF)")

    return viz_output_dir


def phase3_train_model(corners_df, events_dict, tracking_dict, xt_surface,
                       output_dir: Path):
    """
    Phase 3: Train the deep learning model using PyTorch Lightning.

    :param corners_df: DataFrame containing labeled corner events.
    :param events_dict: Dictionary mapping match IDs to event DataFrames.
    :param tracking_dict: Dictionary mapping match IDs to tracking DataFrames.
    :param xt_surface: xT surface grid.
    :param output_dir: Directory to save model checkpoints and artifacts.
    :return: Trained Lightning module.
    """
    print("\n" + "="*60)
    print("PHASE 3: Deep Learning Training")
    print("="*60)

    import pytorch_lightning as L
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from torch_geometric.loader import DataLoader

    # Create dataset
    print("\nCreating PyTorch Geometric dataset...")
    dataset = CornerGraphDataset(
        corners_df,
        tracking_dict,
        events_dict,
        xt_surface,
        radius=2.2
    )

    # Quick sanity: label prevalence (helps catch all-zeros)
    try:
        n_check = min(1000, len(dataset))
        pos = {k: 0 for k in ['y1','y3']}
        nonzero = {k: 0 for k in ['y2','y4','y5']}
        for i in range(n_check):
            d = dataset.get(i)
            if getattr(d, 'y', None) is None:
                continue
            y = d.y
            if y.numel() == 5:
                pos['y1'] += int(float(y[0]) > 0.5)
                pos['y3'] += int(float(y[2]) > 0.5)
                nonzero['y2'] += int(abs(float(y[1])) > 1e-6)
                nonzero['y4'] += int(abs(float(y[3])) > 1e-6)
                nonzero['y5'] += int(abs(float(y[4])) > 1e-6)
        if n_check > 0:
            print(f"Label preview (first {n_check}): P(shot)={pos['y1']/n_check:.1%}, P(counter)={pos['y3']/n_check:.1%}, xG≠0={nonzero['y2']/n_check:.1%}, xG_opp≠0={nonzero['y4']/n_check:.1%}, ΔxT≠0={nonzero['y5']/n_check:.1%}")
    except Exception as e:
        print(f"(Label preview skipped: {e})")

    # Train/val/test split (70/15/15) using TEMPORAL split by match_id
    # Extract match_ids from dataset and sort by match_id (proxy for chronological order)
    print("[CTI] Creating temporal train/val/test split by match_id...")

    # Get all corners with their match_ids
    corner_match_ids = []
    for i in range(len(dataset)):
        d = dataset.get(i)
        # match_id is stored in the Data object
        match_id = int(d.match_id.item()) if hasattr(d, 'match_id') else i  # fallback to index if no match_id
        corner_match_ids.append((i, match_id))

    # Sort by match_id to get temporal ordering
    corner_match_ids.sort(key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in corner_match_ids]

    # Split temporally: first 70% matches for train, next 15% for val, last 15% for test
    train_end = int(0.70 * len(sorted_indices))
    val_end = int(0.85 * len(sorted_indices))

    train_indices = sorted_indices[:train_end]
    val_indices = sorted_indices[train_end:val_end]
    test_indices = sorted_indices[val_end:]

    # Create Subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"  [TEMPORAL SPLIT] Train: {len(train_dataset)} corners (first 70% of matches)")
    print(f"  [TEMPORAL SPLIT] Val: {len(val_dataset)} corners (next 15% of matches)")
    print(f"  [TEMPORAL SPLIT] Test: {len(test_dataset)} corners (last 15% of matches)")
    print(f"  IMPORTANT: Test set contains temporally FUTURE matches relative to training")

    # Verify match_id ranges
    train_match_ids = {corner_match_ids[i][1] for i in train_indices}
    val_match_ids = {corner_match_ids[i][1] for i in val_indices}
    test_match_ids = {corner_match_ids[i][1] for i in test_indices}
    print(f"  Train match_id range: {min(train_match_ids)} - {max(train_match_ids)} ({len(train_match_ids)} unique matches)")
    print(f"  Val match_id range: {min(val_match_ids)} - {max(val_match_ids)} ({len(val_match_ids)} unique matches)")
    print(f"  Test match_id range: {min(test_match_ids)} - {max(test_match_ids)} ({len(test_match_ids)} unique matches)")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Estimate class imbalance for y1 (shot) and y3 (counter-shot) on train split
    try:
        # random_split returns a Subset with indices
        train_indices = set(train_dataset.indices.tolist()) if hasattr(train_dataset, 'indices') else set(range(len(train_dataset)))
    except Exception:
        train_indices = set(range(len(train_dataset)))
    pos_counts = { 'y1': 0, 'y3': 0 }
    total_checked = 0
    for i in range(len(dataset)):
        if i not in train_indices:
            continue
        d = dataset.get(i)
        if getattr(d, 'y', None) is None or d.y.numel() != 5:
            continue
        y = d.y
        pos_counts['y1'] += int(float(y[0]) > 0.5)
        pos_counts['y3'] += int(float(y[2]) > 0.5)
        total_checked += 1
    neg_y1 = max(0, total_checked - pos_counts['y1'])
    neg_y3 = max(0, total_checked - pos_counts['y3'])
    def _pw(neg, pos):
        if pos <= 0:
            return 1.0
        return float(min(100.0, max(1.0, neg / max(1, pos))))
    pos_weight_y1 = _pw(neg_y1, pos_counts['y1'])
    pos_weight_y3 = _pw(neg_y3, pos_counts['y3'])
    print(f"Class balance (train est.): y1 pos={pos_counts['y1']}/{total_checked}, pos_weight={pos_weight_y1:.2f}; y3 pos={pos_counts['y3']}/{total_checked}, pos_weight={pos_weight_y3:.2f}")

    # Create model
    print("\nInitializing CTI model...")
    model = CTIMultiTaskModel(
        input_dim=5,  # [x,y,vx,vy,team_flag]
        hidden_dim=128,
        num_gnn_layers=3,
        dropout=0.3,
        global_dim=3  # [is_short, delivery_dist, corner_side]
    )

    lightning_model = CTILightningModule(
        model,
        lr=3e-4,
        weight_decay=1e-4,
        pos_weight_y1=pos_weight_y1,
        pos_weight_y3=pos_weight_y3,
        pos_weight_cap=20.0,
        dynamic_pos_weight=True,
        use_focal_loss_y3=True,  # IMPROVED: Use Focal Loss for y3
        focal_alpha=0.75,         # IMPROVED: Alpha balancing factor
        focal_gamma=2.0           # IMPROVED: Gamma focusing parameter
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="cti-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=15,  # IMPROVED: Increased from 10 to 15 for more training stability
        mode="min"
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=50,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0,           # IMPROVED: Add gradient clipping
        gradient_clip_algorithm="norm"    # IMPROVED: Clip by L2 norm
    )

    # Train
    print("\nTraining model...")
    trainer.fit(lightning_model, train_loader, val_loader)

    print("\nOK Phase 3 complete: Model trained")

    # IMPORTANT: Fit Platt scaling calibration on validation set
    print("\n[CTI] Fitting Platt scaling calibration on validation set...")
    from cti_calibration import fit_platt_calibrators, save_calibrators

    try:
        # Load best model for calibration
        best_model_path = checkpoint_callback.best_model_path
        print(f"  Loading best model from: {best_model_path}")

        # Recreate model with same architecture
        best_cti_model = CTIMultiTaskModel(
            input_dim=5,
            hidden_dim=128,
            num_gnn_layers=3,
            dropout=0.3,
            global_dim=3
        )

        # Load the lightning module with the model (use strict=False to ignore pos_weight mismatches)
        best_lightning = CTILightningModule.load_from_checkpoint(
            best_model_path,
            model=best_cti_model,
            lr=3e-4,
            weight_decay=1e-4,
            pos_weight_y1=pos_weight_y1,
            pos_weight_y3=pos_weight_y3,
            pos_weight_cap=20.0,
            use_focal_loss_y3=True,  # IMPROVED: Match training config
            focal_alpha=0.75,
            focal_gamma=2.0,
            dynamic_pos_weight=True,
            strict=False  # Ignore pos_weight parameters in checkpoint
        )

        # Fit calibrators on validation set
        calibrators = fit_platt_calibrators(best_lightning, val_loader, device='cpu')

        # Save calibrators
        save_calibrators(calibrators, output_dir)

        print("[CTI] Platt scaling calibration complete and saved")
    except Exception as e:
        print(f"[WARNING] Could not fit Platt scaling: {e}")
        print("         Proceeding without calibration")

    # Save test_dataset INDICES only (not full dataset - causes MemoryError!)
    print(f"\nSaving test dataset indices ({len(test_dataset)} corners) for final evaluation...")
    test_dataset_path = output_dir / 'test_indices.pt'
    torch.save({
        'test_indices': test_dataset.indices if hasattr(test_dataset, 'indices') else list(range(len(test_dataset)))
    }, test_dataset_path)
    print(f"Test indices saved to: {test_dataset_path}")
    print(f"  (To load test set: use these indices with the original dataset)")

    return lightning_model


def phase4_evaluate(corners_df, lightning_model, output_dir: Path):
    """
    Phase 4: Evaluate the model and compute CTI metrics (placeholder).

    :param corners_df: DataFrame containing corner events.
    :param lightning_model: Trained PyTorch Lightning model.
    :param output_dir: Directory to save evaluation results.
    :return: None
    """
    print("\n" + "="*60)
    print("PHASE 4: Evaluation & CTI Computation")
    print("="*60)

    # TODO: Full evaluation with calibration, metrics, etc.
    print("OK Phase 4 complete (placeholder)")

    return None


def phase5_generate_reports(output_dir: Path):
    """
    Phase 5: Generate all reports and visuals from the pipeline outputs.

    :param output_dir: Directory containing pipeline outputs.
    """
    print("\n" + "="*60)
    print("PHASE 5: Generating Reports and Visuals")
    print("="*60)

    # Refresh key visuals needed inside the HTML report
    generate_team_cti_table_v2(DATA_OUT_DIR, output_dir, FINAL_PROJECT_DIR / "assets")

    offense_vs_counter_script = ROOT / "Final_Project" / "fix_offense_vs_counter_plot.py"
    if not offense_vs_counter_script.exists():
         offense_vs_counter_script = ROOT / "Final_Project" / "cti" / "cti_offense_vs_counter_plot.py"

    if offense_vs_counter_script.exists():
        _run_subprocess_py(offense_vs_counter_script, [])
    else:
        print(f"Warning: Offense vs Counter plot script not found: {offense_vs_counter_script}")

    scripts_to_run = [
        "cti_reliability_report.py",  # Interactive HTML calibration report with dynamic zones
        "cti_generate_team_feature_grid.py",  # Team feature grid
        "cti_generate_team_pitch_visuals.py",  # Individual team pitch visuals
        "cti_generate_top_corners_visual.py",  # Top corners visualization
    ]

    for script_name in scripts_to_run:
        script_path = ROOT / "Final_Project" / "cti" / script_name
        if script_path.exists():
            _run_subprocess_py(script_path, [])
        else:
            print(f"Warning: Report script not found, skipping: {script_name}")


def main(args):
    """
    Main pipeline execution.

    :param args: Parsed CLI arguments.
    """
    print("\n" + "="*70)
    print("         CORNER THREAT INDEX (CTI) PIPELINE")
    print("="*70)

    # Load match list - check multiple locations
    matches_file_root = RAW_DATA_DIR / "matches.parquet"
    matches_file_meta = RAW_DATA_DIR / "meta" / "matches.parquet"

    if matches_file_root.exists():
        df_matches = pl.read_parquet(matches_file_root)
    elif matches_file_meta.exists():
        df_matches = pl.read_parquet(matches_file_meta)
    else:
        print("Warning: No matches.parquet found, scanning directories...")
        # Scan for available matches
        dynamic_dir = RAW_DATA_DIR / "dynamic"
        match_ids = [int(f.stem) for f in dynamic_dir.glob("*.parquet")]
        df_matches = pl.DataFrame({"id": match_ids})

    # Use subset for testing
    if args.max_matches:
        match_ids = df_matches["id"].to_list()[:args.max_matches]
    else:
        match_ids = df_matches["id"].to_list()

    print(f"\nProcessing {len(match_ids)} matches")

    # Ensure output directories exist
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        corners_dataset_path = DATA_OUT_DIR / "corners_dataset.parquet"
        run_extraction = True

        # Check if dataset already exists and ask user if they want to reload
        if not args.force_rerun and corners_dataset_path.exists():
            print(f"\n[INFO] Found existing corners dataset: {corners_dataset_path}")
            # Check if running in non-interactive mode
            import sys
            if not sys.stdin.isatty():
                print("Running in non-interactive mode. Using existing dataset.")
                response = 'n'
            else:
                response = input("Do you want to reload data from all matches? (y/n): ").strip().lower()

            if response != 'y':
                run_extraction = False
                print("Skipping data loading. Will use existing dataset.")
            else:
                print("Reloading data from all matches...")

        if run_extraction:
            # Full training pipeline
            corners_df, events_dict, tracking_dict = phase1_extract_corners(
                match_ids,
                corners_dataset_path
            )
        else:
            print(f"\nSkipping corner extraction. Loading existing dataset from '{corners_dataset_path.name}'.")
            corners_df = pl.read_parquet(corners_dataset_path)

            # Manually load data for subsequent phases, since we skipped phase 1
            print("\nNOTE: Subsequent steps require the original event and tracking data.")
            print("Loading this data now (this may take a while)...")
            events_dict, tracking_dict = load_data_for_matches(match_ids)

        corners_df, zone_models, nmf_model, xt_surface = phase2_fit_features(
            corners_df,
            events_dict,
            tracking_dict,
            OUTPUT_DIR
        )

        # Add labels to dataset (requires xT surface)
        corners_df = phase2b_add_labels(
            corners_df,
            events_dict,
            tracking_dict,
            xt_surface,
            DATA_OUT_DIR / "corners_dataset.parquet"
        )

        lightning_model = phase3_train_model(
            corners_df,
            events_dict,
            tracking_dict,
            xt_surface,
            OUTPUT_DIR
        )

        phase4_evaluate(corners_df, lightning_model, OUTPUT_DIR)

        # Post-processing (default ON): inference + reports + GIF (animations run LAST)
        if not args.skip_infer:
            print("\n[POST] Running inference on all matches to generate predictions for all teams...")
            infer_script = ROOT / "Final_Project" / "cti" / "cti_infer_cti.py"
            # Run inference on ALL matches to ensure all teams get predictions
            # Note: Detailed team table (team_cti_detailed.csv) is generated automatically by inference script
            _run_subprocess_py(infer_script, ["--checkpoint", "best"])

            # Build v2 goal-weighted CTI table using the newly written CSV
            generate_team_cti_table_v2(DATA_OUT_DIR, OUTPUT_DIR, FINAL_PROJECT_DIR / "assets")
        else:
            print("[POST] Skipping inference and table generation (per flag)")

        # Generate reports BEFORE animations (faster, more important)
        if not args.skip_reports:
            phase5_generate_reports(OUTPUT_DIR)
        else:
            print("[POST] Skipping report generation (per flag)")

        # Corner visualization (static images instead of GIF for memory efficiency)
        if not args.skip_gif:
            print("\n[POST] Creating corner visualizations...")
            static_script = ROOT / "Final_Project" / "cti" / "cti_create_corner_static.py"
            _run_subprocess_py(static_script, ["--count", "4"])
            print("[POST] Note: Using static visualization instead of GIF animation for memory efficiency")
            print("[POST] To create GIF animation (requires more memory), run:")
            print("[POST]   python Final_Project/cti/cti_create_corner_animation.py --count 2 --freeze 3 --fps 8")
        else:
            print("[POST] Skipping corner visualizations (per flag)")

        print("\n" + "="*70)
        print("OK PIPELINE COMPLETE!")
        print("="*70)
        print(f"\nOutputs saved to: {OUTPUT_DIR}")

        if not args.skip_infer:
            predictions_csv = DATA_OUT_DIR / "predictions.csv"
            team_table_png = OUTPUT_DIR / "team_cti_detailed.png"
            team_table_csv = DATA_OUT_DIR / "team_cti_detailed.csv"
            corner_viz = OUTPUT_DIR / "corners_showcase_static.png"

            print("\nKey Results:")
            if predictions_csv.exists():
                print(f"  - Predictions: {predictions_csv}")
            if team_table_png.exists():
                print(f"  - Team Table (PNG): {team_table_png}")
            if team_table_csv.exists():
                print(f"  - Team Table (CSV): {team_table_csv}")
            if corner_viz.exists():
                print(f"  - Corner Showcase: {corner_viz}")

        print("\nNext steps:")
        print("  - View team rankings: open cti_outputs/team_cti_detailed.png")
        print("  - View top corners: open cti_outputs/corners_showcase_static.png")
        print("  - Analyze predictions: check cti_data/predictions.csv")
        print("  - Review model: check cti_outputs/checkpoints/")

    elif args.mode == "evaluate":
        print("Evaluation mode not yet implemented")

    elif args.mode == "infer":
        print("Inference mode not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTI Pipeline")
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "evaluate", "infer"],
                       help="Pipeline mode")
    parser.add_argument("--max-matches", type=int, default=None,
                       help="Limit number of matches (for testing)")
    # Post-processing toggles (default ON)
    parser.add_argument("--skip-infer", action="store_true", help="Skip post-training inference step")
    parser.add_argument("--skip-gif", action="store_true", help="Skip GIF animation generation")
    parser.add_argument("--skip-reports", action="store_true", help="Skip report generation step")
    parser.add_argument("--skip-viz", action="store_true", help="Skip target variable visualizations")
    parser.add_argument("--n-viz-examples", type=int, default=3, help="Number of corner examples to visualize")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-run of all steps")

    args = parser.parse_args()
    main(args)
