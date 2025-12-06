"""
Author: Tiago
Date: 2025-12-04
Description: Run CTI inference with a trained checkpoint and export predictions. Outputs predictions.csv and team_cti_table.png.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import polars as pl
import torch

from cti_integration import (
    CornerGraphDataset,
    CTIMultiTaskModel,
    CTILightningModule,
    compute_cti,
)
from cti_corner_extraction import load_events_basic, load_tracking_full
from cti_team_mapping import build_team_name_map
from cti_nmf_routines import save_team_cti_table
from cti_xt_surface_half_pitch import compute_delta_xt
from cti_generate_team_html import generate_team_cti_html

from cti_paths import REPO_ROOT, FINAL_PROJECT_DIR, DATA_2024

ROOT = REPO_ROOT
RAW_DATA_DIR = DATA_2024
DATA_DIR = FINAL_PROJECT_DIR / "cti_data"
OUT_FIG = FINAL_PROJECT_DIR / "cti_outputs"


def pick_best_checkpoint() -> Path | None:
    """
    Select the best model checkpoint from the output directory.

    Selection strategy:
    1. Prefer files with "val_loss" in the name, sorting by lowest loss.
    2. Fallback to sorting by name/modification time if no loss metric is found.

    :return: Path to the selected checkpoint or None if not found.
    """
    ckpt_dir = OUT_FIG / "checkpoints"
    if not ckpt_dir.exists():
        return None
    # Choose the most recent file (fallback) or the smallest val_loss if name contains it
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return None
    # Prefer filenames that include val_loss
    def parse_loss(p: Path) -> float:
        name = p.stem
        if "-" in name and name.rsplit("-", 1)[-1].replace(".", "", 1).isdigit():
            try:
                return float(name.rsplit("-", 1)[-1])
            except Exception:
                return float("inf")
        return float("inf")

    ckpts_sorted = sorted(ckpts, key=parse_loss)
    return ckpts_sorted[0]


def load_xt_surface(path: Path):
    """
    Load the xT surface grid from a pickle file.

    :param path: Path to the .pkl file containing the xT grid.
    :return: Loaded xT surface object (numpy array).
    """
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def build_dicts_for_matches(match_ids: list[int]):
    """
    Load events and tracking data for a list of matches.

    :param match_ids: List of match IDs to load.
    :return: Tuple of (events_dict, tracking_dict) mapping match IDs to DataFrames.
    """
    events_dict, tracking_dict = {}, {}
    for mid in match_ids:
        try:
            events_dict[mid] = load_events_basic(mid)
            tracking_dict[mid] = load_tracking_full(mid, sort_rows=False)
        except Exception:
            continue
    return events_dict, tracking_dict


def summarize_by_team(pred_df: pl.DataFrame, corners_df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate CTI predictions by team.

    :param pred_df: DataFrame containing model predictions (cti, y1-y5).
    :param corners_df: DataFrame containing corner metadata (team_id, match_id).
    :return: Aggregated DataFrame with team averages.
    """
    joined = pred_df.join(
        corners_df.select(["corner_id", "team_id", "match_id"]),
        on="corner_id",
        how="left",
    )
    # Filter out rows with null team_id
    joined = joined.filter(pl.col("team_id").is_not_null())
    return (
        joined.group_by("team_id")
        .agg([
            pl.len().alias("n_corners"),
            pl.col("cti").mean().alias("cti_avg"),
            pl.when(pl.col('y1_cal').is_not_null()).then(pl.col('y1_cal')).otherwise(pl.col('y1')).mean().alias("p_shot"),
            # Scale counter risk by 100 for readability
            ((pl.when(pl.col('y3_cal').is_not_null()).then(pl.col('y3_cal')).otherwise(pl.col('y3')) * pl.col("y4")).mean() * 100.0).alias("counter_risk"),
            pl.col("y5").mean().alias("delta_xt"),
        ])
        .sort("cti_avg", descending=True)
    )


def compute_team_cti_detailed(pred_df: pl.DataFrame, corners_df: pl.DataFrame, use_calibrated: bool = True) -> pl.DataFrame:
    """
    Compute detailed team CTI statistics including all component averages.

    :param pred_df: DataFrame containing model predictions.
    :param corners_df: DataFrame containing corner metadata.
    :param use_calibrated: Whether to use calibrated probabilities for y1 and y3.
    :return: DataFrame with detailed team stats.
    """
    joined = pred_df.join(
        corners_df.select(["corner_id", "team_id", "match_id"]),
        on="corner_id",
        how="left"
    )

    # Filter out rows with null team_id
    joined = joined.filter(pl.col("team_id").is_not_null())

    # Use calibrated predictions if available
    y1_col = pl.when(pl.col('y1_cal').is_not_null()).then(pl.col('y1_cal')).otherwise(pl.col('y1')) if use_calibrated else pl.col('y1')
    y3_col = pl.when(pl.col('y3_cal').is_not_null()).then(pl.col('y3_cal')).otherwise(pl.col('y3')) if use_calibrated else pl.col('y3')

    team_stats = (
        joined.group_by("team_id")
        .agg([
            pl.len().alias("n_corners"),
            pl.col("cti").mean().alias("cti_avg"),
            pl.col("cti").std().alias("cti_std"),
            y1_col.mean().alias("y1_avg"),
            pl.col("y2").mean().alias("y2_avg"),
            y3_col.mean().alias("y3_avg"),
            pl.col("y4").mean().alias("y4_avg"),
            pl.col("y5").mean().alias("y5_avg"),
            # Scale counter risk by 100 for readability
            ((y3_col * pl.col("y4")).mean() * 100.0).alias("counter_risk"),
        ])
        .sort("cti_avg", descending=True)
    )

    return team_stats


def _compute_corner_goal_stats(corners_df: pl.DataFrame, window_seconds: float = 10.0, fps: float = 25.0) -> pl.DataFrame:
    """
    Compute per-team corner goal rates in a post-delivery window using dynamic event files.

    A corner counts as a goal if any event for the attacking team has `lead_to_goal=True`
    within [frame_start, frame_start + window_seconds].

    :param corners_df: DataFrame containing corner events.
    :param window_seconds: Time window in seconds to check for goals.
    :param fps: Frames per second.
    :return: DataFrame with 'team_id', 'corner_goals', 'corner_goal_rate'.
    """
    window_frames = int(window_seconds * fps)
    dyn_dir = RAW_DATA_DIR / "dynamic"

    records = []
    for match_id in corners_df["match_id"].unique():
        dyn_path = dyn_dir / f"{int(match_id)}.parquet"
        if not dyn_path.exists():
            continue

        # Minimal columns for the check
        ev = pl.read_parquet(
            dyn_path,
            columns=["period", "frame_start", "team_id", "lead_to_goal"],
        ).sort(["period", "frame_start"])

        corners_subset = corners_df.filter(pl.col("match_id") == match_id)
        for row in corners_subset.iter_rows(named=True):
            fs = int(row["frame_start"])
            fe = fs + window_frames
            team_id = int(row["team_id"])
            period = row["period"]

            has_goal = (
                ev.filter(
                    (pl.col("period") == period)
                    & (pl.col("frame_start") >= fs)
                    & (pl.col("frame_start") <= fe)
                    & (pl.col("team_id") == team_id)
                    & (pl.col("lead_to_goal") == True)
                ).height
                > 0
            )

            records.append(
                {
                    "corner_id": row["corner_id"],
                    "team_id": team_id,
                    "goal_in_window": has_goal,
                }
            )

    if not records:
        return pl.DataFrame(
            {
                "team_id": [],
                "corner_goals": [],
                "corner_goal_rate": [],
            }
        )

    goal_df = pl.DataFrame(records)
    stats = (
        goal_df.group_by("team_id")
        .agg(
            [
                pl.len().alias("corner_goal_samples"),
                pl.col("goal_in_window").sum().alias("corner_goals"),
            ]
        )
        .with_columns(
            (
                pl.col("corner_goals")
                / pl.when(pl.col("corner_goal_samples") < 1)
                .then(1)
                .otherwise(pl.col("corner_goal_samples"))
            ).alias("corner_goal_rate")
        )
    )
    return stats


def summarize_by_team_variant(pred_df: pl.DataFrame, corners_df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """
    Summarize team statistics using suffixed column names (e.g., for empirical baselines).

    :param pred_df: DataFrame containing predictions/values with suffixes.
    :param corners_df: DataFrame containing corner metadata.
    :param prefix: Suffix string (e.g., '_e' for empirical) to append to column names.
    :return: Aggregated DataFrame.
    """
    joined = pred_df.join(
        corners_df.select(["corner_id", "team_id", "match_id"]),
        on="corner_id",
        how="left",
    )
    # Filter out rows with null team_id
    joined = joined.filter(pl.col("team_id").is_not_null())
    return (
        joined.group_by("team_id")
        .agg([
            pl.len().alias("n_corners"),
            pl.col(f"cti{prefix}").mean().alias("cti_avg"),
            pl.col(f"y1{prefix}").mean().alias("p_shot"),
            # Scale counter risk by 100
            ((pl.col(f"y3{prefix}") * pl.col(f"y4{prefix}")).mean() * 100.0).alias("counter_risk"),
            pl.col(f"y5{prefix}").mean().alias("delta_xt"),
        ])
        .sort("cti_avg", descending=True)
    )


def render_team_cti_table(team_df: pl.DataFrame, out_png: Path, title: str):
    """
    Render the team CTI table PNG. Prefer the already-built v2 table (goal-weighted)
    to keep visuals consistent across scatter/table/HTML. Fall back to the provided
    dataframe if the v2 CSV is missing.

    :param team_df: Team-level dataframe with CTI metrics (may contain team_id).
    :param out_png: Path to write the PNG table.
    :param title: Title for the table.
    """
    assets = ROOT / 'Final_Project' / 'assets'
    v2_path = DATA_DIR / 'team_cti_v2.csv'

    if v2_path.exists():
        df_named = pl.read_csv(v2_path)
    else:
        # Build team name map
        meta_dir = RAW_DATA_DIR / 'meta'
        team_name_map = build_team_name_map(meta_dir, use_fallback=True)
        team_names = [team_name_map.get(int(tid), str(tid)) for tid in team_df['team_id'].to_list()]
        df_named = team_df.with_columns([pl.Series('team', team_names)])

    save_team_cti_table(
        df_named.select(['team','cti_avg','p_shot','counter_risk','delta_xt','n_corners']),
        DATA_DIR / 'team_cti_summary.csv',
        out_png,
        title=title,
        logo_dir=assets
    )


def render_offense_vs_counter_plot(team_df: pl.DataFrame, out_png: Path):
    """
    Generate and save the offense vs counter-attack risk scatter plot.

    :param team_df: Team-level CTI statistics DataFrame.
    :param out_png: Output path for the plot PNG.
    """
    import subprocess
    import sys

    # Prefer the goal-weighted v2 table to stay consistent with the PNG table.
    v2_path = DATA_DIR / 'team_cti_v2.csv'
    if v2_path.exists():
        temp_csv = v2_path
    else:
        # Build team name map and write fallback CSV
        meta_dir = RAW_DATA_DIR / 'meta'
        team_name_map = build_team_name_map(meta_dir, use_fallback=True)
        team_names = [team_name_map.get(int(tid), str(tid)) for tid in team_df['team_id'].to_list()]
        df_named = team_df.with_columns([pl.Series('team', team_names)])
        temp_csv = DATA_DIR / 'team_cti_v2.csv'
        df_named.select(['team','cti_avg','p_shot','counter_risk','delta_xt','n_corners']).write_csv(temp_csv)

    # Call the offense vs counter plot script
    plot_script = ROOT / 'Final_Project' / 'fix_offense_vs_counter_plot.py'
    if not plot_script.exists():
        # Fallback to legacy path if needed
        plot_script = ROOT / 'Final_Project' / 'cti' / 'cti_offense_vs_counter_plot.py'
    try:
        subprocess.run([sys.executable, str(plot_script)], check=True, capture_output=True, text=True)
        print(f"OK generated offense vs counter plot: {out_png}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to generate offense vs counter plot: {e}")
        print(f"stderr: {e.stderr}")


def compute_empirical_targets(corner: dict, events_df: pl.DataFrame, xt_surface) -> dict:
    """
    Compute empirical baseline targets (y1-y5) from event data only.

    Used for sanity checking model predictions against actual outcomes.

    :param corner: Corner event dictionary.
    :param events_df: DataFrame containing match events.
    :param xt_surface: xT surface object.
    :return: Dictionary of empirical target values.
    """
    period = corner.get("period")
    frame_start = corner.get("frame_start")
    team_id = corner.get("team_id")
    fps = 25
    fs0 = int(frame_start + 0 * fps)
    fe0 = int(frame_start + 10 * fps)
    fs1 = int(frame_start + 10 * fps)
    fe1 = int(frame_start + 25 * fps)

    ev10 = events_df.filter(
        (pl.col("period") == period) & (pl.col("frame_start") >= fs0) & (pl.col("frame_start") <= fe0)
    )
    ev25 = events_df.filter(
        (pl.col("period") == period) & (pl.col("frame_start") >= fs1) & (pl.col("frame_start") <= fe1)
    )

    cols = set(events_df.columns)
    # Shot detector
    def _is_shot_expr():
        expr = None
        if "event_type" in cols:
            e = pl.col("event_type").cast(pl.Utf8, strict=False).str.to_lowercase() == "shot"
            expr = e if expr is None else (expr | e)
        if "event_subtype" in cols:
            e = pl.col("event_subtype").cast(pl.Utf8, strict=False).str.to_lowercase() == "shot"
            expr = e if expr is None else (expr | e)
        if "lead_to_shot" in cols:
            e = pl.col("lead_to_shot") == True
            expr = e if expr is None else (expr | e)
        if "is_shot" in cols:
            e = pl.col("is_shot") == True
            expr = e if expr is None else (expr | e)
        if expr is None and "end_type" in cols:
            expr = pl.col("end_type").cast(pl.Utf8, strict=False).str.to_lowercase() == "shot"
        return expr if expr is not None else pl.lit(False)

    team_mask = (pl.col("team_id") == team_id) if "team_id" in cols else pl.lit(True)
    opp_mask = (pl.col("team_id") != team_id) if "team_id" in cols else pl.lit(True)

    y1 = 1.0 if ev10.filter(team_mask & _is_shot_expr()).height > 0 else 0.0

    if "xthreat" in ev10.columns:
        xg_for = ev10.filter(team_mask).select(pl.col("xthreat").drop_nulls())
        y2 = float(xg_for.max().item()) if xg_for.height > 0 else 0.0
    else:
        y2 = 0.0

    y3 = 1.0 if ev25.filter(opp_mask & _is_shot_expr()).height > 0 else 0.0

    if "xthreat" in ev25.columns:
        xg_opp = ev25.filter(opp_mask).select(pl.col("xthreat").drop_nulls())
        y4 = float(xg_opp.max().item()) if xg_opp.height > 0 else 0.0
    else:
        y4 = 0.0

    # ΔxT approximation from end positions
    if ev10.height > 0 and {"x_end","y_end"}.issubset(cols):
        bp = ev10.select(["x_end","y_end"]).drop_nulls().rename({"x_end":"x","y_end":"y"})
        if bp.height >= 2:
            y5 = compute_delta_xt(bp, xt_surface)
        else:
            y5 = 0.0
    else:
        y5 = 0.0

    return {"y1": y1, "y2": y2, "y3": y3, "y4": y4, "y5": y5}


def _load_calibrators() -> dict | None:
    """
    Load trained Platt scaling calibrators from disk.

    :return: Dictionary of calibrators or None if not found.
    """
    path = DATA_DIR / 'calibrators.pkl'
    if not path.exists():
        return None
    try:
        import pickle as _p
        with open(path, 'rb') as f:
            return _p.load(f)
    except Exception:
        return None


def _reliability_points(pred: np.ndarray, label: np.ndarray, bins: int = 10):
    """
    Compute reliability curve points (predicted vs observed probabilities).

    :param pred: Predicted probabilities.
    :param label: Binary labels (0 or 1).
    :param bins: Number of bins for reliability curve.
    :return: Tuple of (bin_means_pred, bin_means_obs, bin_counts).
    """
    pred = np.asarray(pred).reshape(-1)
    label = np.asarray(label).reshape(-1)
    if pred.size == 0 or label.size == 0:
        return (np.array([]), np.array([]), np.array([]))
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(pred, edges[1:-1], right=False)
    bin_pred = []
    bin_obs = []
    bin_cnt = []
    for b in range(bins):
        m = idx == b
        if not np.any(m):
            bin_pred.append(np.nan)
            bin_obs.append(np.nan)
            bin_cnt.append(0)
            continue
        bin_pred.append(float(np.nanmean(pred[m])))
        bin_obs.append(float(np.nanmean(label[m])))
        bin_cnt.append(int(m.sum()))
    return np.array(bin_pred), np.array(bin_obs), np.array(bin_cnt)


def _plot_reliability(pred: np.ndarray, label: np.ndarray, out_png: Path, title: str):
    """
    Generate and save a reliability plot.

    :param pred: Predicted probabilities.
    :param label: Binary labels.
    :param out_png: Output path for the plot.
    :param title: Plot title.
    """
    import matplotlib.pyplot as plt
    px, py, cnt = _reliability_points(pred, label, bins=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='perfect')
    ax.plot(px, py, 'o-', color='#2C7FB8', label='model')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title(title)
    for (x, y, n) in zip(px, py, cnt):
        if np.isfinite(x) and np.isfinite(y):
            ax.text(x, y, str(int(n)), fontsize=8, ha='right', va='bottom', color='#444')
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _brier(pred: np.ndarray, label: np.ndarray) -> float:
    """
    Compute Brier score.

    :param pred: Predicted probabilities.
    :param label: Binary labels.
    :return: Brier score value.
    """
    pred = np.asarray(pred).reshape(-1)
    label = np.asarray(label).reshape(-1)
    if pred.size == 0 or label.size == 0:
        return float('nan')
    return float(np.nanmean((pred - label) ** 2))


def _ece(pred: np.ndarray, label: np.ndarray, bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    :param pred: Predicted probabilities.
    :param label: Binary labels.
    :param bins: Number of bins.
    :return: ECE value.
    """
    px, py, cnt = _reliability_points(pred, label, bins=bins)
    m = np.isfinite(px) & np.isfinite(py) & (cnt > 0)
    if not np.any(m):
        return float('nan')
    w = cnt[m] / np.sum(cnt[m])
    return float(np.sum(np.abs(py[m] - px[m]) * w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=None, help="number of matches to score (from the corners parquet order). Use None for all matches.")
    ap.add_argument("--checkpoint", type=str, default="best", help='path to .ckpt or "best" to auto-pick')
    ap.add_argument("--batch-size", type=int, default=50, help="number of matches to process at once (to avoid memory issues)")
    args = ap.parse_args()

    corners_path = DATA_DIR / "corners_dataset.parquet"
    xt_path = DATA_DIR / "xt_surface.pkl"
    if not corners_path.exists():
        raise SystemExit(f"Missing corners parquet: {corners_path}")
    if not xt_path.exists():
        raise SystemExit(f"Missing xT surface: {xt_path}")

    corners_df = pl.read_parquet(corners_path)

    # Choose subset of matches (or all if args.matches is None)
    all_match_ids = corners_df["match_id"].unique().to_list()
    if args.matches is not None:
        match_ids = all_match_ids[: args.matches]
        print(f"Processing {len(match_ids)} matches (subset)")
    else:
        match_ids = all_match_ids
        print(f"Processing all {len(match_ids)} matches in batches of {args.batch_size}")

    # Load checkpoint and model BEFORE batching
    ckpt_path = Path(args.checkpoint)
    if args.checkpoint == "best":
        picked = pick_best_checkpoint()
        if picked is None:
            raise SystemExit("No checkpoints found in cti_outputs/checkpoints")
        ckpt_path = picked

    # Build model + lightning module and load state
    base_model = CTIMultiTaskModel(input_dim=5, global_dim=3)
    lit = CTILightningModule(model=base_model)
    state = torch.load(ckpt_path, map_location="cpu")
    # Lightning checkpoint stores 'state_dict'
    lit.load_state_dict(state["state_dict"], strict=False)
    model = lit.model.eval()
    # Override Lambda/Gamma for better balance (Arsenal fix)
    lambda_cti = 0.1   # Reduced further to prioritize offensive output
    gamma_cti = 5.0    # Reduced from 80.0 (too penalizing for clearances) to 5.0
    calibrators = _load_calibrators()

    xt_surface = load_xt_surface(xt_path)

    # Process in batches to avoid memory issues
    all_records = []
    for batch_start in range(0, len(match_ids), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(match_ids))
        batch_match_ids = match_ids[batch_start:batch_end]
        print(f"\nProcessing batch {batch_start//args.batch_size + 1}/{(len(match_ids) + args.batch_size - 1)//args.batch_size}: matches {batch_start+1}-{batch_end}/{len(match_ids)}")

        events_dict, tracking_dict = build_dicts_for_matches(batch_match_ids)
        
        # Filter corners to only those matches that were successfully loaded
        valid_matches = set(tracking_dict.keys()) & set(events_dict.keys())
        corners_sub = corners_df.filter(pl.col("match_id").is_in(list(valid_matches)))
        
        if corners_sub.height == 0:
            print(f"Skipping batch {batch_start//args.batch_size + 1} (no valid data loaded)")
            continue

        dataset = CornerGraphDataset(corners_sub, tracking_dict, events_dict, xt_surface)
        from torch_geometric.loader import DataLoader
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        records = []
        idx = 0
        with torch.no_grad():
            for batch in loader:
                # Ensure batch has .batch when single-graph
                if not hasattr(batch, "batch"):
                    from torch_geometric.data import Batch
                    batch = Batch.from_data_list([batch])
                out = model(batch)
                bs = out["cti"].shape[0]
                for b in range(bs):
                    row = corners_sub.row(idx + b, named=True)
                    # Empirical baseline from events
                    mid = row.get("match_id")
                    ev = events_dict.get(mid)
                    emp = compute_empirical_targets(row, ev, xt_surface) if ev is not None else {k: 0.0 for k in ["y1","y2","y3","y4","y5"]}
                    # Calibrated probabilities if available
                    y1 = float(out["y1"][b].item())
                    y3 = float(out["y3"][b].item())
                    y1c = y1
                    y3c = y3
                    if calibrators is not None:
                        try:
                            import numpy as _np
                            if calibrators.get('y1') is not None and 'y1_logit' in out:
                                y1c = float(calibrators['y1'].predict_proba(_np.array([[float(out['y1_logit'][b].item())]]) )[:,1][0])
                            if calibrators.get('y3') is not None and 'y3_logit' in out:
                                y3c = float(calibrators['y3'].predict_proba(_np.array([[float(out['y3_logit'][b].item())]]) )[:,1][0])
                        except Exception:
                            pass

                    records.append({
                        "corner_id": row.get("corner_id"),
                        "match_id": mid,
                        "team_id": row.get("team_id"),
                    "y1": y1,
                    "y2": float(out["y2"][b].item()),
                    "y3": y3,
                    "y4": float(out["y4"][b].item()),
                    "y5": float(out["y5"][b].item()),
                    "cti": float(out["cti"][b].item()),
                    "y1_cal": y1c,
                    "y3_cal": y3c,
                    # empirical
                    "y1_e": float(emp["y1"]),
                    "y2_e": float(emp["y2"]),
                    "y3_e": float(emp["y3"]),
                    "y4_e": float(emp["y4"]),
                    "y5_e": float(emp["y5"]),
                    "cti_e": float(compute_cti(emp["y1"], emp["y2"], emp["y3"], emp["y4"], emp["y5"]))
                })
                idx += bs

        # Append batch records to all_records
        all_records.extend(records)

        # Clean up to free memory
        del events_dict, tracking_dict, dataset, loader
        import gc
        gc.collect()

    # Convert all records to DataFrame
    pred_df = pl.from_dicts(all_records)

    # Rescale y4 if significantly below empirical baseline
    y4_scale = 1.0
    try:
        y4_mean = float(pred_df["y4"].mean())
        y4_emp = float(pred_df["y4_e"].mean()) if "y4_e" in pred_df.columns else None
        if y4_emp and y4_emp > 0.0:
            ratio = y4_emp / max(y4_mean, 1e-8)
            if ratio > 5.0:
                y4_scale = min(10000.0, ratio)
                print(f"[CAL] Boosting y4 predictions by x{y4_scale:.1f} (mean {y4_mean:.6f} vs empirical {y4_emp:.6f})")
        if y4_scale != 1.0:
            pred_df = pred_df.with_columns((pl.col("y4") * y4_scale).alias("y4"))
        
        # Always recalculate CTI with new gamma_cti
        pred_df = pred_df.with_columns(
            (pl.col("y1") * pl.col("y2") - lambda_cti * pl.col("y3") * pl.col("y4") + gamma_cti * pl.col("y5")).alias("cti")
        )
    except Exception as e:
        print(f"[WARN] Could not rescale y4 predictions: {e}")

    out_csv = DATA_DIR / "predictions.csv"
    pred_df.write_csv(out_csv)
    print(f"OK wrote predictions: {out_csv}")

    # Compute corner-goal contribution (0-20s window) for better goal capture
    print("Computing corner goal stats (0-20s window) for goal-weighted CTI...")
    goal_stats = _compute_corner_goal_stats(corners_df, window_seconds=20.0, fps=25.0)

    # Generate team summaries (model + empirical)
    team_df_model = summarize_by_team(pred_df, corners_df)
    team_df_emp = summarize_by_team_variant(pred_df, corners_df, prefix="_e")
    meta_dir = RAW_DATA_DIR / "meta"
    team_name_map = build_team_name_map(meta_dir, use_fallback=True)
    team_names_model = [team_name_map.get(int(tid), str(tid)) for tid in team_df_model['team_id'].to_list()]
    team_df_model = team_df_model.with_columns([pl.Series('team', team_names_model)])

    # Apply goal-weighting to CTI (cti_avg + corner_goal_rate) and persist v2 CSV
    if goal_stats.height > 0:
        team_df_model = (
            team_df_model.join(goal_stats, on="team_id", how="left")
            .with_columns(
                [
                    pl.col("corner_goals").fill_null(0),
                    pl.col("corner_goal_rate").fill_null(0.0),
                    (pl.col("cti_avg") * 100.0).alias("cti_base"),  # Scale base CTI by 100
                    # New formula: (CTI_base*100 + GoalRate*100) / 2
                    # This balances the theoretical model with actual outcomes
                    ((pl.col("cti_avg") * 100.0 + pl.col("corner_goal_rate") * 100.0) / 2.0).alias("cti_goal_weighted"),
                ]
            )
        )
    else:
        team_df_model = team_df_model.with_columns(
            [
                (pl.col("cti_avg") * 100.0).alias("cti_base"),
                (pl.col("cti_avg") * 100.0).alias("cti_goal_weighted"),
                pl.lit(0).alias("corner_goals"),
                pl.lit(0.0).alias("corner_goal_rate"),
            ]
        )

    # Use goal-weighted CTI as the primary metric
    team_df_model = team_df_model.with_columns(pl.col("cti_goal_weighted").alias("cti_avg"))
    team_df_model = team_df_model.sort("cti_avg", descending=True)

    # Save goal-weighted table for downstream visuals (v2)
    team_df_v2 = team_df_model.select(
        [
            "team",
            "cti_avg",
            "p_shot",
            "counter_risk",
            "delta_xt",
            "n_corners",
            "cti_base",
            "cti_goal_weighted",
            "corner_goals",
            "corner_goal_rate",
            "team_id",
        ]
    )
    team_cti_v2_path = DATA_DIR / "team_cti_v2.csv"
    team_df_v2.write_csv(team_cti_v2_path)
    print(f"OK wrote goal-weighted team CTI (v2): {team_cti_v2_path}")

    render_team_cti_table(
        team_df_v2,
        OUT_FIG / "team_cti_table.png",
        "Team CTI Summary (Goal-Weighted)",
    )

    # Generate offense vs counter plot (reads team_cti_v2.csv if present)
    render_offense_vs_counter_plot(team_df_v2, OUT_FIG / "team_offense_vs_counter_presentation.png")

    # Generate interactive HTML report
    # meta_dir = RAW_DATA_DIR / 'meta'
    # team_name_map = build_team_name_map(meta_dir, use_fallback=True)
    # team_names_model = [team_name_map.get(int(tid), str(tid)) for tid in team_df_model['team_id'].to_list()]
    # df_named_model = team_df_model.with_columns([pl.Series('team', team_names_model)])
    # generate_team_cti_html(
    #     df_named_model,
    #     OUT_FIG / "team_cti_analysis.html",
    #     OUT_FIG / "team_cti_table.png",
    #     OUT_FIG / "team_offense_vs_counter_presentation.png",
    #     ROOT / 'Final_Project' / 'assets'
    # )

    # Save empirical CSV
    team_names_emp = [team_name_map.get(int(tid), str(tid)) for tid in team_df_emp['team_id'].to_list()]
    df_named_emp = team_df_emp.with_columns([pl.Series('team', team_names_emp)])
    # Save CSV only
    df_named_emp.select(['team','cti_avg','p_shot','counter_risk','delta_xt','n_corners']).write_csv(
        DATA_DIR / 'team_cti_summary_empirical.csv'
    )
    print(f"OK wrote team summary table: {OUT_FIG / 'team_cti_table.png'}")

    # Generate sanity report
    import numpy as np
    def _s(x):
        return float(np.nanmean(x)) if len(x) else float('nan')
    def _corr(a, b):
        if len(a) < 2:
            return float('nan')
        try:
            return float(np.corrcoef(a, b)[0, 1])
        except Exception:
            return float('nan')

    ya = pred_df["y1"].to_numpy(); yb = pred_df["y1_e"].to_numpy()
    ca = pred_df["cti"].to_numpy(); cb = pred_df["cti_e"].to_numpy()
    dxa = pred_df["y5"].to_numpy(); dxb = pred_df["y5_e"].to_numpy()

    ece_y1 = _ece(ya, yb); brier_y1 = _brier(ya, yb)
    ece_y3 = _ece(pred_df["y3"].to_numpy(), pred_df["y3_e"].to_numpy()); brier_y3 = _brier(pred_df["y3"].to_numpy(), pred_df["y3_e"].to_numpy())
    lines = [
        f"mean P(shot): model={_s(ya):.3f} emp={_s(yb):.3f} corr={_corr(ya,yb):.3f} ECE={ece_y1:.3f} Brier={brier_y1:.3f}",
        f"mean DeltaXT: model={_s(dxa):.4f} emp={_s(dxb):.4f} corr={_corr(dxa,dxb):.3f}",
        f"mean CTI:     model={_s(ca):.4f} emp={_s(cb):.4f} corr={_corr(ca,cb):.3f}",
        f"P(counter):   ECE={ece_y3:.3f} Brier={brier_y3:.3f}",
    ]
    report_path = OUT_FIG / "sanity_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\nSanity report:")
    for line in lines:
        print(f"  {line}")

    # Plot reliability curves
    _plot_reliability(pred_df["y1"].to_numpy(), pred_df["y1_e"].to_numpy(), OUT_FIG / "reliability_y1.png", "Reliability: P(shot)")
    _plot_reliability(pred_df["y3"].to_numpy(), pred_df["y3_e"].to_numpy(), OUT_FIG / "reliability_y3.png", "Reliability: P(counter)")
    print(f"Saved reliability plots to {OUT_FIG / 'reliability_y1.png'} and {OUT_FIG / 'reliability_y3.png'}")

    # Generate reliability HTML
    try:
        from cti_reliability_report import create_reliability_html
        create_reliability_html(
            pred_df["y1"].to_numpy(), pred_df["y1_e"].to_numpy(),
            pred_df["y3"].to_numpy(), pred_df["y3_e"].to_numpy(),
            OUT_FIG / "reliability_report.html",
            title="CTI Model Reliability Report"
        )
    except Exception as e:
        print(f"Warning: Could not generate HTML reliability report: {e}")

    # Generate detailed team table
    print("\nGenerating detailed team CTI table...")
    # Use the full corners_df for aggregation (not just the last batch subset)
    team_detailed = compute_team_cti_detailed(pred_df, corners_df, use_calibrated=True)

    # Add goal-weighted CTI
    print("Computing goal-weighted CTI (0-20s window, +1.0 * goal_rate)...")
    goal_stats_detail = goal_stats if goal_stats is not None else _compute_corner_goal_stats(corners_df, window_seconds=20.0, fps=25.0)
    if goal_stats_detail.height > 0:
        team_detailed = (
            team_detailed.join(goal_stats_detail, on="team_id", how="left")
            .with_columns(
                [
                    pl.col("corner_goals").fill_null(0),
                    pl.col("corner_goal_rate")
                    .fill_null(0.0)
                    .alias("corner_goal_rate"),
                    (
                        (pl.col("cti_avg") * 100.0 + pl.col("corner_goal_rate") * 100.0) / 2.0
                    ).alias("cti_goal_weighted"),
                ]
            )
        )
    else:
        team_detailed = team_detailed.with_columns(
            [
                pl.lit(0).alias("corner_goals"),
                pl.lit(0.0).alias("corner_goal_rate"),
                (pl.col("cti_avg") * 100.0).alias("cti_goal_weighted"),
            ]
        )

    # Add team names
    team_names = [team_name_map.get(int(tid), str(tid)) for tid in team_detailed['team_id'].to_list()]
    team_detailed = team_detailed.with_columns([pl.Series('team', team_names)])

    # Reorder columns
    team_detailed = team_detailed.select([
        'team', 'cti_avg', 'cti_goal_weighted', 'cti_std',
        'y1_avg', 'y2_avg', 'y3_avg', 'y4_avg', 'y5_avg',
        'counter_risk', 'corner_goals', 'corner_goal_rate',
        'n_corners', 'team_id'
    ])

    # Save CSV
    team_detailed.write_csv(DATA_DIR / "team_cti_detailed.csv")
    print(f"OK Saved detailed team table: {DATA_DIR / 'team_cti_detailed.csv'}")


if __name__ == "__main__":
    main()
