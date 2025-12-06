"""
Author: Tiago
Date: 2025-12-04
Description: Create static visualizations of top corners instead of animated GIF. This is a memory-efficient alternative to cti_create_corner_animation.py.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pickle
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mplsoccer import Pitch

from cti_xt_surface_half_pitch import (
    GRID_SHAPE,
    SC_ATTACK_ZONE_START,
    SC_ATTACK_ZONE_LENGTH,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from cti_team_mapping import build_team_name_map
from cti_paths import REPO_ROOT, FINAL_PROJECT_DIR, DATA_2024, ASSETS_DIR
from cti_corner_extraction import load_events_basic, load_tracking_full
from cti_gmm_zones import (
    build_player_team_map,
    ensure_skillcorner_xy,
    canonicalize_positions_sc,
    resolve_flip_signs,
)


ROOT = REPO_ROOT
DATA_DIR = FINAL_PROJECT_DIR / "cti_data"
OUT_DIR = FINAL_PROJECT_DIR / "cti_outputs"
ASSETS = ASSETS_DIR


def load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def sanitize(name: str) -> str:
    return ''.join(ch.lower() for ch in str(name) if ch.isalnum())


def _sc_to_std(xy: np.ndarray) -> np.ndarray:
    """Convert SkillCorner centered meters to standard [0,105]x[0,68]."""
    if xy.size == 0:
        return xy
    return xy + np.array([52.5, 34.0])


def _add_team_ids(df: pl.DataFrame, team_map: dict) -> pl.DataFrame:
    if df.height == 0:
        return df
    if not team_map:
        return df.with_columns(pl.lit(None).alias('team_id_map'))
    map_df = pl.DataFrame({
        'player_id': list(team_map.keys()),
        'team_id_map': list(team_map.values())
    })
    return df.join(map_df, on='player_id', how='left')


def _get_team_ids_from_map(tracking_df: pl.DataFrame, period: int, frame_start: int, attacking_team_id: int, team_map: dict) -> tuple[int, int]:
    df = tracking_df.filter((pl.col("period") == period) & (pl.col("frame") == frame_start) & (~pl.col("is_ball")))
    if df.height == 0:
        return attacking_team_id, attacking_team_id
    mapped = _add_team_ids(df, team_map)
    tids = [int(t) for t in mapped.select('team_id_map').drop_nulls().unique().to_series().to_list() if t is not None]
    defend = next((t for t in tids if t != attacking_team_id), attacking_team_id)
    return attacking_team_id, defend


def draw_corner_snapshot(ax, xt_grid, zone_models, corner_row, tracking_df, events_df, team_name_map):
    """Draw a single corner snapshot on the given axis."""
    # Clear and setup pitch
    ax.clear()
    pitch = Pitch(pitch_type='custom', pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, half=True)
    pitch.draw(ax=ax)
    ax.set_facecolor('#313332')
    ax.set_xlim(52.5, 105.0)
    ax.set_ylim(0.0, 68.0)

    # Draw xT surface
    n_x, n_y = xt_grid.shape
    x_start_std = SC_ATTACK_ZONE_START + 52.5
    x_end_std = SC_ATTACK_ZONE_START + 52.5 + SC_ATTACK_ZONE_LENGTH
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_under(color='black', alpha=1.0)
    eps = 1e-9
    ax.imshow(
        xt_grid.T,
        extent=[x_start_std, x_end_std, 0, 68],
        origin='lower',
        cmap=cmap,
        vmin=eps,
        aspect='auto',
        alpha=0.6
    )

    # Draw GMM zones
    for idx, (mean, covar) in enumerate(zip(zone_models.gmm_tgt.means_, zone_models.gmm_tgt.covariances_)):
        v, w = np.linalg.eigh(covar)
        angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        color = '#d9d9d9'
        mean_std = mean + np.array([52.5, 34.0])
        ell = Ellipse(mean_std, v[0], v[1], angle=angle, edgecolor=color, facecolor=color,
                     alpha=0.35, linewidth=1.8, zorder=3)
        ax.add_patch(ell)

    # Get corner details
    mid = int(corner_row['match_id'])
    period = int(corner_row['period'])
    frame_start = int(corner_row['frame_start'])
    atk_id = int(corner_row['team_id'])

    # Build player team map
    team_map = build_player_team_map(events_df)

    # Compute canonicalization
    corner_event_meta = {
        'frame_start': frame_start,
        'period': period,
        'attacking_side': corner_row.get('attacking_side'),
        'y_start': corner_row.get('y_start'),
    }
    s_x, s_y, _ = resolve_flip_signs(corner_event_meta, tracking_df, fps=25, frame_window=2)

    # Get team ids
    atk_id, def_id = _get_team_ids_from_map(tracking_df, period, frame_start, atk_id, team_map)

    # Get positions at corner kick moment
    df = tracking_df.filter((pl.col('period') == period) & (pl.col('frame') == frame_start))
    ppl = df.filter(~pl.col('is_ball'))
    ball = df.filter(pl.col('is_ball'))

    if ppl.height > 0:
        ppl = _add_team_ids(ppl, team_map)
        ppl = ensure_skillcorner_xy(ppl)
        ppl = canonicalize_positions_sc(ppl, s_x, s_y)
        ax_atk = ppl.filter(pl.col('team_id_map') == atk_id)
        ax_def = ppl.filter(pl.col('team_id_map') == def_id)

        atk_xy = _sc_to_std(ax_atk.select(['x_m','y_m']).to_numpy()) if ax_atk.height > 0 else np.empty((0,2))
        def_xy = _sc_to_std(ax_def.select(['x_m','y_m']).to_numpy()) if ax_def.height > 0 else np.empty((0,2))

        ax.scatter(atk_xy[:,0], atk_xy[:,1], s=60, c='#ffd166', edgecolors='black', linewidths=0.5, zorder=5, label='Attacking')
        ax.scatter(def_xy[:,0], def_xy[:,1], s=60, c='#00b4d8', edgecolors='black', linewidths=0.5, zorder=5, label='Defending')

    if ball.height > 0:
        ball_sc = ensure_skillcorner_xy(ball)
        ball_sc = canonicalize_positions_sc(ball_sc, s_x, s_y)
        ball_xy = _sc_to_std(ball_sc.select(['x_m','y_m']).to_numpy())
        if ball_xy.size > 0:
            ax.scatter(ball_xy[:,0], ball_xy[:,1], s=40, c='white', edgecolors='black', linewidths=0.5, zorder=6, label='Ball')

    # Add text overlay with CTI info
    t_atk = team_name_map.get(atk_id, str(atk_id))
    t_def = team_name_map.get(def_id, str(def_id))
    cti = float(corner_row.get('cti', 0.0))
    y1 = float(corner_row.get('y1', 0.0))
    y2 = float(corner_row.get('y2', 0.0))

    title = f"{t_atk} vs {t_def}\nCTI: {cti:.3f}  |  P(shot): {y1*100:.1f}%  |  xG: {y2:.3f}"
    ax.set_title(title, fontsize=10, color='white', pad=10, weight='bold')

    # Small legend
    ax.legend(loc='upper left', fontsize=8, framealpha=0.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=3, help="number of corners to visualize")
    args = ap.parse_args()

    corners_path = DATA_DIR / "corners_dataset.parquet"
    xt_path = DATA_DIR / "xt_surface.pkl"
    zones_path = DATA_DIR / "gmm_zones.pkl"
    preds_path = DATA_DIR / "predictions.csv"

    for p in [corners_path, xt_path, zones_path, preds_path]:
        if not p.exists():
            raise SystemExit(f"Missing prerequisite artifact: {p}")

    corners = pl.read_parquet(corners_path)
    xt_grid = load_pickle(xt_path)
    zone_models = load_pickle(zones_path)
    preds = pl.read_csv(preds_path)

    # Team name mapping
    meta_dir = DATA_2024 / "meta"
    team_name_map = build_team_name_map(meta_dir, use_fallback=True)

    # Select corners with highest CTI
    corners_with_preds = corners.select(["corner_id", "match_id", "team_id", "period", "frame_start", "attacking_side", "y_start"])\
        .join(preds, on="corner_id", how="inner")

    # Sort by CTI and take top N
    top_corners = corners_with_preds.sort("cti", descending=True).head(min(args.count, corners_with_preds.height))

    if top_corners.height == 0:
        raise SystemExit("No corners with predictions found")

    # Create figure with subplots
    n_corners = top_corners.height
    cols = min(2, n_corners)
    rows = (n_corners + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
    fig.patch.set_facecolor('#313332')

    if n_corners == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    print(f"Creating static visualization for {n_corners} top corners...")

    for idx, row_data in enumerate(top_corners.iter_rows(named=True)):
        if idx >= len(axes):
            break

        mid = int(row_data['match_id'])
        print(f"  Corner {idx+1}/{n_corners}: match {mid}, CTI={row_data['cti']:.3f}")

        events = load_events_basic(mid)
        tracking = load_tracking_full(mid, sort_rows=False)

        draw_corner_snapshot(axes[idx], xt_grid, zone_models, row_data, tracking, events, team_name_map)

    # Hide unused axes
    for idx in range(n_corners, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    out_path = OUT_DIR / "corners_showcase_static.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#313332')
    plt.close(fig)

    print(f"\nOK saved static visualization: {out_path}")


if __name__ == "__main__":
    main()
