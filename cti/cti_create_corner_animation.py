"""
Author: Tiago
Date: 2025-12-04
Description: Create an animation that combines tracking + models for chosen corners.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pickle
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib import animation
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

# Local CTI function to avoid importing the full training stack (torch)
def compute_cti(y1: float, y2: float, y3: float, y4: float, y5: float, lambda_: float = 0.5, gamma_: float = 1.0) -> float:
    """CTI = y1*y2 - lambda*y3*y4 + gamma*y5"""
    try:
        return float(y1) * float(y2) - float(lambda_) * float(y3) * float(y4) + float(gamma_) * float(y5)
    except Exception:
        return float('nan')
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


def draw_xt_base(ax, xt_grid: np.ndarray):
    ax.clear()
    pitch = Pitch(pitch_type='custom', pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, half=True)
    pitch.draw(ax=ax)
    ax.set_facecolor('#313332')
    # Lock to standard half-pitch that corresponds to SkillCorner right half
    ax.set_xlim(52.5, 105.0)
    ax.set_ylim(0.0, 68.0)
    n_x, n_y = xt_grid.shape
    x_start_std = SC_ATTACK_ZONE_START + 52.5
    x_end_std = SC_ATTACK_ZONE_START + 52.5 + SC_ATTACK_ZONE_LENGTH
    # Paint zero areas black; use RdYlGn for >0
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_under(color='black', alpha=1.0)
    eps = 1e-9  # anything <= eps will use 'under' color (black)
    im = ax.imshow(
        xt_grid.T,
        extent=[x_start_std, x_end_std, 0, 68],
        origin='lower',
        cmap=cmap,
        vmin=eps,
        aspect='auto'
    )
    # No colorbar for animation compactness
    return im


def _sc_to_std(xy: np.ndarray) -> np.ndarray:
    """Convert SkillCorner centered meters to standard [0,105]x[0,68]."""
    if xy.size == 0:
        return xy
    return xy + np.array([52.5, 34.0])


def draw_gmm_zones(ax, zone_models, emphasize: bool = False):
    # Draw target GMM (active zones highlighted)
    BLUE = '#1877f2'
    ellipses = []
    for idx, (mean, covar) in enumerate(zip(zone_models.gmm_tgt.means_, zone_models.gmm_tgt.covariances_)):
        v, w = np.linalg.eigh(covar)
        angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # 95% conf ellipse
        # Use uniform light grey for all target zones
        color = '#d9d9d9'
        # Emphasize during freeze by increasing alpha and linewidth
        alpha = 0.50 if emphasize else 0.22
        lw = 2.2 if emphasize else 1.4
        mean_std = mean + np.array([52.5, 34.0])
        ell = Ellipse(mean_std, v[0], v[1], angle=angle, edgecolor=color, facecolor=color, alpha=alpha, linewidth=lw, zorder=3)
        ax.add_patch(ell)
        ellipses.append(ell)
    return ellipses


def _build_frame_list(tracking_df: pl.DataFrame, period: int, f0: int, f1: int) -> list[int]:
    # Ensure proper filtering and ordering
    df = tracking_df.filter((pl.col("period") == period) & (pl.col("frame") >= f0) & (pl.col("frame") <= f1))
    frames = df.select("frame").unique().sort("frame").to_series().to_list()
    return [int(x) for x in frames]


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


def _get_home_away_ids(match_id: int, events_df: pl.DataFrame) -> tuple[int | None, int | None]:
    """Try to resolve (home_id, away_id) for a match.
    Attempts:
      1) Columns in events_df: home_team_id/away_team_id
      2) meta/matches.parquet
      3) Fallback: None (caller can use attacking/defending ids)
    """
    home_id = away_id = None
    try:
        if 'home_team_id' in events_df.columns and 'away_team_id' in events_df.columns:
            row = events_df.select(['home_team_id', 'away_team_id']).drop_nulls().row(0, named=True)
            home_id = row.get('home_team_id')
            away_id = row.get('away_team_id')
            if home_id is not None and away_id is not None:
                return int(home_id), int(away_id)
    except Exception:
        pass
    try:
        meta_matches = DATA_2024 / 'meta/matches.parquet'
        if meta_matches.exists():
            dfm = pl.read_parquet(meta_matches)
            if 'id' in dfm.columns and 'home_team_id' in dfm.columns and 'away_team_id' in dfm.columns:
                row = dfm.filter(pl.col('id') == match_id)
                if row.height > 0:
                    home_id = row['home_team_id'].item()
                    away_id = row['away_team_id'].item()
                    return int(home_id), int(away_id)
    except Exception:
        pass
    return None, None


def _get_team_ids_from_map(tracking_df: pl.DataFrame, period: int, frame_start: int, attacking_team_id: int, team_map: dict) -> tuple[int, int]:
    df = tracking_df.filter((pl.col("period") == period) & (pl.col("frame") == frame_start) & (~pl.col("is_ball")))
    if df.height == 0:
        return attacking_team_id, attacking_team_id
    mapped = _add_team_ids(df, team_map)
    tids = [int(t) for t in mapped.select('team_id_map').drop_nulls().unique().to_series().to_list() if t is not None]
    defend = next((t for t in tids if t != attacking_team_id), attacking_team_id)
    return attacking_team_id, defend


def _find_kick_frame(tracking_df: pl.DataFrame, period: int, approx_frame: int, fps: int = 25) -> int:
    """Refine the corner kick frame by detecting the first high-speed ball movement
    within a small window around the annotated start frame.

    Returns an integer frame index. Falls back to approx_frame if not enough data.
    """
    if tracking_df.height == 0:
        return approx_frame
    win = fps * 2  # +/- 2 seconds
    df = (
        tracking_df
        .filter(
            (pl.col('period') == period) &
            (pl.col('is_ball') == True) &
            (pl.col('frame') >= (approx_frame - win)) &
            (pl.col('frame') <= (approx_frame + win))
        )
        .select(['frame', 'x_m', 'y_m'])
        .sort('frame')
    )
    if df.height < 2:
        return approx_frame
    frames = df['frame'].to_numpy()
    x = df['x_m'].to_numpy()
    y = df['y_m'].to_numpy()
    # compute per-frame speed (m/s) using frame deltas
    dfx = np.diff(x)
    dfy = np.diff(y)
    dframe = np.diff(frames)
    dtime = np.where(dframe > 0, dframe / float(fps), 1.0 / float(fps))
    speed = np.sqrt(dfx**2 + dfy**2) / dtime
    # choose first time speed exceeds threshold
    # threshold tuned for corner delivery (~3.0 m/s)
    idx = int(np.argmax(speed > 3.0)) if np.any(speed > 3.0) else -1
    if idx >= 0:
        return int(frames[idx + 1])
    # fallback: choose the earliest frame where ball is detected near the annotation
    return int(frames[0]) if frames.size > 0 else approx_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", type=int, default=6, help="seconds to freeze per corner overlay")
    ap.add_argument("--fps", type=int, default=10, help="frames per second")
    ap.add_argument("--count", type=int, default=3, help="number of distinct corners to include when --corner-id not provided")
    ap.add_argument("--corner-id", type=int, default=0, help="explicit corner_id to showcase (overrides --matches)")
    args = ap.parse_args()

    corners_path = DATA_DIR / "corners_dataset.parquet"
    xt_path = DATA_DIR / "xt_surface.pkl"
    zones_path = DATA_DIR / "gmm_zones.pkl"
    nmf_path = DATA_DIR / "nmf_model.pkl"
    team_table_path = DATA_DIR / "team_top_feature.csv"
    preds_path = DATA_DIR / "predictions.csv"

    for p in [corners_path, xt_path, zones_path, nmf_path, team_table_path, preds_path]:
        if not p.exists():
            raise SystemExit(f"Missing prerequisite artifact: {p}")

    corners = pl.read_parquet(corners_path)
    xt_grid = load_pickle(xt_path)
    zone_models = load_pickle(zones_path)
    team_table = pl.read_csv(team_table_path)
    preds = pl.read_csv(preds_path)

    # Team id -> name mapping (for lookups)
    meta_dir = DATA_2024 / "meta"
    team_name_map = build_team_name_map(meta_dir, use_fallback=True)

    # Choose a specific corner or sample multiple corners from the dataset
    if args.corner_id:
        sel_corners = corners.filter(pl.col("corner_id") == args.corner_id)
        if sel_corners.height == 0:
            raise SystemExit(f"corner_id={args.corner_id} not found in {corners_path}")
    else:
        # Prefer corners that passed quality gates if column exists
        if "passes_quality" in corners.columns:
            pool = corners.filter(pl.col("passes_quality") == True)
            if pool.height == 0:
                pool = corners
        else:
            pool = corners
        import numpy as _np
        n = int(min(max(1, args.count), pool.height))
        idx = _np.random.RandomState().choice(pool.height, size=n, replace=False)
        # Use with_row_index instead of deprecated with_row_count
        sel_corners = (
            pool.with_row_index('_rn')
                .filter(pl.col('_rn').is_in(idx.tolist()))
                .drop('_rn')
        )

    # Join predictions
    sel = sel_corners.select(["corner_id", "match_id", "team_id", "period", "frame_start"])\
        .join(preds, on="corner_id", how="left")

    # Figure + artists for animation (reduced size for memory efficiency)
    fig, ax = plt.subplots(figsize=(8, 5.6))
    fig.patch.set_facecolor('#313332')
    draw_xt_base(ax, xt_grid)
    # Leave space below the pitch for centered crests + VS
    plt.subplots_adjust(top=0.90, bottom=0.18)

    # Artists: two team scatters + ball
    atk_scatter = ax.scatter([], [], s=36, c='#ffd166', edgecolors='black', linewidths=0.3, zorder=5)
    def_scatter = ax.scatter([], [], s=36, c='#00b4d8', edgecolors='black', linewidths=0.3, zorder=5)
    ball_scatter = ax.scatter([], [], s=20, c='white', edgecolors='black', linewidths=0.3, zorder=6)

    # Overlays that we will update once per corner
    text_left = ax.text(0.01, 0.99, '', transform=ax.transAxes, ha='left', va='top', fontsize=10, color='w',
                        bbox=dict(boxstyle='round,pad=0.35', facecolor='black', alpha=0.55))
    text_right = ax.text(0.99, 0.99, '', transform=ax.transAxes, ha='right', va='top', fontsize=11, color='w', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.35', facecolor='black', alpha=0.55))
    crest_artists: list = []
    zone_artists: list = []
    vs_text = ax.text(0.5, -0.07, '', transform=ax.transAxes, ha='center', va='top', fontsize=11, color='w', fontweight='bold', clip_on=False)
    # Legend for GMM zones (left side)
    from matplotlib.patches import Patch
    leg_patch = Patch(facecolor='#d9d9d9', edgecolor='#d9d9d9', label='Target GMM zones')
    leg = ax.legend(handles=[leg_patch], loc='upper left', frameon=True)
    for txt in leg.get_texts():
        txt.set_color('w')
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_facecolor('none')
    leg.get_frame().set_edgecolor('none')

    frames_data = []  # list of (atk_xy, def_xy, ball_xy, emphasize_bool, left_txt, right_txt, crest_paths)

    # Build frames for each selected corner (concatenated)
    break_frames = int(args.fps * 1.0)  # 1 second pause between corners
    for row in sel.iter_rows(named=True):
        mid = int(row.get('match_id'))
        period = int(row.get('period')) if 'period' in sel.columns else int(row.get('period'))
        frame_start = int(row.get('frame_start'))
        fps = 25
        pre_s, post_s = 2, 10
        # Use the annotated dataset moment for the corner (exact frame_start)
        kick_frame = frame_start
        f0 = kick_frame - pre_s * fps
        f1 = kick_frame + post_s * fps

        events = load_events_basic(mid)
        tracking = load_tracking_full(mid, sort_rows=False)
        # Build player->team mapping and add team ids to tracking rows
        team_map = build_player_team_map(events)

        # Compute canonicalization flips so the attacking team always attacks the top-right corner
        corner_event_meta = {
            'frame_start': frame_start,
            'period': period,
            'attacking_side': row.get('attacking_side') if 'attacking_side' in sel.columns else None,
            'y_start': row.get('y_start') if 'y_start' in sel.columns else None,
        }
        s_x, s_y, _ = resolve_flip_signs(corner_event_meta, tracking, fps=fps, frame_window=2)

        # Determine team ids using the mapping at kick frame
        atk_id, def_id = _get_team_ids_from_map(tracking, period, kick_frame, int(row.get('team_id')), team_map)
        # Try to resolve home/away
        home_id, away_id = _get_home_away_ids(mid, events)
        if home_id is None or away_id is None:
            # Fallback to (attacking, defending) with attacking as 'home' for display purposes
            home_id, away_id = atk_id, def_id

        # Prepare text overlays
        def _fmt(x: float) -> str:
            try:
                xv = float(x)
            except Exception:
                return "N/A"
            if xv == 0.0:
                return "0"
            if abs(xv) >= 1e-3:
                return f"{xv:.3f}"
            return f"{xv:.2e}"

        def _fmt_pct(x: float) -> str:
            try:
                xv = float(x) * 100.0
            except Exception:
                return "N/A"
            if xv == 0.0:
                return "0%"
            if abs(xv) >= 0.1:
                return f"{xv:.1f}%"
            return f"{xv:.2e}%"
        t_atk = team_name_map.get(atk_id, str(atk_id))
        t_def = team_name_map.get(def_id, str(def_id))
        t_home = team_name_map.get(home_id, str(home_id))
        t_away = team_name_map.get(away_id, str(away_id))
        tf_row = team_table.filter(pl.col("team") == t_atk)
        top_feat = int(tf_row["top_feature_id"].item()) if tf_row.height > 0 else None
        cti = float(row.get('cti')) if row.get('cti') is not None else 0.0
        y1 = float(row.get('y1') or 0.0)
        y2 = float(row.get('y2') or 0.0)
        y3 = float(row.get('y3') or 0.0)
        y4 = float(row.get('y4') or 0.0)
        y5 = float(row.get('y5') or 0.0)
        # Calibrated probabilities if present in predictions
        y1_cal = row.get('y1_cal') if 'y1_cal' in sel.columns else None
        y3_cal = row.get('y3_cal') if 'y3_cal' in sel.columns else None
        risk_raw = y3 * y4
        risk_cal = (float(y3_cal) * y4) if (y3_cal is not None) else None
        # Optional calibrated CTI using defaults (lambda=0.5, gamma=1.0)
        cti_cal = None
        try:
            if y1_cal is not None and y3_cal is not None:
                cti_cal = compute_cti(float(y1_cal), y2, float(y3_cal), y4, y5)
        except Exception:
            cti_cal = None
        pshot_txt = _fmt_pct(y1) if y1_cal is None else f"{_fmt_pct(y1)} (cal {_fmt_pct(y1_cal)})"
        risk_txt = _fmt(risk_raw) if risk_cal is None else f"{_fmt(risk_raw)} (cal {_fmt(risk_cal)})"
        cti_txt = _fmt(cti) if cti_cal is None else f"{_fmt(cti)} (cal {_fmt(cti_cal)})"
        left_txt = (
            f"{t_atk} vs {t_def}\n"
            f"CTI: {cti_txt}   P(shot): {pshot_txt}   xG: {_fmt(y2)}\n"
            f"Counter risk: {risk_txt}   ΔxT: {_fmt(y5)}"
        )
        right_txt = f"Top Feature: {top_feat}" if top_feat is not None else ""

        crest_paths = []
        crest_a = ASSETS / f"{sanitize(t_home)}.png"
        crest_b = ASSETS / f"{sanitize(t_away)}.png"
        if crest_a.exists():
            crest_paths.append(('left', crest_a))
        if crest_b.exists():
            crest_paths.append(('right', crest_b))

        # Frame list for this corner
        frames_list = _build_frame_list(tracking, period, f0, f1)
        kick_atk = kick_def = kick_ball = []
        for fr in frames_list:
            df = tracking.filter((pl.col('period') == period) & (pl.col('frame') == fr))
            ppl = df.filter(~pl.col('is_ball'))
            ball = df.filter(pl.col('is_ball'))
            # Convert to standard coords
            if ppl.height > 0:
                ppl = _add_team_ids(ppl, team_map)
                # Ensure SkillCorner coords then canonicalize to attacking top-right
                ppl = ensure_skillcorner_xy(ppl)
                ppl = canonicalize_positions_sc(ppl, s_x, s_y)
                ax_atk = ppl.filter(pl.col('team_id_map') == atk_id)
                ax_def = ppl.filter(pl.col('team_id_map') == def_id)
            else:
                ax_atk = ppl
                ax_def = ppl
            # Ball: ensure SC coords and canonicalize too
            if ball.height > 0:
                ball_sc = ensure_skillcorner_xy(ball)
                ball_sc = canonicalize_positions_sc(ball_sc, s_x, s_y)
                ball_xy = _sc_to_std(ball_sc.select(['x_m','y_m']).to_numpy()).tolist()
            else:
                ball_xy = []
            # Convert players to standard
            atk_xy = _sc_to_std(ax_atk.select(['x_m','y_m']).to_numpy()).tolist() if ax_atk.height>0 else []
            def_xy = _sc_to_std(ax_def.select(['x_m','y_m']).to_numpy()).tolist() if ax_def.height>0 else []

            emphasize = (fr == kick_frame)
            if emphasize:
                kick_atk, kick_def = atk_xy, def_xy
                kick_ball = ball_xy
            frames_data.append((atk_xy, def_xy, ball_xy, emphasize, left_txt, right_txt, crest_paths))

        # Add freeze at the kick moment
        freeze_frames = args.fps * max(1, int(args.freeze))
        for _ in range(freeze_frames):
            frames_data.append((kick_atk, kick_def, kick_ball, True, left_txt, right_txt, crest_paths))
        # Small separator pause (empty overlays) between corners
        empty_txt_left = f"{t_home} vs {t_away}"
        for _ in range(break_frames):
            frames_data.append(([], [], [], False, empty_txt_left, '', crest_paths))

    # Prepare crest loader so they do not overlap with texts
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image
    crest_cache = {}

    def set_crests(crest_paths_local):
        # Clear old crests
        for art in crest_artists:
            try:
                art.remove()
            except Exception:
                pass
        crest_artists.clear()
        for side, path in crest_paths_local:
            try:
                arr = crest_cache.get(path)
                if arr is None:
                    img = Image.open(path).convert('RGBA')
                    aspect = img.width / max(1, img.height)
                    img = img.resize((int(40*aspect), 40))
                    arr = np.asarray(img)
                    crest_cache[path] = arr
                im = OffsetImage(arr, zoom=1.0)
                # Place centered below the pitch: home (left) and away (right)
                x = 0.40 if side == 'left' else 0.60
                ab = AnnotationBbox(im, (x, -0.07), frameon=False, xycoords='axes fraction', clip_on=False, zorder=12)
                ax.add_artist(ab)
                crest_artists.append(ab)
            except Exception:
                continue
        # Update 'VS' label between crests
        vs_text.set_text('VS')
        vs_text.set_position((0.50, -0.07))

    def init():
        import numpy as _np
        empty = _np.empty((0, 2))
        atk_scatter.set_offsets(empty)
        def_scatter.set_offsets(empty)
        ball_scatter.set_offsets(empty)
        text_left.set_text('')
        text_right.set_text('')
        return [atk_scatter, def_scatter, ball_scatter, text_left, text_right]

    def update(i):
        atk_xy, def_xy, ball_xy, emphasize, left_txt, right_txt, crests = frames_data[i]
        atk_scatter.set_offsets(np.array(atk_xy) if len(atk_xy) else np.empty((0,2)))
        def_scatter.set_offsets(np.array(def_xy) if len(def_xy) else np.empty((0,2)))
        ball_scatter.set_offsets(np.array(ball_xy) if len(ball_xy) else np.empty((0,2)))
        # Remove prior zone ellipses only, keep pitch
        for art in zone_artists:
            try:
                art.remove()
            except Exception:
                pass
        zone_artists.clear()
        zone_artists.extend(draw_gmm_zones(ax, zone_models, emphasize=emphasize))
        text_left.set_text(left_txt)
        text_right.set_text(right_txt)
        set_crests(crests)
        return [atk_scatter, def_scatter, ball_scatter, text_left, text_right, *crest_artists, *zone_artists]

    # Fallback: if no frames were built (e.g., no tracking available), render a static freeze per selected corner
    if len(frames_data) == 0:
        print("Warning: No tracking frames built; falling back to static overlays.")
        # Try to build at least one freeze frame using the first selected corner's metadata
        if sel.height > 0:
            row0 = sel.row(0, named=True)
            t_atk = team_name_map.get(int(row0.get('team_id')), str(row0.get('team_id')))
            left_txt = f"{t_atk}\n(no tracking available)"
            frames_data = [([], [], [], True, left_txt, '', [('left', ASSETS / f"{sanitize(t_atk)}.png")])] * max(1, args.fps * args.freeze)
        else:
            raise SystemExit("No corners available for animation.")

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=len(frames_data), interval=1000/args.fps, blit=False)

    out_path = OUT_DIR / "corners_showcase.gif"
    # Use lower DPI and optimize for memory efficiency
    writer = animation.PillowWriter(fps=args.fps)
    # Reduce figure DPI to save memory
    success = False
    try:
        ani.save(out_path, writer=writer, dpi=72)
        print(f"OK saved animation: {out_path}")
        success = True
    except (MemoryError, ValueError, np.core._exceptions._ArrayMemoryError) as e:
        print(f"Warning: Failed to save GIF due to memory constraints: {e}")
        print("Attempting fallback: saving as MP4 instead...")
        out_path_mp4 = OUT_DIR / "corners_showcase.mp4"
        try:
            ani.save(out_path_mp4, writer='ffmpeg', fps=args.fps, dpi=72, bitrate=1800)
            print(f"OK saved animation as MP4: {out_path_mp4}")
            success = True
        except Exception as e2:
            print(f"Error: Could not save animation in any format: {e2}")
            print("Try reducing --count or --freeze parameters to reduce memory usage.")
    finally:
        # Clean up to free memory
        plt.close(fig)


if __name__ == "__main__":
    main()
