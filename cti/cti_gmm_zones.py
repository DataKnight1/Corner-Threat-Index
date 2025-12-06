"""
Author: Tiago
Date: 2025-12-04
Description: CTI GMM Zone Classification & Run Vector Encoding. Implements 6-component GMM for initial positions, 7-component GMM for target positions, and 42-dimensional run vector encoding.
"""

import numpy as np
import polars as pl
from sklearn.mixture import GaussianMixture
from typing import Tuple, List, Optional, Dict
import pickle
from pathlib import Path
from dataclasses import dataclass


@dataclass
@dataclass
class ZoneModels:
    """
    Container for GMM zone models and metadata.

    :param gmm_init: 6-component Gaussian Mixture for initial positions.
    :param gmm_tgt: 15-component Gaussian Mixture (containing active and inactive zones).
    :param active_tgt_ids: List of indices corresponding to active target components in the penalty area.
    :param n_initial_zones: Number of initial zones (default 6).
    :param n_target_zones: Number of active target zones (default 7).
    :param n_runs: Total dimension of run vector (6 * 7 = 42).
    """
    gmm_init: GaussianMixture      # 6-component for initial positions
    gmm_tgt: GaussianMixture       # 15-component (7 active + 8 inactive)
    active_tgt_ids: List[int]      # Which target components are in penalty area
    n_initial_zones: int = 6
    n_target_zones: int = 7
    n_runs: int = 42  # 6 x 7


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PITCH_CX = PITCH_LENGTH / 2.0
PITCH_CY = PITCH_WIDTH / 2.0
# Canonical corner in STANDARD coords (used only to infer flips); plotting occurs in standard
CANON_CORNER = (PITCH_LENGTH, PITCH_WIDTH)
# SkillCorner coordinate bounds (centered meters)
SC_X_MIN = -52.5
SC_X_MAX = 52.5
SC_Y_MIN = -34.0
SC_Y_MAX = 34.0
SC_X_SPAN = SC_X_MAX - SC_X_MIN
SC_Y_SPAN = SC_Y_MAX - SC_Y_MIN
# Penalty area bounds in SkillCorner coords
SC_PA_X_MIN = 36.0  # 52.5 - 16.5 (16.5m from goal line)
SC_PA_X_MAX = 52.5
# Penalty-area lateral bounds: 40.32m wide -> half-width 20.16m around center line
SC_PA_Y_MIN = -20.16
SC_PA_Y_MAX = 20.16
# Exclude backfield attackers in SC: x >= 0
MIN_TARGET_X = 0.0
FALLBACK_WARNING_LIMIT = 10
_fallback_warning_count = 0
_fallback_warning_suppressed = False
ON_BALL_EVENT_TYPES = {"pass", "cross", "carry", "take_on", "aerial_duel_won", "header", "shot", "ball_recovery", "touch"}
TAU_ACTIVE = 0.3
CHI2_THRESHOLD_SQ = 7.377758908227871  # chi2.ppf(0.975, df=2)
ACTIVE_ATTACKER_RANGE = (3, 10)
CORNER_TAKER_RADIUS_M = 3.0  # spatial exclusion radius around the corner-spot (SC coords)


def nearest_corner(x0: float, y0: float, pitch_length: float = 105.0, pitch_width: float = 68.0) -> Tuple[float, float]:
    """
    Find which of the 4 pitch corners is nearest to the given point.

    :param x0: X coordinate of the point.
    :param y0: Y coordinate of the point.
    :param pitch_length: Length of the pitch.
    :param pitch_width: Width of the pitch.
    :return: Tuple (x, y) of the nearest corner.
    """
    corners = np.array([
        [0.0, 0.0],
        [0.0, pitch_width],
        [pitch_length, 0.0],
        [pitch_length, pitch_width]
    ])
    diffs = corners - np.array([x0, y0])
    distances = (diffs ** 2).sum(axis=1)
    nearest_idx = np.argmin(distances)
    return tuple(corners[nearest_idx])



def compute_flip_signs(
    src_corner: Tuple[float, float],
    dst_corner: Tuple[float, float] = (105.0, 68.0),
    pitch_length: float = 105.0,
    pitch_width: float = 68.0
) -> Tuple[int, int]:
    """
    Compute flip signs to map source corner to destination corner.

    Uses center-relative flips: x' = s_x * (x - cx) + cx.

    :param src_corner: Source corner coordinates (x, y).
    :param dst_corner: Destination (canonical) corner coordinates (default: 105, 68).
    :param pitch_length: Pitch length.
    :param pitch_width: Pitch width.
    :return: Tuple (s_x, s_y) where s_x, s_y are in {+1, -1}.
    """
    s_x = +1 if src_corner[0] == pitch_length else -1
    s_y = +1 if src_corner[1] == pitch_width else -1
    return s_x, s_y



def canonicalize_positions_sc(
    positions_df: pl.DataFrame,
    s_x: int,
    s_y: int,
) -> pl.DataFrame:
    """
    Apply center-relative flips to canonicalize positions in SkillCorner coords (centered).

    Transformation: x' = s_x * x, y' = s_y * y.

    :param positions_df: DataFrame containing position columns (x_m, y_m, etc.).
    :param s_x: X-axis flip sign (+1 or -1).
    :param s_y: Y-axis flip sign (+1 or -1).
    :return: DataFrame with transformed coordinates.
    """
    if positions_df.height == 0:
        return positions_df

    result = positions_df.with_columns([
        (s_x * pl.col("x_m")).alias("x_m"),
        (s_y * pl.col("y_m")).alias("y_m")
    ])

    if "vx" in positions_df.columns:
        result = result.with_columns((s_x * pl.col("vx")).alias("vx"))
    if "vy" in positions_df.columns:
        result = result.with_columns((s_y * pl.col("vy")).alias("vy"))
    if "ax" in positions_df.columns:
        result = result.with_columns((s_x * pl.col("ax")).alias("ax"))
    if "ay" in positions_df.columns:
        result = result.with_columns((s_y * pl.col("ay")).alias("ay"))

    return result



def ensure_skillcorner_xy(positions_df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure coordinates are in SkillCorner centered meters (x in [-52.5, 52.5], y in [-34, 34]).

    If positions look like standard coordinates (0..105, 0..68), they are converted by subtracting the center.
    Otherwise, they are clipped to SC bounds.

    :param positions_df: DataFrame containing position columns 'x_m', 'y_m'.
    :return: DataFrame with standard SkillCorner coordinates.
    """
    if positions_df.height == 0 or "x_m" not in positions_df.columns:
        return positions_df

    x_min = positions_df["x_m"].min()
    x_max = positions_df["x_m"].max()
    if x_min is None or x_max is None:
        return positions_df

    # Already SC
    if x_min >= SC_X_MIN - 1.0 and x_max <= SC_X_MAX + 1.0:
        return positions_df

    # Likely standard -> convert to SC by subtracting center
    if x_min >= -1.0 and x_max <= PITCH_LENGTH + 1.0:
        return positions_df.with_columns([
            (pl.col("x_m") - PITCH_CX).alias("x_m"),
            (pl.col("y_m") - PITCH_CY).alias("y_m")
        ])

    # Otherwise, clip to SC bounds conservatively
    return positions_df.with_columns([
        pl.col("x_m").clip(SC_X_MIN, SC_X_MAX).alias("x_m"),
        pl.col("y_m").clip(SC_Y_MIN, SC_Y_MAX).alias("y_m")
    ])


def ensure_standard_xy(positions_df: pl.DataFrame) -> pl.DataFrame:
    """
    Deprecated in SC-only mode; retained for compatibility. Returns input unchanged.
    """
    return positions_df



def infer_ball_takepoint(
    tracking_df: pl.DataFrame,
    corner_event: Dict,
    fps: int = 25,
    frame_window: int = 2
) -> Optional[Tuple[float, float]]:
    """
    Locate the ball position (takepoint) near the corner execution frame in tracking data.

    :param tracking_df: Tracking data DataFrame.
    :param corner_event: Corner event dictionary.
    :param fps: Frames per second.
    :param frame_window: Window of frames around the start frame to search.
    :return: Tuple (x_std, y_std) in standard 0-105 x 0-68 coordinates, or None if not detected.
    """
    frame_take = corner_event.get("frame_start")
    period = corner_event.get("period")

    if frame_take is None or period is None:
        return None

    ball_rows = tracking_df.filter(
        (pl.col("period") == period) &
        (pl.col("is_ball") == True) &
        pl.col("frame").is_between(frame_take - frame_window, frame_take + frame_window) &
        pl.col("x_m").is_not_null() &
        pl.col("y_m").is_not_null()
    )

    if "is_detected" in ball_rows.columns:
        ball_rows = ball_rows.filter(pl.col("is_detected") == True)

    if ball_rows.height == 0:
        return None

    ball_rows = ball_rows.with_columns(
        (pl.col("frame") - frame_take).abs().alias("frame_delta")
    ).sort(["frame_delta", "frame"])

    take_row = ball_rows.row(0, named=True)
    x_raw = float(take_row["x_m"])
    y_raw = float(take_row["y_m"])
    x_clamped = min(max(x_raw, SC_X_MIN), SC_X_MAX)
    y_clamped = min(max(y_raw, SC_Y_MIN), SC_Y_MAX)
    # Convert SC -> standard for corner inference
    x_std = ((x_clamped - SC_X_MIN) / SC_X_SPAN) * PITCH_LENGTH
    y_std = ((y_clamped - SC_Y_MIN) / SC_Y_SPAN) * PITCH_WIDTH
    return x_std, y_std


def fallback_flip_signs(corner_event: Dict) -> Tuple[int, int]:
    """
    Fallback flip computation using event metadata when the ball takepoint is unavailable.
    """
    attacking_side = corner_event.get("attacking_side")
    y_start = corner_event.get("y_start")

    if attacking_side == "right_to_left":
        s_x = -1
    else:
        s_x = +1

    if y_start is None:
        s_y = +1
    else:
        y_std = y_start + 34.0
        s_y = +1 if y_std >= PITCH_CY else -1

    return s_x, s_y



def resolve_flip_signs(
    corner_event: Dict,
    tracking_df: pl.DataFrame,
    fps: int = 25,
    frame_window: int = 2
) -> Tuple[int, int, bool]:
    """
    Determine canonicalization flip signs for a corner.

    Tries to use ball tracking data first. If unavailable, falls back to event metadata.

    :param corner_event: Corner event dictionary.
    :param tracking_df: Tracking data DataFrame.
    :param fps: Frames per second.
    :param frame_window: Frame window for search.
    :return: Tuple (s_x, s_y, used_fallback).
    """
    takepoint = infer_ball_takepoint(tracking_df, corner_event, fps=fps, frame_window=frame_window)
    if takepoint is not None:
        src_corner = nearest_corner(*takepoint, pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH)
        s_x, s_y = compute_flip_signs(src_corner, dst_corner=CANON_CORNER,
                                      pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH)
        return s_x, s_y, False

    s_x, s_y = fallback_flip_signs(corner_event)
    return s_x, s_y, True



def audit_canonicalization_coverage(
    corners_df: pl.DataFrame
) -> Dict:
    """
    Audit how many corners have required canonicalization metadata.

    :param corners_df: Corners DataFrame.
    :return: Dictionary containing coverage statistics.
    """
    total = corners_df.height
    has_attacking_side = corners_df.filter(pl.col("attacking_side").is_not_null()).height
    has_y_start = corners_df.filter(pl.col("y_start").is_not_null()).height
    has_both = corners_df.filter(
        pl.col("attacking_side").is_not_null() &
        pl.col("y_start").is_not_null()
    ).height

    return {
        "total_corners": total,
        "has_attacking_side": has_attacking_side,
        "has_y_start": has_y_start,
        "has_both": has_both,
        "coverage_pct": (has_both / total * 100) if total > 0 else 0.0
    }



def audit_right_half_occupancy(
    positions: np.ndarray,
    pitch_length: float = 105.0
) -> float:
    """
    Check if positions are concentrated in the attacking (right) half.

    :param positions: Array of shape (N, 2) with [x, y] positions.
    :param pitch_length: Pitch length (default 105.0).
    :return: Fraction of positions located in the right half.
    """
    if positions.shape[0] == 0:
        return 0.0

    # If data are in SkillCorner (min x < 0), use threshold 0; else use half of standard length
    if positions.shape[0] and np.nanmin(positions[:, 0]) < 0:
        thresh = 0.0
    else:
        thresh = pitch_length / 2.0
    right_half_count = (positions[:, 0] > thresh).sum()
    return float(right_half_count / positions.shape[0])



def audit_takepoint_cluster(
    corners_df: pl.DataFrame,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0
) -> Dict:
    """
    Verify ball takepoints are clustered near the attacking corner.

    After canonicalization, expect:
    - x_start median near 105m (attacking corner).
    - y_start near chosen touchline (low variance).

    :param corners_df: Corners DataFrame with x_start, y_start columns.
    :param pitch_length: Pitch length.
    :param pitch_width: Pitch width.
    :return: Dictionary containing takepoint statistics.
    """
    # Convert from centered to standard coords if needed
    if "x_start" not in corners_df.columns:
        return {"error": "Missing x_start column"}

    x_vals = corners_df["x_start"].to_list()
    y_vals = corners_df["y_start"].to_list()

    # If in centered coords, convert for audit
    x_mean = np.mean(x_vals)
    if abs(x_mean) < 60:  # Likely centered coords
        x_vals = [x + 52.5 for x in x_vals]
        y_vals = [y + 34.0 for y in y_vals]

    return {
        "x_median": np.median(x_vals),
        "x_std": np.std(x_vals),
        "y_median": np.median(y_vals),
        "y_std": np.std(y_vals),
        "expect_x": f"near {pitch_length}",
        "expect_y_std": "< 5m"
    }



def infer_corner_taker_player_id(corner_event: Dict, events_df: Optional[pl.DataFrame]) -> Optional[int]:
    """
    Infer the corner-taker's player_id from events if not provided.

    Looks for the first on-ball attacking event after the corner starts within a small window.

    :param corner_event: Corner event dictionary.
    :param events_df: Events DataFrame.
    :return: Player ID of the corner taker, or None if not found.
    """
    if events_df is None or events_df.height == 0:
        return None

    frame_start = corner_event.get("frame_start")
    period = corner_event.get("period")
    team_id = corner_event.get("team_id") or corner_event.get("attacking_team_id")
    if frame_start is None or period is None or team_id is None:
        return None

    filt = (
        (pl.col("period") == period) &
        (pl.col("frame_start") >= frame_start) &
        (pl.col("frame_start") <= frame_start + 50)  # ~2s at 25 fps
    )
    if "event_type" in events_df.columns:
        filt &= pl.col("event_type").is_in(ON_BALL_EVENT_TYPES)
    if "team_id" in events_df.columns:
        filt &= pl.col("team_id") == team_id

    ev = events_df.filter(filt).sort("frame_start")
    if ev.height == 0 or "player_id" not in ev.columns:
        return None
    pid = ev[0, "player_id"]
    return int(pid) if pid is not None else None



def extract_initial_positions(
    corner_event: Dict,
    tracking_df: pl.DataFrame,
    events_df: Optional[pl.DataFrame] = None,
    time_before: float = 2.0,
    fps: int = 25
) -> pl.DataFrame:
    """
    Extract attacker positions at a fixed time before the corner kick.

    Standard: t = -2.0s before corner is taken.

    :param corner_event: Dictionary containing corner event details.
    :param tracking_df: Full match tracking data.
    :param events_df: Optional events data.
    :param time_before: Seconds before corner execution to extract (default 2.0).
    :param fps: Frames per second.
    :return: DataFrame with columns [player_id, x, y, x_m, y_m].
    """
    frame_target = int(corner_event["frame_start"] - time_before * fps)
    period = corner_event["period"]

    # Try exact frame first
    positions = tracking_df.filter(
        (pl.col("period") == period) &
        (pl.col("frame") == frame_target) &
        (~pl.col("is_ball"))
    )

    # If empty, apply ±2 frame tolerance and pick closest frame
    if positions.height == 0:
        positions = tracking_df.filter(
            (pl.col("period") == period) &
            (pl.col("frame") >= frame_target - 2) &
            (pl.col("frame") <= frame_target + 2) &
            (~pl.col("is_ball"))
        ).with_columns(
            (pl.col("frame") - frame_target).abs().alias("frame_delta")
        )

        # Group by player and take the frame closest to target
        if positions.height > 0:
            min_delta = positions["frame_delta"].min()
            positions = positions.filter(pl.col("frame_delta") == min_delta)
            positions = positions.drop("frame_delta")

    if "x_m" not in positions.columns:
        # Coordinates should already be rescaled, but handle if not
        return positions.select(["player_id", "x", "y"])

    # Ensure SkillCorner coordinates
    positions = ensure_skillcorner_xy(positions)

    player_team_map = corner_event.get("player_team_map")
    positions = assign_team_ids(positions, player_team_map)

    attacking_team_id = corner_event.get("attacking_team_id") or corner_event.get("team_id")
    if attacking_team_id is not None and "team_id_mapped" in positions.columns:
        positions = positions.with_columns(
            pl.col("team_id_mapped").alias("team_id")
        ).drop("team_id_mapped")
    elif "team_id_mapped" in positions.columns:
        positions = positions.rename({"team_id_mapped": "team_id"})

    if attacking_team_id is not None and "team_id" in positions.columns:
        positions = positions.with_columns(
            (pl.col("team_id") == attacking_team_id).alias("is_attacking")
        )
    else:
        positions = positions.with_columns(pl.lit(None).alias("is_attacking"))

    s_x, s_y, used_fallback = resolve_flip_signs(corner_event, tracking_df, fps=fps)
    positions = canonicalize_positions_sc(positions, s_x, s_y)

    taker_id = corner_event.get("taker_player_id") or corner_event.get("player_in_possession_id")
    if taker_id is not None:
        positions = positions.filter(pl.col("player_id") != taker_id)

    # Spatial exclusion: remove any player standing in the corner-taker zone
    # After canonicalization to the attacking top-right corner, the corner-spot is (SC_X_MAX, SC_Y_MAX)
    # Exclude points within CORNER_TAKER_RADIUS_M of that spot
    dx = pl.col("x_m") - SC_X_MAX
    dy = pl.col("y_m") - SC_Y_MAX
    positions = positions.filter((dx * dx + dy * dy) > (CORNER_TAKER_RADIUS_M ** 2))

    positions = positions.with_columns([
        pl.col("x_m").clip(SC_X_MIN, SC_X_MAX).alias("x_m"),
        pl.col("y_m").clip(SC_Y_MIN, SC_Y_MAX).alias("y_m"),
    ])

    if used_fallback:
        global _fallback_warning_count, _fallback_warning_suppressed
        _fallback_warning_count += 1
        if _fallback_warning_count <= FALLBACK_WARNING_LIMIT:
            corner_id = corner_event.get("corner_id", "unknown")
            match_id = corner_event.get("match_id", "unknown")
            print(f"  Warning: Corner {corner_id} (match {match_id}) used metadata fallback for canonicalization (initial)")
        elif not _fallback_warning_suppressed:
            print("  Warning: additional canonicalization fallbacks suppressed (initial)")
            _fallback_warning_suppressed = True

    select_cols = ["player_id", "x_m", "y_m", "is_attacking"]
    if "team_id" in positions.columns:
        select_cols.append("team_id")

    return positions.select(select_cols)


def extract_target_positions(
    corner_event: Dict,
    tracking_df: pl.DataFrame,
    events_df: pl.DataFrame,
    time_after_event: float = 1.0,
    time_after_corner: float = 2.0,
    fps: int = 25
) -> pl.DataFrame:
    """
    Extract attacker target positions.

    Defined as positions 1.0s after the first on-ball event, or 2.0s after the corner
    if no event occurs within the window.

    :param corner_event: Corner event dictionary.
    :param tracking_df: Full match tracking data.
    :param events_df: Events data to find the first on-ball event.
    :param time_after_event: Seconds after first event to extract (default 1.0).
    :param time_after_corner: Seconds after corner to extract if no event (default 2.0).
    :param fps: Frames per second.
    :return: DataFrame with columns [player_id, x_m, y_m].
    """
    frame_start = corner_event["frame_start"]
    period = corner_event["period"]

    # Find first on-ball event after corner
    on_ball_filter = (
        (pl.col("period") == period) &
        (pl.col("frame_start") > frame_start) &
        (pl.col("frame_start") <= frame_start + int(time_after_corner * fps))
    )
    if "event_type" in events_df.columns:
        on_ball_filter &= pl.col("event_type").is_in(ON_BALL_EVENT_TYPES)

    next_events = events_df.filter(on_ball_filter).sort("frame_start")

    if next_events.height > 0:
        # Use 1 second after first event
        first_event_frame = next_events[0, "frame_start"]
        target_frame = int(first_event_frame + time_after_event * fps)
    else:
        # Use 2 seconds after corner
        target_frame = int(frame_start + time_after_corner * fps)

    # Try exact frame first
    positions = tracking_df.filter(
        (pl.col("period") == period) &
        (pl.col("frame") == target_frame) &
        (~pl.col("is_ball"))
    )

    # If empty, apply ±2 frame tolerance and pick closest frame
    if positions.height == 0:
        positions = tracking_df.filter(
            (pl.col("period") == period) &
            (pl.col("frame") >= target_frame - 2) &
            (pl.col("frame") <= target_frame + 2) &
            (~pl.col("is_ball"))
        ).with_columns(
            (pl.col("frame") - target_frame).abs().alias("frame_delta")
        )

        # Group by player and take the frame closest to target
        if positions.height > 0:
            min_delta = positions["frame_delta"].min()
            positions = positions.filter(pl.col("frame_delta") == min_delta)
            positions = positions.drop("frame_delta")

    # Ensure SkillCorner coordinates
    positions = ensure_skillcorner_xy(positions)

    player_team_map = corner_event.get("player_team_map")
    positions = assign_team_ids(positions, player_team_map)

    attacking_team_id = corner_event.get("attacking_team_id") or corner_event.get("team_id")
    if attacking_team_id is not None and "team_id_mapped" in positions.columns:
        positions = positions.with_columns(pl.col("team_id_mapped").alias("team_id")).drop("team_id_mapped")
    elif "team_id_mapped" in positions.columns:
        positions = positions.rename({"team_id_mapped": "team_id"})

    if attacking_team_id is not None and player_team_map and "team_id" in positions.columns:
        positions = positions.filter(pl.col("team_id").is_not_null())
        positions = positions.filter(pl.col("team_id") == attacking_team_id)

    s_x, s_y, used_fallback = resolve_flip_signs(corner_event, tracking_df, fps=fps)
    positions = canonicalize_positions_sc(positions, s_x, s_y)

    taker_id = corner_event.get("taker_player_id") or corner_event.get("player_in_possession_id")
    if taker_id is None:
        taker_id = infer_corner_taker_player_id(corner_event, events_df)
    if taker_id is not None:
        positions = positions.filter(pl.col("player_id") != taker_id)

    # Spatial exclusion: remove any player standing in the corner-taker zone
    # After canonicalization to the attacking top-right corner, the corner-spot is (SC_X_MAX, SC_Y_MAX)
    # Exclude points within CORNER_TAKER_RADIUS_M of that spot
    dx = pl.col("x_m") - SC_X_MAX
    dy = pl.col("y_m") - SC_Y_MAX
    positions = positions.filter((dx * dx + dy * dy) > (CORNER_TAKER_RADIUS_M ** 2))

    # Exclude backfield attackers from target clustering/visuals
    positions = positions.filter(pl.col("x_m") >= MIN_TARGET_X)

    positions = positions.with_columns([
        pl.col("x_m").clip(SC_X_MIN, SC_X_MAX).alias("x_m"),
        pl.col("y_m").clip(SC_Y_MIN, SC_Y_MAX).alias("y_m"),
    ])

    if used_fallback:
        global _fallback_warning_count, _fallback_warning_suppressed
        _fallback_warning_count += 1
        if _fallback_warning_count <= FALLBACK_WARNING_LIMIT:
            corner_id = corner_event.get("corner_id", "unknown")
            match_id = corner_event.get("match_id", "unknown")
            print(f"  Warning: Corner {corner_id} (match {match_id}) used metadata fallback for canonicalization (target)")
        elif not _fallback_warning_suppressed:
            print("  Warning: additional canonicalization fallbacks suppressed (target)")
            _fallback_warning_suppressed = True

    select_cols = ["player_id", "x_m", "y_m"]
    if "team_id" in positions.columns:
        select_cols.append("team_id")

    return positions.select(select_cols)



def fit_gmm_zones(
    corner_samples: List[Dict[str, pl.DataFrame]],
    n_init_components: int = 6,
    n_target_components: int = 10,
    penalty_area_bounds: Tuple[float, float, float, float] = (SC_PA_X_MIN, SC_PA_Y_MIN, SC_PA_X_MAX, SC_PA_Y_MAX),
    random_state: int = 42,
    restrict_targets_to_pa: bool = True
) -> ZoneModels:
    """
    Fit GMM models to initial and target positions of attacking players.

    matches the Routine Inspection methodology.

    :param corner_samples: List of dictionaries with 'initial' and 'target' DataFrames.
    :param n_init_components: Number of components for initial positions GMM.
    :param n_target_components: Number of components for target GMM (before restriction).
    :param penalty_area_bounds: Bounds (xmin, ymin, xmax, ymax) of the penalty area.
    :param random_state: Seed for reproducibility.
    :param restrict_targets_to_pa: Whether to restrict active target zones to the penalty area.
    :return: Trained ZoneModels object.
    """
    target_arrays: List[np.ndarray] = []
    target_player_ids: List[np.ndarray] = []
    initial_dfs: List[pl.DataFrame] = []

    for sample in corner_samples:
        init_df = sample.get('initial', pl.DataFrame())
        tgt_df = sample.get('target', pl.DataFrame())
        if init_df.height == 0 or tgt_df.height == 0:
            continue
        # Exclude backfield attackers from clustering input
        if 'x_m' in tgt_df.columns:
            tgt_df = tgt_df.filter(pl.col('x_m') >= MIN_TARGET_X)
        # Optionally restrict clustering to penalty area to avoid corner-taker clusters
        if restrict_targets_to_pa and all(c in tgt_df.columns for c in ['x_m','y_m']):
            x_min, y_min, x_max, y_max = penalty_area_bounds
            tgt_df = tgt_df.filter(
                (pl.col('x_m') >= x_min) & (pl.col('x_m') <= x_max) &
                (pl.col('y_m') >= y_min) & (pl.col('y_m') <= y_max)
            )
        if tgt_df.height == 0:
            continue
        target_arrays.append(tgt_df.select(['x_m', 'y_m']).to_numpy())
        target_player_ids.append(tgt_df['player_id'].to_numpy())
        initial_dfs.append(init_df)

    if not target_arrays:
        raise ValueError('No target positions available for GMM fitting')

    all_target_positions = np.vstack(target_arrays)
    total_initial = sum(df.height for df in initial_dfs)

    print('Fitting GMM zones...')
    print(f'  Initial positions: {total_initial:,}')
    print(f'  Target positions: {all_target_positions.shape[0]:,}')

    # If restricted to PA, cluster only 7 active zones directly
    n_tgt = 7 if restrict_targets_to_pa else n_target_components
    if all_target_positions.shape[0] < n_tgt:
        n_tgt = all_target_positions.shape[0]

    gmm_tgt = GaussianMixture(
        n_components=n_tgt,
        covariance_type='full',
        random_state=random_state,
        max_iter=500,
        n_init=5,
        reg_covar=1e-4
    )
    gmm_tgt.fit(all_target_positions)

    x_min, y_min, x_max, y_max = penalty_area_bounds
    pa_mask = (
        (all_target_positions[:, 0] >= x_min) &
        (all_target_positions[:, 0] <= x_max) &
        (all_target_positions[:, 1] >= y_min) &
        (all_target_positions[:, 1] <= y_max)
    )

    if restrict_targets_to_pa:
        # All components are already in PA
        active_zones = list(range(gmm_tgt.n_components))
    else:
        if pa_mask.sum() == 0:
            active_zones = [
                idx for idx, mean in enumerate(gmm_tgt.means_)
                if x_min <= mean[0] <= x_max and y_min <= mean[1] <= y_max
            ][:7]
        else:
            pa_positions = all_target_positions[pa_mask]
            pa_probs = gmm_tgt.predict_proba(pa_positions)
            pa_mass = pa_probs.sum(axis=0)
            active_zones = sorted(np.argsort(pa_mass)[-7:].tolist())

    initial_active_points: List[np.ndarray] = []
    for init_df, tgt_array, player_ids in zip(initial_dfs, target_arrays, target_player_ids):
        probs = gmm_tgt.predict_proba(tgt_array)
        probs_active = probs[:, active_zones]
        active_mask = probs_active.max(axis=1) >= TAU_ACTIVE
        if not active_mask.any():
            continue
        active_ids = set(player_ids[active_mask])
        if not (ACTIVE_ATTACKER_RANGE[0] <= len(active_ids) <= ACTIVE_ATTACKER_RANGE[1]):
            continue
        init_active = init_df.filter(pl.col('player_id').is_in(list(active_ids)))
        if init_active.height == 0:
            continue
        initial_active_points.append(init_active.select(['x_m', 'y_m']).to_numpy())

    if not initial_active_points:
        raise ValueError('No active attackers available for initial GMM fit')

    all_initial_active = np.vstack(initial_active_points)
    gmm_init = GaussianMixture(
        n_components=n_init_components,
        covariance_type='full',
        random_state=random_state,
        max_iter=500,
        n_init=5,
        reg_covar=1e-4
    )
    gmm_init.fit(all_initial_active)

    labels = gmm_init.predict(all_initial_active)
    covariances = gmm_init.covariances_
    precisions = [np.linalg.inv(cov) for cov in covariances]
    mahal_sq = []
    for point, comp in zip(all_initial_active, labels):
        diff = point - gmm_init.means_[comp]
        mahal_sq.append(float(diff @ precisions[comp] @ diff))
    mahal_sq = np.array(mahal_sq)
    mask = mahal_sq <= CHI2_THRESHOLD_SQ
    trimmed_initial = all_initial_active[mask]
    if trimmed_initial.shape[0] >= n_init_components:
        gmm_init.fit(trimmed_initial)

    print('OK GMM models fitted')
    print(f'  Initial zones: {n_init_components}')
    print(f'  Target active zones: {len(active_zones)}/{n_target_components}')

    return ZoneModels(
        gmm_init=gmm_init,
        gmm_tgt=gmm_tgt,
        active_tgt_ids=active_zones
    )


def encode_run_vector_42d(
    initial_df: pl.DataFrame,
    target_df: pl.DataFrame,
    zone_models: ZoneModels
) -> np.ndarray:
    """
    Encode a single corner as a 42-dimensional run vector using player-ID matching.
    Vector dimension is n_init_zones (6) * n_active_target_zones (7).

    :param initial_df: DataFrame with initial player positions.
    :param target_df: DataFrame with target player positions.
    :param zone_models: Trained GMM zone models.
    :return: Flattened 42-dimensional run vector (numpy array).
    """
    run_vector = np.zeros(42)

    if initial_df.height == 0 or target_df.height == 0:
        return run_vector

    init_coords = initial_df.select(['x_m', 'y_m']).to_numpy()
    tgt_coords = target_df.select(['x_m', 'y_m']).to_numpy()

    init_probs = zone_models.gmm_init.predict_proba(init_coords)  # (n_init, 6)
    # Select only active target components (7) if model has more comps
    tgt_resp = zone_models.gmm_tgt.predict_proba(tgt_coords)
    if zone_models.active_tgt_ids:
        target_probs = tgt_resp[:, zone_models.active_tgt_ids]
    else:
        target_probs = tgt_resp

    # Pad target_probs to have 7 columns if it has less
    if target_probs.shape[1] < 7:
        pad_width = 7 - target_probs.shape[1]
        target_probs = np.pad(target_probs, ((0, 0), (0, pad_width)), 'constant')

    # Renormalize target probs across active zones per player (avoid dropping players)
    row_sums = target_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    target_probs = target_probs / row_sums

    init_player_ids = initial_df['player_id'].to_numpy()
    target_player_ids = target_df['player_id'].to_numpy()
    player_index_map = {pid: idx for idx, pid in enumerate(init_player_ids)}

    for tprob, pid in zip(target_probs, target_player_ids):
        init_idx = player_index_map.get(pid)
        if init_idx is None:
            continue
        p_init = init_probs[init_idx]
        run_vector += np.outer(p_init, tprob).reshape(-1)

    return run_vector


def encode_all_corners(
    corners_df: pl.DataFrame,
    tracking_dict: Dict[int, pl.DataFrame],
    events_dict: Dict[int, pl.DataFrame],
    zone_models: ZoneModels,
    verbose: bool = True
) -> np.ndarray:
    """
    Encode all corners as 42-d run vectors.

    Process:
    1. Extracts initial and target positions for each corner.
    2. Encodes them using GMM zone distributions.

    :param corners_df: DataFrame containing all corners.
    :param tracking_dict: Dictionary mapping match_id to tracking DataFrame.
    :param events_dict: Dictionary mapping match_id to events DataFrame.
    :param zone_models: Trained ZoneModels object.
    :param verbose: Whether to print progress information.
    :return: Array of shape (N, 42) containing run vectors.
    """
    n_corners = corners_df.height
    run_vectors = np.zeros((n_corners, 42))
    team_maps: Dict[int, Dict[int, int]] = {}

    if verbose:
        print(f"Encoding {n_corners:,} corners as 42-d run vectors...")

    for idx, corner in enumerate(corners_df.iter_rows(named=True)):
        match_id = corner['match_id']
        if match_id not in tracking_dict or match_id not in events_dict:
            continue

        tracking_df = tracking_dict[match_id]
        events_df = events_dict[match_id]
        team_map = team_maps.setdefault(match_id, build_player_team_map(events_df))

        corner_payload = dict(corner)
        corner_payload['player_team_map'] = team_map
        corner_payload['attacking_team_id'] = corner.get('team_id')
        corner_payload['taker_player_id'] = corner.get('player_in_possession_id')

        try:
            init_df = extract_initial_positions(corner_payload, tracking_df, events_df)
            target_df = extract_target_positions(corner_payload, tracking_df, events_df)

            if init_df.height > 0 and target_df.height > 0:
                run_vectors[idx] = encode_run_vector_42d(init_df, target_df, zone_models)
        except Exception as exc:
            if verbose and idx < 10:
                print(f"  Warning: Corner {idx} (match {match_id}) failed: {exc}")

    if verbose:
        non_zero = (run_vectors.sum(axis=1) > 0).sum()
        print(f"OK Encoded {non_zero}/{n_corners} corners successfully")

    return run_vectors


def save_zone_models(zone_models: ZoneModels, output_path: Path) -> None:
    """
    Save fitted GMM models to disk using pickle.

    :param zone_models: ZoneModels object to save.
    :param output_path: Destination file path.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(zone_models, f)
    print(f"OK Saved GMM models to {output_path}")



def load_zone_models(input_path: Path) -> ZoneModels:
    """
    Load fitted GMM models from disk.

    :param input_path: path to the pickle file.
    :return: Loaded ZoneModels object.
    """
    with open(input_path, 'rb') as f:
        zone_models = pickle.load(f)
    print(f"OK Loaded GMM models from {input_path}")
    return zone_models



def visualize_zones(
    zone_models: ZoneModels,
    sample_positions_init: Optional[np.ndarray] = None,
    sample_positions_target: Optional[np.ndarray] = None,
    initial_active_mask: Optional[np.ndarray] = None,
    active_prob_threshold: float = TAU_ACTIVE,
) -> 'matplotlib.figure.Figure':
    """
    Reproduce Paper Figure 2: GMM zone visualizations with pitch overlay.
    Generates a 4-subplot figure showing target zones, initial zones, and active player distributions.

    :param zone_models: Fitted ZoneModels object.
    :param sample_positions_init: Optional sample of initial positions for scatter plot.
    :param sample_positions_target: Optional sample of target positions for scatter plot.
    :param initial_active_mask: Boolean mask indicating which initial positions correspond to active attackers.
    :param active_prob_threshold: Threshold for active attacker classification (default 0.3).
    :return: Matplotlib Figure with 4 subplots (2x2).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from mplsoccer import Pitch

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.patch.set_facecolor('#313332')

    # Define colormap for zones
    colors = plt.cm.tab10.colors

    # Helper function to draw half pitch
    def draw_pitch(ax):
        pitch = Pitch(
            pitch_type='custom',
            pitch_length=PITCH_LENGTH,
            pitch_width=PITCH_WIDTH,
            half=True,
            line_color='white',
            pitch_color='#22aa44'
        )
        pitch.draw(ax=ax)
        # Plotting happens in STANDARD meters; convert SC samples before scatter
        ax.set_xlim(PITCH_LENGTH / 2.0, PITCH_LENGTH)
        ax.set_ylim(0, PITCH_WIDTH)

    # Plot 1: Target positions scatter with pitch (upper left)
    ax = axes[0, 0]
    draw_pitch(ax)
    POINT_SIZE = 6
    POINT_ALPHA = 0.45
    if sample_positions_target is not None:
        tgt_std = np.column_stack((sample_positions_target[:, 0] + PITCH_CX,
                                   sample_positions_target[:, 1] + PITCH_CY))
        ax.scatter(tgt_std[:, 0], tgt_std[:, 1],
                   s=POINT_SIZE, alpha=POINT_ALPHA,
                   c='black', edgecolors='none', label='All attackers')
        from matplotlib.lines import Line2D
        leg = ax.legend(
            handles=[Line2D([0],[0], marker='o', color='w', label='All attackers',
                            markerfacecolor='black', markeredgecolor='black', markersize=6)],
            loc='upper left', frameon=True
        )
        # Make legend text white and box transparent for dark background
        for txt in leg.get_texts():
            txt.set_color('w')
        leg.get_frame().set_alpha(0.0)
        leg.get_frame().set_facecolor('none')
        leg.get_frame().set_edgecolor('none')
    ax.set_title("Target positions (all attackers)", fontsize=14, fontweight='bold', color='w')
    ax.set_facecolor('#313332')

    # Plot 2: Initial positions with active players highlighted (upper right)
    ax = axes[0, 1]
    draw_pitch(ax)
    if sample_positions_init is not None:
        init_std = np.column_stack((sample_positions_init[:, 0] + PITCH_CX,
                                    sample_positions_init[:, 1] + PITCH_CY))
        if initial_active_mask is not None and len(initial_active_mask) == init_std.shape[0]:
            # inactive first as background
            inactive = ~initial_active_mask
            if inactive.any():
                ax.scatter(init_std[inactive, 0], init_std[inactive, 1],
                           s=POINT_SIZE, alpha=0.25, c='black', edgecolors='none')
            if initial_active_mask.any():
                ax.scatter(init_std[initial_active_mask, 0], init_std[initial_active_mask, 1],
                           s=POINT_SIZE+2, alpha=0.7, c='#1877f2', edgecolors='none', label='Active attackers')
                leg2 = ax.legend(loc='upper left', frameon=True)
                for txt in leg2.get_texts():
                    txt.set_color('w')
                leg2.get_frame().set_alpha(0.0)
                leg2.get_frame().set_facecolor('none')
                leg2.get_frame().set_edgecolor('none')
        else:
            # no mask provided
            ax.scatter(init_std[:, 0], init_std[:, 1], s=POINT_SIZE,
                       alpha=POINT_ALPHA, c='black', edgecolors='none')
    ax.set_title("Initial positions (active in blue)", fontsize=14, fontweight='bold', color='w')
    ax.set_facecolor('#313332')

    # Plot 3: Target GMM (15 comps) with active zones highlighted (lower left)
    ax = axes[1, 0]
    draw_pitch(ax)
    # background target points
    if sample_positions_target is not None:
        # Restrict background points to penalty area to avoid a corner-taker cluster
        mask_pa = (
            (sample_positions_target[:, 0] >= SC_PA_X_MIN) &
            (sample_positions_target[:, 0] <= SC_PA_X_MAX) &
            (sample_positions_target[:, 1] >= SC_PA_Y_MIN) &
            (sample_positions_target[:, 1] <= SC_PA_Y_MAX)
        )
        pts = sample_positions_target[mask_pa] if mask_pa.any() else sample_positions_target
        tgt_std = np.column_stack((pts[:, 0] + PITCH_CX,
                                   pts[:, 1] + PITCH_CY))
        ax.scatter(tgt_std[:, 0], tgt_std[:, 1], s=POINT_SIZE, alpha=0.15,
                   c='black', edgecolors='none')

    legend_elements = []
    # Ellipses: 15 components, active in blue (a-g)
    BLUE = '#1877f2'
    for idx, (mean, covar) in enumerate(zip(zone_models.gmm_tgt.means_,
                                             zone_models.gmm_tgt.covariances_)):
        v, w = np.linalg.eigh(covar)
        angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # 95% conf ellipse

        is_active = idx in zone_models.active_tgt_ids
        color = BLUE if is_active else 'gray'
        alpha = 0.35 if is_active else 0.12
        lw = 2.5 if is_active else 1.0
        label_char = None
        if is_active:
            # order within active list
            rank = zone_models.active_tgt_ids.index(idx)
            label_char = chr(ord('a') + rank)

        mean_std = mean + np.array([PITCH_CX, PITCH_CY])
        ell = Ellipse(mean_std, v[0], v[1], angle=angle,
                      edgecolor=color, facecolor=color, alpha=alpha, linewidth=lw)
        ax.add_patch(ell)
        if label_char is not None:
            ax.text(mean_std[0], mean_std[1], label_char, fontsize=14, ha='center',
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='circle', facecolor=color, alpha=0.85))

        n_total = zone_models.gmm_tgt.n_components
    n_active = len(zone_models.active_tgt_ids)
    active_labels = 'a–' + chr(ord('a') + n_active - 1) if n_active > 0 else 'none'
    ax.set_title(f"Target zones ({n_total}-comp GMM; {active_labels} active)", fontsize=14, fontweight='bold', color='w')

    # Plot 4: Initial GMM (6 comps) on active players (lower right)
    ax = axes[1, 1]
    draw_pitch(ax)

    # background initial points (active highlighted)
    if sample_positions_init is not None:
        init_std = np.column_stack((sample_positions_init[:, 0] + PITCH_CX,
                                    sample_positions_init[:, 1] + PITCH_CY))
        if initial_active_mask is not None and len(initial_active_mask) == init_std.shape[0]:
            inactive = ~initial_active_mask
            if inactive.any():
                ax.scatter(init_std[inactive, 0], init_std[inactive, 1],
                           s=POINT_SIZE, alpha=0.15, c='black', edgecolors='none')
            if initial_active_mask.any():
                ax.scatter(init_std[initial_active_mask, 0], init_std[initial_active_mask, 1],
                           s=POINT_SIZE, alpha=0.25, c='#1877f2', edgecolors='none')
        else:
            ax.scatter(init_std[:, 0], init_std[:, 1], s=POINT_SIZE, alpha=0.15,
                       c='black', edgecolors='none')

    legend_elements = []
    for idx, (mean, covar) in enumerate(zip(zone_models.gmm_init.means_,
                                             zone_models.gmm_init.covariances_)):
        v, w = np.linalg.eigh(covar)
        angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # 95% conf ellipse

        color = colors[idx % len(colors)]
        mean_std = mean + np.array([PITCH_CX, PITCH_CY])
        ell = Ellipse(mean_std, v[0], v[1], angle=angle,
                      edgecolor=color, facecolor=color, alpha=0.30, linewidth=2.2)
        ax.add_patch(ell)
        ax.text(mean_std[0], mean_std[1], str(idx + 1), fontsize=14, ha='center',
                fontweight='bold', color='white',
                bbox=dict(boxstyle='circle', facecolor=color, alpha=0.85))

        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor=color, edgecolor=color, label=f'Zone {idx+1}'))

    ax.set_title("Initial zones (6-comp GMM on active)", fontsize=14, fontweight='bold', color='w')
    leg3 = ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
              frameon=True, title='Initial Zones')
    for txt in leg3.get_texts():
        txt.set_color('w')
    if leg3.get_title() is not None:
        leg3.get_title().set_color('w')
    leg3.get_frame().set_alpha(0.0)
    leg3.get_frame().set_facecolor('none')
    leg3.get_frame().set_edgecolor('none')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("CTI GMM Zones Module loaded")
    print("Functions: fit_gmm_zones, encode_run_vector_42d, visualize_zones")

def assign_team_ids(
    positions_df: pl.DataFrame,
    player_team_map: Optional[Dict[int, int]]
) -> pl.DataFrame:
    """
    Attach team_id to tracking positions using a player->team lookup if available.

    :param positions_df: DataFrame with player positions.
    :param player_team_map: Dictionary mapping player_id to team_id.
    :return: DataFrame with added 'team_id' column (or 'team_id_mapped').
    """
    if positions_df.height == 0:
        return positions_df

    if "team_id" in positions_df.columns and positions_df["team_id"].null_count() == 0:
        return positions_df

    if not player_team_map:
        return positions_df

    mapping_series = pl.Series(
        name="team_id_mapped",
        values=[player_team_map.get(pid) if pid is not None else None for pid in positions_df["player_id"].to_list()]
    )

    if "team_id" in positions_df.columns:
        positions_df = positions_df.drop("team_id")

    return positions_df.with_columns(mapping_series)

def build_player_team_map(events_df: pl.DataFrame) -> Dict[int, int]:
    """
    Build player_id -> team_id mapping from events data.

    :param events_df: DataFrame containing event data with player_id and team_id.
    :return: Dictionary mapping player_id to team_id.
    """
    if "player_id" not in events_df.columns or "team_id" not in events_df.columns:
        return {}

    mapping = (
        events_df
        .select(["player_id", "team_id"])
        .drop_nulls()
        .unique(subset=["player_id"], keep="first")
    )
    return {row["player_id"]: row["team_id"] for row in mapping.iter_rows(named=True)}
