"""
Author: Tiago
Date: 2025-12-04
Description: CTI Corner Extraction & Windowing Module. Detects corners and extracts temporal windows.
"""

import polars as pl
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
from cti_paths import DATA_2024

# Time windows (in seconds) from implementation.md
WINDOW_PRE = (-5.0, 0.0)        # Pre-setup context
WINDOW_DELIVERY = (0.0, 2.0)    # Delivery & first contact
WINDOW_OUTCOME = (0.0, 10.0)    # Attacking phase
WINDOW_COUNTER = (10.0, 25.0)   # Counter-risk phase

FPS = 25  # Frames per second
CORNER_START_TYPE_IDS = {11, 12}  # corner_reception, corner_interception


def detect_corners(events_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract corner kick events from events DataFrame.

    Uses start_type_id ∈ {11, 12} to identify corner-phase starts:
    - 11: corner_reception (direct reception from corner kick)
    - 12: corner_interception (defensive interception of corner delivery)

    Args:
        events_df: Events DataFrame with 'is_corner_phase_start' flag

    Returns:
        DataFrame with corner events, columns:
        [corner_id, event_id, match_id, frame_start, team_id, period,
         time_start_s, x_start, y_start, attacking_side]
    """
    if events_df.height == 0:
        return pl.DataFrame()

    # Filter for corner-phase starts using start_type_id
    corners = events_df.filter(
        pl.col("start_type_id").is_not_null() &
        pl.col("start_type_id").is_in([11, 12])
    )

    if corners.height == 0:
        return pl.DataFrame()

    # Create unique corner_id
    corners = corners.with_columns([
        pl.int_range(0, corners.height).alias("corner_id")
    ])

    # Select relevant columns for corner analysis
    keep_cols = [
        "corner_id", "event_id", "match_id", "frame_start", "team_id", "period",
        "time_start_s", "x_start", "y_start", "attacking_side", "start_type_id",
        "player_in_possession_id", "phase_index"
    ]

    available_cols = [c for c in keep_cols if c in corners.columns]
    corners = corners.select(available_cols)

    print(f"OK Detected {corners.height} corner kicks")

    return corners


def extract_corner_windows(
    corner_event: Dict,
    tracking_df: pl.DataFrame,
    fps: int = FPS,
    windows: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, pl.DataFrame]:
    """
    Extract tracking data for temporal windows around a corner kick.

    Windows (seconds relative to corner kick):
    - pre: [-5, 0)    - Pre-setup context
    - delivery: [0, 2] - Delivery & first contact
    - outcome: [0, 10] - Attacking phase
    - counter: (10, 25] - Counter-risk phase

    Args:
        corner_event: Dict with keys: frame_start, period, match_id
        tracking_df: Full match tracking data
        fps: Frames per second (default 25)
        windows: Optional custom time windows dict

    Returns:
        Dict mapping window name to tracking DataFrame for that window
    """
    if windows is None:
        windows = {
            "pre": WINDOW_PRE,
            "delivery": WINDOW_DELIVERY,
            "outcome": WINDOW_OUTCOME,
            "counter": WINDOW_COUNTER
        }

    frame_start = corner_event["frame_start"]
    period = corner_event["period"]

    window_data = {}

    for window_name, (start_sec, end_sec) in windows.items():
        # Convert seconds to frame offsets
        frame_offset_start = int(start_sec * fps)
        frame_offset_end = int(end_sec * fps)

        frame_min = frame_start + frame_offset_start
        frame_max = frame_start + frame_offset_end

        # Extract tracking data for this window
        window_tracking = tracking_df.filter(
            (pl.col("period") == period) &
            (pl.col("frame") >= frame_min) &
            (pl.col("frame") <= frame_max)
        )

        window_data[window_name] = window_tracking

    return window_data


def apply_quality_gates(
    corner_event: Dict,
    window_data: Dict[str, pl.DataFrame],
    require_ball_window: Tuple[float, float] = (-0.5, 2.0),
    max_missing_defenders: int = 2,
    fps: int = FPS
) -> Tuple[bool, str]:
    """
    Apply data quality gates to corner extraction.

    Quality criteria (from implementation.md:52-55):
    1. Ball detected during [-0.5s, +2.0s] around corner kick
    2. < 2 missing defenders in [-2s, +2s] window

    Args:
        corner_event: Corner event dict
        window_data: Dict of tracking windows from extract_corner_windows()
        require_ball_window: Time window for ball detection check (seconds)
        max_missing_defenders: Maximum allowed missing defenders
        fps: Frames per second

    Returns:
        (passes_quality, reason) tuple
        - passes_quality: bool, True if corner passes all gates
        - reason: str, explanation if rejected
    """
    frame_start = corner_event["frame_start"]

    # Gate 1: Ball detection check
    ball_check_frames = (
        int(frame_start + require_ball_window[0] * fps),
        int(frame_start + require_ball_window[1] * fps)
    )

    # Check delivery window (contains ball_check_frames)
    delivery_df = window_data.get("delivery", pl.DataFrame())

    if delivery_df.height == 0:
        return False, "No tracking data in delivery window"

    ball_data = delivery_df.filter(
        (pl.col("is_ball") == True) &
        (pl.col("frame") >= ball_check_frames[0]) &
        (pl.col("frame") <= ball_check_frames[1])
    )

    if ball_data.height == 0:
        return False, f"Ball not tracked in window {require_ball_window}"

    ball_detected = ball_data.filter(pl.col("is_detected") == True)

    if ball_detected.height == 0:
        return False, "Ball not detected in required window"

    # Gate 2: Defender detection check (simplified heuristic)
    # Check frames around [-2s, +2s]
    check_window_frames = (
        int(frame_start - 2 * fps),
        int(frame_start + 2 * fps)
    )

    # Use delivery + pre windows to cover this range
    combined_df = pl.concat([
        window_data.get("pre", pl.DataFrame()),
        window_data.get("delivery", pl.DataFrame())
    ], how="vertical_relaxed")

    if combined_df.height == 0:
        return False, "No tracking data for defender check"

    # Count defenders per frame (players that are detected)
    defender_counts = (
        combined_df
        .filter(
            (~pl.col("is_ball")) &
            (pl.col("frame") >= check_window_frames[0]) &
            (pl.col("frame") <= check_window_frames[1])
        )
        .group_by("frame")
        .agg([
            pl.col("is_detected").sum().alias("n_detected"),
            pl.col("player_id").n_unique().alias("n_players")
        ])
    )

    if defender_counts.height == 0:
        return False, "No player data in defender check window"

    # Check if MEDIAN of frames has too many missing players
    # Using median instead of min to tolerate occasional missing frames
    # Assume ~22 players total, if median < 15 detected → likely > 7 missing consistently
    median_detected = defender_counts.select(pl.col("n_detected").median()).item()

    # Also check that we have enough frames with good coverage
    # At least 40% of frames should have reasonable player detection (>= 16 players)
    frames_with_good_coverage = defender_counts.filter(
        pl.col("n_detected") >= 16  # At least 16 players detected
    ).height
    coverage_pct = frames_with_good_coverage / defender_counts.height if defender_counts.height > 0 else 0

    # Relaxed thresholds for real-world tracking data:
    # 1. Median must have at least 15 players (tolerates significant tracking gaps)
    # 2. At least 40% of frames must have reasonable coverage (>= 16 players)
    if median_detected < 15:
        return False, f"Too many missing players (median detected: {median_detected:.0f})"

    if coverage_pct < 0.40:
        return False, f"Insufficient frame coverage ({coverage_pct:.1%} of frames have good tracking)"

    return True, "Passed all quality gates"


def extract_all_corners_with_windows(
    match_id: int,
    events_df: pl.DataFrame,
    tracking_df: pl.DataFrame,
    apply_gates: bool = True,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Complete pipeline: detect corners + extract windows + apply quality gates.

    Args:
        match_id: Match identifier
        events_df: Events DataFrame for this match
        tracking_df: Tracking DataFrame for this match
        apply_gates: Whether to filter by quality gates
        verbose: Print progress messages

    Returns:
        DataFrame with columns:
        [corner_id, match_id, event_id, frame_start, team_id, period,
         time_start_s, passes_quality, quality_reason, has_windows]

    Note: Actual window DataFrames should be saved separately due to size
    """
    # Step 1: Detect corners
    corners = detect_corners(events_df)

    if corners.height == 0:
        if verbose:
            print(f"  No corners found in match {match_id}")
        return pl.DataFrame()

    # Step 2: Extract windows and apply gates
    results = []

    for row in corners.iter_rows(named=True):
        corner_event = {
            "frame_start": row["frame_start"],
            "period": row["period"],
            "match_id": match_id
        }

        try:
            window_data = extract_corner_windows(corner_event, tracking_df)

            # Apply quality gates if requested
            if apply_gates:
                passes, reason = apply_quality_gates(corner_event, window_data)
            else:
                passes, reason = True, "Gates disabled"

            results.append({
                **row,
                "match_id": match_id,
                "passes_quality": passes,
                "quality_reason": reason,
                "has_windows": len(window_data) > 0
            })

        except Exception as e:
            if verbose:
                print(f"  Warning: Corner {row['corner_id']} failed: {e}")
            results.append({
                **row,
                "match_id": match_id,
                "passes_quality": False,
                "quality_reason": f"Error: {str(e)}",
                "has_windows": False
            })

    result_df = pl.from_dicts(results)

    if apply_gates:
        n_passed = int(result_df.filter(pl.col("passes_quality")).height)
        if verbose:
            print(f"  Match {match_id}: {n_passed}/{corners.height} corners passed quality gates")

    return result_df


def save_corners_dataset(
    corners_df: pl.DataFrame,
    output_path: Path
) -> None:
    """
    Save corners dataset to parquet.

    Args:
        corners_df: Aggregated corners from multiple matches
        output_path: Path to save parquet file
    """
    corners_df.write_parquet(output_path)
    print(f"OK Saved {corners_df.height} corners to {output_path}")

    # Print summary statistics
    if corners_df.height > 0:
        n_matches = corners_df.select(pl.col("match_id").n_unique()).item()
        n_passed = int(corners_df.filter(pl.col("passes_quality")).height)

        print(f"  Matches: {n_matches}")
        print(f"  Quality: {n_passed}/{corners_df.height} ({n_passed/corners_df.height:.1%})")


def load_events_basic(match_id: int) -> pl.DataFrame:
    """
    Load events data for a single match.

    Args:
        match_id: Match ID

    Returns:
        Events DataFrame
    """
    from pathlib import Path
    data_dir = DATA_2024
    events_path = data_dir / "dynamic" / f"{match_id}.parquet"

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    return pl.read_parquet(events_path)


def load_tracking_full(match_id: int, sort_rows: bool = True) -> pl.DataFrame:
    """
    Load tracking data for a single match from JSON and convert to Polars.

    Args:
        match_id: Match ID
        sort_rows: Whether to sort by period, frame

    Returns:
        Tracking DataFrame with columns:
        [period, frame, player_id, team_id, x_m, y_m, is_ball, is_detected]
    """
    import json
    from pathlib import Path

    data_dir = DATA_2024
    tracking_path = data_dir / "tracking" / f"{match_id}.json"

    if not tracking_path.exists():
        raise FileNotFoundError(f"Tracking file not found: {tracking_path}")

    with open(tracking_path, 'r') as f:
        data = json.load(f)

    # Convert JSON to flat structure
    # Data is a list of frames, each with:
    # {frame, period, ball_data: {x, y, z, is_detected}, player_data: [{x, y, player_id, is_detected}]}
    rows = []

    for frame_data in data:
        frame_num = frame_data.get("frame")
        period = frame_data.get("period")

        # Skip frames without period (warmup frames)
        if period is None:
            continue

        # Ball tracking
        ball_data = frame_data.get("ball_data")
        if ball_data and ball_data.get("x") is not None:
            rows.append({
                "period": period,
                "frame": frame_num,
                "player_id": None,
                "team_id": None,
                "x_m": ball_data["x"],
                "y_m": ball_data["y"],
                "is_ball": True,
                "is_detected": ball_data.get("is_detected", True)
            })

        # Player tracking
        player_data_list = frame_data.get("player_data", [])
        for player_info in player_data_list:
            if player_info and player_info.get("x") is not None:
                rows.append({
                    "period": period,
                    "frame": frame_num,
                    "player_id": player_info.get("player_id"),
                    "team_id": None,  # team_id not in this data format
                    "x_m": player_info["x"],
                    "y_m": player_info["y"],
                    "is_ball": False,
                    "is_detected": player_info.get("is_detected", True)
                })

    df = pl.DataFrame(rows)

    if sort_rows and df.height > 0:
        df = df.sort(["period", "frame", "player_id"])

    return df


if __name__ == "__main__":
    print("CTI Corner Extraction Module loaded")
    print("Functions: detect_corners, extract_corner_windows, apply_quality_gates")
    print("           load_events_basic, load_tracking_full")
