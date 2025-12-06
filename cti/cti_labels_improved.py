"""
Author: Tiago
Date: 2025-12-04
Description: Improved Label Strategy - Domain-Driven Approach. Implements interpretable, balanced labels for corner threat assessment (y1-y5).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Dict


def parse_time_to_seconds(time_str: str) -> float:
    """
    Convert time string like "00:04.2" (MM:SS.ms) to float seconds.

    Args:
        time_str: Time in format "MM:SS.ms" or similar.

    Returns:
        Time in seconds as float. Returns 0.0 on parsing failure.
    """
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        # Fallback: try to parse as float directly
        return float(time_str)
    except Exception:
        return 0.0


def _ensure_time_seconds(events_df: pl.DataFrame, time_col: str = "time_start") -> tuple[pl.DataFrame, str]:
    """
    Ensure events_df has a numeric time column in seconds.

    If `time_start` is Utf8 ("MM:SS.ms"), derive `time_start_seconds`.
    Otherwise, use `time_start` as-is.

    Returns:
        (events_df_with_seconds, time_column_name)
    """
    if time_col not in events_df.columns:
        raise KeyError(f"Expected column '{time_col}' in events_df.")

    if events_df[time_col].dtype == pl.Utf8:
        events_df = events_df.with_columns(
            (
                pl.col(time_col).str.split(":").list.get(0).cast(pl.Float64) * 60.0
                + pl.col(time_col).str.split(":").list.get(1).cast(pl.Float64)
            ).alias("time_start_seconds")
        )
        return events_df, "time_start_seconds"
    else:
        return events_df, time_col


def build_corner_xthreat_model(corners_df: pl.DataFrame, events_dict: Dict[int, pl.DataFrame]) -> Dict[str, Dict[str, float]]:
    """
    Build historical model of corner danger by delivery zone (based on events).

    Assumes:
        - Event positions (x, y) are in a 0–105 / 0–68 style pitch (e.g., Wyscout-like).
        - `time_start` contains the event time, either numeric or "MM:SS.ms".

    Returns:
        Dictionary:
            delivery_zone -> {
                'xthreat_corner', 'p_shot', 'p_goal', 'n_corners'
            }
    """
    print("\n[Corner xThreat Model] Building historical delivery zone statistics...")

    corner_features = []

    for corner_idx, corner in enumerate(corners_df.iter_rows(named=True)):
        match_id = corner["match_id"]

        if match_id not in events_dict:
            continue

        events_df = events_dict[match_id]
        events_df, time_col = _ensure_time_seconds(events_df, "time_start")

        FPS = 25.0
        frame_start = corner["frame_start"]
        period = corner["period"]
        team_id = corner["team_id"]

        # Approximate corner timestamp from frame number
        corner_timestamp = frame_start / FPS

        # Try to refine using corner events if available
        corner_events = (
            events_df.filter(
                (pl.col("start_type_id").is_not_null())
                & (pl.col("start_type_id").is_in([11, 12]))  # corner_reception / corner_interception
                & (pl.col("period") == period)
                & (pl.col("frame_start") == frame_start)
            )
        )

        if len(corner_events) > 0 and time_col in corner_events.columns:
            event_ts = corner_events.row(0, named=True).get(time_col, corner_timestamp)
            if isinstance(event_ts, (int, float)):
                corner_timestamp = float(event_ts)

        # Events in first 3 seconds after corner timestamp (delivery window)
        delivery_events = (
            events_df.filter(
                (pl.col(time_col) >= corner_timestamp)
                & (pl.col(time_col) <= corner_timestamp + 3.0)
                & (pl.col("period") == period)
            )
            .sort(time_col)
        )

        if len(delivery_events) == 0:
            continue

        first_touch = delivery_events.row(0, named=True)
        target_x = first_touch.get("x", 50.0)
        target_y = first_touch.get("y", 34.0)

        # Discretize delivery zone (4x3 grid in attacking third in event coords)
        # X zones: 0=midfield, 1=edge of box, 2=penalty area, 3=6-yard box
        if target_x < 88.5:  # before penalty area
            zone_x = 0
        elif target_x < 94.5:  # front half of box
            zone_x = 1
        elif target_x < 100.5:  # back half of box
            zone_x = 2
        else:  # six-yard box
            zone_x = 3

        # Y zones: 0=far post, 1=central, 2=near post
        if target_y < 30.5:
            zone_y = 0
        elif target_y < 37.5:
            zone_y = 1
        else:
            zone_y = 2

        delivery_zone = f"{zone_x}_{zone_y}"

        # Shot within 10s
        shot_events = events_df.filter(
            (pl.col("end_type") == "shot")
            & (pl.col(time_col) >= corner_timestamp)
            & (pl.col(time_col) <= corner_timestamp + 20.0)
            & (pl.col("team_id") == team_id)
        )
        had_shot = 1 if len(shot_events) > 0 else 0

        # Goal within 10s
        goal_events = events_df.filter(
            (pl.col("end_type") == "goal")
            & (pl.col(time_col) >= corner_timestamp)
            & (pl.col(time_col) <= corner_timestamp + 20.0)
            & (pl.col("team_id") == team_id)
        )
        had_goal = 1 if len(goal_events) > 0 else 0

        corner_features.append(
            {
                "delivery_zone": delivery_zone,
                "had_shot": had_shot,
                "had_goal": had_goal,
                "zone_x": zone_x,
                "zone_y": zone_y,
            }
        )

    if len(corner_features) == 0:
        print("  WARNING: No corner features extracted, using default model")
        return {}

    corner_stats_df = pl.DataFrame(corner_features)

    zone_stats = corner_stats_df.group_by("delivery_zone").agg(
        [
            pl.col("had_shot").mean().alias("p_shot"),
            pl.col("had_goal").mean().alias("p_goal"),
            pl.col("had_shot").sum().alias("n_shots"),
            pl.col("had_goal").sum().alias("n_goals"),
            pl.len().alias("n_corners"),
        ]
    )

    # Corner-specific xThreat as simple weighted combination
    zone_stats = zone_stats.with_columns(
        (0.7 * pl.col("p_shot") + 1.0 * pl.col("p_goal")).alias("xthreat_corner")
    )

    xthreat_model: Dict[str, Dict[str, float]] = {}
    for row in zone_stats.iter_rows(named=True):
        xthreat_model[row["delivery_zone"]] = {
            "xthreat_corner": float(row["xthreat_corner"]),
            "p_shot": float(row["p_shot"]),
            "p_goal": float(row["p_goal"]),
            "n_corners": int(row["n_corners"]),
        }

    print(f"  Built xThreat model for {len(xthreat_model)} delivery zones")
    print(f"  Total corners analyzed: {len(corner_features)}")
    print(f"  Overall shot rate: {corner_stats_df['had_shot'].mean():.1%}")
    print(f"  Overall goal rate: {corner_stats_df['had_goal'].mean():.1%}")

    # Top 5 most dangerous zones
    zone_stats_sorted = zone_stats.sort("xthreat_corner", descending=True).head(5)
    print("\n  Top 5 most dangerous delivery zones:")
    for row in zone_stats_sorted.iter_rows(named=True):
        print(
            f"    Zone {row['delivery_zone']}: "
            f"xT={row['xthreat_corner']:.3f} "
            f"(P(shot)={row['p_shot']:.1%}, P(goal)={row['p_goal']:.1%}, n={row['n_corners']})"
        )

    return xthreat_model


def compute_y2_corner_xthreat(
    corner: dict, events_df: pl.DataFrame, xthreat_model: Dict[str, Dict[str, float]]
) -> float:
    """
    Compute y2 (corner danger) based on historical xThreat model by delivery zone.

    Uses the same delivery-zone discretisation as `build_corner_xthreat_model`.

    Returns:
        Corner-specific xThreat value (typically in [0.0, ~0.5]).
    """
    corner_timestamp = corner.get("timestamp", 0.0)
    period = corner["period"]

    events_df, time_col = _ensure_time_seconds(events_df, "time_start")

    # Delivery window: 0–3s after corner timestamp
    delivery_events = (
        events_df.filter(
            (pl.col(time_col) > corner_timestamp)
            & (pl.col(time_col) <= corner_timestamp + 3.0)
            & (pl.col("period") == period)
        )
        .sort(time_col)
    )

    if len(delivery_events) == 0:
        return 0.0  # failed or very poor delivery

    first_touch = delivery_events.row(0, named=True)
    target_x = first_touch.get("x", 50.0)
    target_y = first_touch.get("y", 34.0)

    # Same discretisation as model
    if target_x < 88.5:
        zone_x = 0
    elif target_x < 94.5:
        zone_x = 1
    elif target_x < 100.5:
        zone_x = 2
    else:
        zone_x = 3

    if target_y < 30.5:
        zone_y = 0
    elif target_y < 37.5:
        zone_y = 1
    else:
        zone_y = 2

    delivery_zone = f"{zone_x}_{zone_y}"

    if delivery_zone in xthreat_model:
        return float(xthreat_model[delivery_zone]["xthreat_corner"])
    else:
        # Conservative fallback for unseen zones
        return 0.05


def detect_counter_attack(
    corner: dict,
    tracking_df: pl.DataFrame,
    events_df: pl.DataFrame,
    team_id_attacking: int,
) -> int:
    """
    Detect counter-attack using a simplified, tracking-enhanced definition:

    Conditions:
    - Defending team (opponent) has an event (possession indication) within 0–7s after the corner.
    - Attacking team does NOT regain possession within 3s after the first defending event.
    - During this defending possession, the ball (SkillCorner x_m) either:
        - crosses midfield (x_m changes sign), OR
        - advances at least MIN_ADVANCE_DISTANCE toward the direction of the defending attack.

    Coordinate assumptions:
    - tracking_df.x_m in SkillCorner [-52.5, 52.5], midfield at 0.0.
    - corner["x_start"] is also in SkillCorner coordinates.

    Returns:
        1 if counter-attack detected, 0 otherwise.
    """
    frame_start = corner["frame_start"]
    period = corner["period"]
    fps = 25

    # DEBUG
    print(f"\n[DEBUG y3] corner_id={corner.get('corner_id', 'N/A')} match_id={corner.get('match_id', 'N/A')} frame_start={frame_start}")

    counter_window_frames = int(7 * fps)  # 0–7s after corner
    frame_end = frame_start + counter_window_frames

    # STEP 1: defending team possession (events-based)
    defending_events = (
        events_df.filter(
            (pl.col("frame_start") > frame_start)
            & (pl.col("frame_start") <= frame_end)
            & (pl.col("period") == period)
            & (pl.col("team_id") != team_id_attacking)
        )
        .sort("frame_start")
    )

    # DEBUG
    print(f"[DEBUG y3] Found {len(defending_events)} defending events in window.")

    if len(defending_events) == 0:
        return 0  # defending team never touches ball in window

    first_defending_event = defending_events.row(0, named=True)
    defending_frame = int(first_defending_event["frame_start"])

    # STEP 1b: ensure attacking team does not quickly regain possession (<3s)
    subsequent_attacking_events = events_df.filter(
        (pl.col("frame_start") > defending_frame)
        & (pl.col("frame_start") <= defending_frame + int(3.0 * fps))
        & (pl.col("period") == period)
        & (pl.col("team_id") == team_id_attacking)
    )

    # DEBUG
    print(f"[DEBUG y3] Found {len(subsequent_attacking_events)} subsequent attacking events.")

    if len(subsequent_attacking_events) > 0:
        return 0  # no sustained counter

    # STEP 2: ball movement during defending possession (tracking-based)

    # DEBUG
    print(f"[DEBUG y3] 'is_ball' in tracking_df.columns: {'is_ball' in tracking_df.columns}")

    if "is_ball" not in tracking_df.columns:
        # Without a ball-flag we cannot safely isolate the ball trajectory -> no detection
        return 0

    ball_positions = (
        tracking_df.filter(
            (pl.col("frame") >= defending_frame)
            & (pl.col("frame") <= frame_end)
            & (pl.col("period") == period)
            & (pl.col("is_ball") == True)
        )
        .sort("frame")
    )

    # DEBUG
    print(f"[DEBUG y3] Found {len(ball_positions)} ball positions in tracking data.")

    if len(ball_positions) < 2:
        return 0  # not enough tracking data

    start_row = ball_positions.row(0, named=True)
    end_row = ball_positions.row(-1, named=True)

    # SkillCorner coordinates: [-52.5, 52.5], midfield 0.0
    start_x = float(start_row.get("x_m", 0.0))
    end_x = float(end_row.get("x_m", 0.0))

    midfield_x = 0.0
    corner_x = float(corner.get("x_start", 0.0))

    # DEBUG
    print(f"[DEBUG y3] Ball start_x={start_x:.2f}, end_x={end_x:.2f}, corner_x={corner_x:.2f}")

    # Defending team attack direction:
    # - If attacking team takes corner on left side (x_start < 0), they attack left (-x),
    #   so defending team attacks right (+x).
    # - If attacking team takes corner on right side (x_start >= 0), they attack right (+x),
    #   so defending team attacks left (-x).
    MIN_ADVANCE_DISTANCE = 15.0  # meters

    if corner_x < midfield_x:
        # Defending team attacks positive X
        crosses_midfield = (start_x < midfield_x) and (end_x >= midfield_x)
        advances_significantly = (end_x - start_x) >= MIN_ADVANCE_DISTANCE

        # DEBUG
        print(f"[DEBUG y3] Defending attack dir: +X. crosses_midfield={crosses_midfield}, advances_significantly={advances_significantly}")

        if crosses_midfield or advances_significantly:
            print("[DEBUG y3] COUNTER DETECTED (returns 1)")
            return 1
    else:
        # Defending team attacks negative X
        crosses_midfield = (start_x >= midfield_x) and (end_x < midfield_x)
        advances_significantly = (start_x - end_x) >= MIN_ADVANCE_DISTANCE

        # DEBUG
        print(f"[DEBUG y3] Defending attack dir: -X. crosses_midfield={crosses_midfield}, advances_significantly={advances_significantly}")

        if crosses_midfield or advances_significantly:
            print("[DEBUG y3] COUNTER DETECTED (returns 1)")
            return 1

    # DEBUG
    print("[DEBUG y3] NO COUNTER (returns 0)")
    return 0


def compute_y4_counter_xthreat(
    corner: dict,
    events_df: pl.DataFrame,
    xt_surface: np.ndarray,  # not used directly here but kept for API consistency
) -> float:
    """
    Compute y4: counter danger from opponent's max xThreat in a counter window (10–25s).

    Implementation detail:
    - Uses event-based xThreat (events_df['xthreat']) for the opponent.
    - Looks at opponent events in frames [frame_start + 10s, frame_start + 25s].

    Returns:
        Max opponent xThreat in that window. 0.0 if no data or no opponent xThreat.
    """
    frame_start = corner["frame_start"]
    period = corner["period"]
    team_id = corner["team_id"]

    fps = 25
    frame_counter_start = frame_start + int(10 * fps)
    frame_counter_end = frame_start + int(25 * fps)

    opp_events = events_df.filter(
        (pl.col("period") == period)
        & (pl.col("frame_start") >= frame_counter_start)
        & (pl.col("frame_start") <= frame_counter_end)
        & (pl.col("team_id") != team_id)
    )

    if len(opp_events) == 0:
        return 0.0

    if "xthreat" not in opp_events.columns:
        return 0.0

    xthreat_vals = opp_events.select(pl.col("xthreat").drop_nulls())
    if xthreat_vals.height == 0:
        return 0.0

    max_xthreat = float(xthreat_vals.max().item())
    return max_xthreat


def compute_improved_labels(
    corner: dict,
    events_df: pl.DataFrame,
    tracking_df: pl.DataFrame,
    xt_surface: np.ndarray,
    xthreat_model: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Compute all improved labels for a single corner.

    Labels:
        - y1: Binary shot indicator within 10 seconds of the corner.
        - y2: Historical corner xThreat by delivery zone.
        - y3: Rule-based counter-attack indicator (0/1).
        - y4: Opponent max xThreat in 10–25s window.
        - y5: Territory change based on ball x_m movement in SkillCorner coords.

    Assumptions:
        - events_df: has 'time_start', 'end_type', 'frame_start', 'team_id', 'period'.
        - tracking_df: has 'frame', 'period', 'is_ball', 'x_m' in SkillCorner coords [-52.5, 52.5].
        - corner dict: includes 'frame_start', 'period', 'team_id', 'timestamp', 'x_start' (SkillCorner x of corner).
    """
    corner_timestamp = float(corner.get("timestamp", 0.0))
    period = corner["period"]
    team_id = corner["team_id"]

    events_df, time_col = _ensure_time_seconds(events_df, "time_start")

    # y1: shot within 10s (events-based)
    shot_events = events_df.filter(
        (pl.col("end_type") == "shot")
        & (pl.col(time_col) >= corner_timestamp)
        & (pl.col(time_col) <= corner_timestamp + 20.0)
        & (pl.col("team_id") == team_id)
    )
    y1 = 1.0 if len(shot_events) > 0 else 0.0

    # y2: corner-specific xThreat
    y2 = compute_y2_corner_xthreat(corner, events_df, xthreat_model)

    # y3: counter-attack detection (0/1)
    y3 = float(detect_counter_attack(corner, tracking_df, events_df, team_id))

    # y4: opponent counter danger (max xThreat in 10–25s window)
    y4 = compute_y4_counter_xthreat(corner, events_df, xt_surface)

    # y5: territory change (SkillCorner x_m movement)
    frame_start = corner["frame_start"]
    fps = 25
    frame_end = frame_start + int(15 * fps)  # 15s horizon

    # Ball around corner instant (±5 frames)
    if "is_ball" in tracking_df.columns:
        ball_start = tracking_df.filter(
            (pl.col("frame") >= frame_start - 5)
            & (pl.col("frame") <= frame_start + 5)
            & (pl.col("period") == period)
            & (pl.col("is_ball") == True)
        )
        ball_end = tracking_df.filter(
            (pl.col("frame") >= frame_end - 5)
            & (pl.col("frame") <= frame_end + 5)
            & (pl.col("period") == period)
            & (pl.col("is_ball") == True)
        )
    else:
        ball_start = pl.DataFrame([])
        ball_end = pl.DataFrame([])

    if len(ball_start) > 0 and len(ball_end) > 0 and "x_m" in ball_start.columns and "x_m" in ball_end.columns:
        # Use mean x_m over small windows for robustness
        start_x = float(ball_start.select(pl.col("x_m").mean()).item())
        end_x = float(ball_end.select(pl.col("x_m").mean()).item())

        # Attack direction from corner position (SkillCorner x_start)
        # If x_start > 0: attacking right (+x). If x_start <= 0: attacking left (-x).
        corner_x = float(corner.get("x_start", 0.0))
        attacking_right = corner_x > 0.0

        if attacking_right:
            # Positive territory_change = ball moved further right (toward +52.5)
            territory_change = end_x - start_x
        else:
            # Positive territory_change = ball moved further left (toward -52.5)
            territory_change = start_x - end_x

        # Normalise by full pitch length (105m) to get roughly [-1, 1]
        y5 = float(np.clip(territory_change / 105.0, -1.0, 1.0))
    else:
        y5 = 0.0

    return {
        "y1": float(y1),
        "y2": float(y2),
        "y3": float(y3),
        "y4": float(y4),
        "y5": float(y5),
    }
