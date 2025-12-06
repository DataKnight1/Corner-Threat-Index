"""
Author: Tiago
Date: 2025-12-04
Description: Counter-Attack Risk Quantification from Tracking Data. Computes y3 (P(counter)) and y4 (counter xG) using tracking data to avoid the y4=0 problem caused by missing event data.
"""

from pathlib import Path
import polars as pl
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class CounterRiskMetrics:
    """Container for counter-attack risk metrics"""
    has_counter: bool  # y3: Did a counter occur?
    counter_xg: float  # y4: Expected goals from counter

    # Detailed metrics for analysis
    space_control: float  # How much space opponent controls
    defensive_compactness: float  # How spread out defenders are
    transition_speed: float  # Speed of counter-attack
    ball_recovery_zone: str  # Where ball was recovered
    attackers_ahead: int  # Number of attackers vs defenders


def compute_space_control(
    ball_pos: np.ndarray,
    attacking_players: np.ndarray,
    defending_players: np.ndarray,
    pitch_control_sigma: float = 2.0
) -> float:
    """
    Compute how much space the counter-attacking team controls.

    Uses simplified pitch control: sum of Gaussian influence functions.
    Higher value = more dangerous counter.

    Args:
        ball_pos: (x, y) ball position in meters (SkillCorner coords)
        attacking_players: (N, 2) positions of counter-attackers
        defending_players: (M, 2) positions of defenders
        pitch_control_sigma: Gaussian spread parameter

    Returns:
        Space control score (0-1, higher = more dangerous)
    """
    if len(attacking_players) == 0:
        return 0.0

    # Compute distance from ball to all players
    ball_to_attackers = np.linalg.norm(attacking_players - ball_pos, axis=1)
    ball_to_defenders = np.linalg.norm(defending_players - ball_pos, axis=1) if len(defending_players) > 0 else np.array([100.0])

    # Gaussian influence (closer = more influence)
    attacker_influence = np.exp(-ball_to_attackers**2 / (2 * pitch_control_sigma**2)).sum()
    defender_influence = np.exp(-ball_to_defenders**2 / (2 * pitch_control_sigma**2)).sum()

    # Space control ratio
    total_influence = attacker_influence + defender_influence
    if total_influence == 0:
        return 0.0

    space_control = attacker_influence / total_influence
    return float(space_control)


def compute_defensive_compactness(
    defending_players: np.ndarray,
    goal_line_x: float = 52.5  # Right goal in SC coords
) -> float:
    """
    Measure how compact/spread out the defense is.

    A spread-out defense (low compactness) is more vulnerable to counters.

    Args:
        defending_players: (M, 2) defender positions
        goal_line_x: X-coordinate of goal being defended

    Returns:
        Compactness score (0-1, lower = more spread = more dangerous counter)
    """
    if len(defending_players) < 2:
        return 0.0  # No defense = very vulnerable

    # Filter defenders in defensive half
    defensive_half = defending_players[defending_players[:, 0] > 0]
    if len(defensive_half) == 0:
        return 0.0

    # Measure spread: standard deviation of positions
    x_std = np.std(defensive_half[:, 0])
    y_std = np.std(defensive_half[:, 1])

    # Higher std = more spread = lower compactness
    # Normalize by typical pitch dimensions
    compactness = 1.0 - min(1.0, (x_std + y_std) / 30.0)

    return float(compactness)


def compute_transition_speed(
    ball_positions: np.ndarray,
    timestamps: np.ndarray,
    time_window: float = 3.0
) -> float:
    """
    Compute speed of transition (how fast ball moves up the field).

    Args:
        ball_positions: (T, 2) ball positions over time
        timestamps: (T,) frame numbers or timestamps
        time_window: seconds to measure transition speed

    Returns:
        Transition speed in m/s (higher = faster counter)
    """
    if len(ball_positions) < 2:
        return 0.0

    # Compute distance traveled
    distances = np.linalg.norm(np.diff(ball_positions, axis=0), axis=1)
    total_distance = distances.sum()

    # Compute time elapsed
    time_diff = timestamps[-1] - timestamps[0]
    if time_diff == 0:
        return 0.0

    # Speed in m/s (assuming timestamps in frames at 25fps)
    speed = total_distance / (time_diff / 25.0)  # frames to seconds

    return float(speed)


def count_attackers_vs_defenders(
    ball_pos: np.ndarray,
    attacking_players: np.ndarray,
    defending_players: np.ndarray,
    goal_line_x: float = 52.5,
    ahead_threshold: float = 5.0
) -> Tuple[int, int]:
    """
    Count attackers vs defenders ahead of the ball.

    A numerical advantage (more attackers than defenders) = dangerous counter.

    Args:
        ball_pos: (x, y) ball position
        attacking_players: (N, 2) attacker positions
        defending_players: (M, 2) defender positions
        goal_line_x: X-coordinate of goal being attacked
        ahead_threshold: meters ahead to be considered "dangerous"

    Returns:
        (num_attackers_ahead, num_defenders_ahead)
    """
    ball_x = ball_pos[0]

    # Count attackers ahead of ball (moving toward goal)
    attackers_ahead = np.sum(attacking_players[:, 0] > ball_x + ahead_threshold) if len(attacking_players) > 0 else 0

    # Count defenders between ball and goal
    defenders_ahead = np.sum(
        (defending_players[:, 0] > ball_x) & (defending_players[:, 0] < goal_line_x)
    ) if len(defending_players) > 0 else 0

    return int(attackers_ahead), int(defenders_ahead)


def compute_counter_xg_from_metrics(
    space_control: float,
    compactness: float,
    attackers_ahead: int,
    defenders_ahead: int,
    ball_x: float,
    ball_y: float,
    goal_x: float = 52.5,
    goal_y: float = 0.0
) -> float:
    """
    Compute expected goals for a counter-attack situation.

    Combines multiple risk factors into an xG estimate.

    Args:
        space_control: Space control score (0-1)
        compactness: Defensive compactness (0-1)
        attackers_ahead: Number of attackers ahead of ball
        defenders_ahead: Number of defenders between ball and goal
        ball_x, ball_y: Ball position
        goal_x, goal_y: Goal position

    Returns:
        Counter-attack xG (0-1)
    """
    # Base xG from distance and angle to goal
    distance = np.sqrt((goal_x - ball_x)**2 + (goal_y - ball_y)**2)
    angle = np.arctan2(abs(goal_y - ball_y), goal_x - ball_x)

    # Distance factor (closer = higher xG)
    distance_factor = max(0.0, 1.0 - distance / 60.0)  # Normalize by pitch length

    # Angle factor (central = higher xG)
    angle_factor = np.cos(angle)  # 1.0 when straight ahead, 0.0 at 90 degrees

    # Numerical advantage factor
    if defenders_ahead == 0:
        numerical_factor = 1.0  # Open goal
    else:
        numerical_advantage = max(0, attackers_ahead - defenders_ahead)
        numerical_factor = min(1.0, 0.5 + 0.1 * numerical_advantage)

    # Vulnerability factor (low compactness + high space control = vulnerable)
    vulnerability = (1.0 - compactness) * space_control

    # Combine factors (multiplicative model)
    base_xg = distance_factor * angle_factor
    counter_multiplier = 1.0 + vulnerability + 0.5 * numerical_factor

    counter_xg = base_xg * counter_multiplier

    # Clip to reasonable range
    return float(np.clip(counter_xg, 0.0, 0.8))  # Counter xG rarely exceeds 0.8


def detect_counter_attack_window(
    tracking_df: pl.DataFrame,
    corner_frame: int,
    corner_period: int,
    attacking_team_id: int,
    window_start: int = 250,  # 10s after corner (at 25fps)
    window_end: int = 625,    # 25s after corner
    possession_threshold: float = 0.6,
    min_possession_duration: float = 10.0,  # NEW: Minimum 10 seconds of possession
    min_progressive_distance: float = 15.0,  # NEW: Minimum field progression (meters)
    min_passes: int = 3  # NEW: Minimum number of passes to qualify
) -> Tuple[bool, int, int]:
    """
    Detect if a counter-attack occurred in the time window.

    UPDATED Definition (per user requirements):
    A counter-attack is defined as:
    1. Defending team gains possession after the corner
    2. They maintain possession for MORE than 10 seconds
    3. They make progressive passes (advancing up the field)
    4. The more distance/territory they gain, the more dangerous
    5. Fast progression indicates a real counter-attack threat

    Args:
        tracking_df: Tracking data for the match
        corner_frame: Frame number when corner was taken
        corner_period: Period of the corner
        attacking_team_id: Team that took the corner
        window_start: Frames after corner to start checking (default 10s)
        window_end: Frames after corner to stop checking (default 25s)
        possession_threshold: Minimum distance to consider "in possession"
        min_possession_duration: Minimum sustained possession in seconds (default 10s)
        min_progressive_distance: Minimum field progression in meters (default 15m)
        min_passes: Minimum number of passes to qualify as counter (default 3)

    Returns:
        (has_counter, counter_start_frame, counter_end_frame)
    """
    # Extract tracking window
    start_frame = corner_frame + window_start
    end_frame = corner_frame + window_end

    window_df = tracking_df.filter(
        (pl.col("period") == corner_period) &
        (pl.col("frame") >= start_frame) &
        (pl.col("frame") <= end_frame)
    )

    if window_df.height == 0:
        return False, 0, 0

    # Get ball positions over time
    ball_df = window_df.filter(pl.col("is_ball") == True)
    if ball_df.height == 0:
        return False, 0, 0

    ball_positions = ball_df.select(["x_m", "y_m"]).to_numpy()
    frames = ball_df["frame"].to_numpy()

    if len(ball_positions) < 250:  # Need at least 10s of data (250 frames at 25fps)
        return False, 0, 0

    # 1. Check possession duration
    # Sustained possession means ball doesn't change possession rapidly
    # We approximate this by checking if ball trajectory is smooth (no sudden reversals)
    x_positions = ball_positions[:, 0]

    # Calculate direction changes (sign changes in velocity)
    x_velocity = np.diff(x_positions)
    direction_changes = np.sum(np.diff(np.sign(x_velocity)) != 0)

    # Too many direction changes = not sustained possession
    if direction_changes > len(x_positions) / 50:  # Allow ~2% direction changes
        return False, 0, 0

    # 2. Check field progression (territorial gain)
    # Counter-attack moves toward attacking team's goal (opposite from corner direction)
    x_start = ball_positions[0, 0]
    x_end = ball_positions[-1, 0]

    # If corner was from right side (x > 0), counter moves left (x decreases)
    # If corner was from left side (x < 0), counter moves right (x increases)
    counter_direction = -np.sign(x_start)
    progressive_distance = (x_end - x_start) * counter_direction

    # Must gain at least min_progressive_distance meters
    if progressive_distance < min_progressive_distance:
        return False, 0, 0

    # 3. Check for progressive passes
    # Estimate passes by finding segments where ball moves quickly (> 5 m/s)
    # then slows down (controlled), repeated multiple times
    segment_size = 25  # 1 second at 25fps
    n_segments = len(ball_positions) // segment_size

    pass_like_segments = 0
    for i in range(n_segments - 1):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size

        segment_distance = np.linalg.norm(
            ball_positions[end_idx] - ball_positions[start_idx]
        )

        # Pass-like if ball moves 5+ meters in 1 second (5 m/s)
        if segment_distance > 5.0:
            pass_like_segments += 1

    # Need at least min_passes pass-like segments
    if pass_like_segments < min_passes:
        return False, 0, 0

    # 4. Check progression speed (danger factor)
    # Faster progression = more dangerous counter
    time_elapsed = (frames[-1] - frames[0]) / 25.0  # Convert frames to seconds
    if time_elapsed < min_possession_duration:
        return False, 0, 0

    progression_speed = progressive_distance / time_elapsed  # meters per second

    # Must be progressing at least 1 m/s (slow build-up) to be considered counter
    if progression_speed < 1.0:
        return False, 0, 0

    # All criteria met: this is a proper counter-attack
    has_counter = True

    return has_counter, int(frames[0]), int(frames[-1])


def compute_counter_risk_from_tracking(
    corner: dict,
    tracking_df: pl.DataFrame,
    events_df: pl.DataFrame = None,
    player_team_map: Dict[int, int] = None
) -> CounterRiskMetrics:
    """
    Main function: Compute counter-attack risk metrics from tracking data.

    This computes y3 and y4 using only tracking data, avoiding reliance on
    incomplete event data.

    Args:
        corner: Corner metadata dict with keys:
                - corner_id, match_id, frame_start, period, team_id
        tracking_df: Full tracking data for the match
        events_df: Optional events data for validation
        player_team_map: Dict mapping player_id -> team_id

    Returns:
        CounterRiskMetrics with y3, y4, and detailed metrics
    """
    frame_start = corner["frame_start"]
    period = corner["period"]
    attacking_team_id = corner["team_id"]

    # Detect counter-attack window
    has_counter, counter_start, counter_end = detect_counter_attack_window(
        tracking_df, frame_start, period, attacking_team_id
    )

    if not has_counter:
        # No counter detected
        return CounterRiskMetrics(
            has_counter=False,
            counter_xg=0.0,
            space_control=0.0,
            defensive_compactness=1.0,
            transition_speed=0.0,
            ball_recovery_zone="none",
            attackers_ahead=0
        )

    # Extract frame at peak counter-attack (middle of counter window)
    peak_frame = (counter_start + counter_end) // 2
    peak_df = tracking_df.filter(
        (pl.col("period") == period) &
        (pl.col("frame") == peak_frame)
    )

    if peak_df.height == 0:
        return CounterRiskMetrics(
            has_counter=True,
            counter_xg=0.01,  # Minimal xG if detected but no tracking
            space_control=0.0,
            defensive_compactness=0.5,
            transition_speed=0.0,
            ball_recovery_zone="unknown",
            attackers_ahead=0
        )

    # Get ball position
    ball_row = peak_df.filter(pl.col("is_ball") == True)
    if ball_row.height == 0:
        return CounterRiskMetrics(has_counter=True, counter_xg=0.01, space_control=0.0,
                                  defensive_compactness=0.5, transition_speed=0.0,
                                  ball_recovery_zone="unknown", attackers_ahead=0)

    ball_pos = ball_row.select(["x_m", "y_m"]).to_numpy()[0]

    # Get player positions (split by team)
    players_df = peak_df.filter(pl.col("is_ball") == False)

    # Map players to teams
    if player_team_map is None:
        # Fallback: assume team_id column exists
        if "team_id" in players_df.columns:
            defending_team_id = [tid for tid in players_df["team_id"].unique() if tid != attacking_team_id][0] if len(players_df["team_id"].unique()) > 1 else None
        else:
            # Can't determine teams
            return CounterRiskMetrics(has_counter=True, counter_xg=0.05, space_control=0.5,
                                      defensive_compactness=0.5, transition_speed=0.0,
                                      ball_recovery_zone="unknown", attackers_ahead=0)
    else:
        # Use player_team_map to split teams
        players_with_teams = []
        for row in players_df.iter_rows(named=True):
            pid = row.get("player_id")
            if pid and pid in player_team_map:
                tid = player_team_map[pid]
                players_with_teams.append((row["x_m"], row["y_m"], tid))

        if not players_with_teams:
            return CounterRiskMetrics(has_counter=True, counter_xg=0.05, space_control=0.5,
                                      defensive_compactness=0.5, transition_speed=0.0,
                                      ball_recovery_zone="unknown", attackers_ahead=0)

        # Split into counter-attackers (defending team) and defenders (attacking team)
        counter_attackers = np.array([(x, y) for x, y, tid in players_with_teams if tid != attacking_team_id])
        defenders = np.array([(x, y) for x, y, tid in players_with_teams if tid == attacking_team_id])

    if len(counter_attackers) == 0:
        counter_attackers = np.array([[ball_pos[0], ball_pos[1]]])  # Fallback
    if len(defenders) == 0:
        defenders = np.array([[ball_pos[0] - 10, ball_pos[1]]])  # Fallback

    # Compute metrics
    space_control = compute_space_control(ball_pos, counter_attackers, defenders)
    compactness = compute_defensive_compactness(defenders)

    # Get ball trajectory for transition speed
    ball_trajectory_df = tracking_df.filter(
        (pl.col("period") == period) &
        (pl.col("frame") >= counter_start) &
        (pl.col("frame") <= counter_end) &
        (pl.col("is_ball") == True)
    )

    if ball_trajectory_df.height > 1:
        ball_trajectory = ball_trajectory_df.select(["x_m", "y_m"]).to_numpy()
        frames = ball_trajectory_df["frame"].to_numpy()
        transition_speed = compute_transition_speed(ball_trajectory, frames)
    else:
        transition_speed = 0.0

    # Count attackers vs defenders
    attackers_ahead, defenders_ahead = count_attackers_vs_defenders(
        ball_pos, counter_attackers, defenders
    )

    # Compute counter xG
    counter_xg = compute_counter_xg_from_metrics(
        space_control, compactness, attackers_ahead, defenders_ahead,
        ball_pos[0], ball_pos[1]
    )

    # Determine ball recovery zone
    ball_recovery_zone = "defensive_third" if ball_pos[0] < -17.5 else "middle_third" if ball_pos[0] < 17.5 else "attacking_third"

    return CounterRiskMetrics(
        has_counter=True,
        counter_xg=counter_xg,
        space_control=space_control,
        defensive_compactness=compactness,
        transition_speed=transition_speed,
        ball_recovery_zone=ball_recovery_zone,
        attackers_ahead=attackers_ahead
    )


# ============================================================================
# Integration with CTI Pipeline
# ============================================================================

def add_tracking_based_counter_labels(
    corners_df: pl.DataFrame,
    tracking_dict: Dict[int, pl.DataFrame],
    events_dict: Dict[int, pl.DataFrame],
    player_team_maps: Dict[int, Dict[int, int]]
) -> pl.DataFrame:
    """
    Add y3_tracking and y4_tracking columns to corners dataset.

    These are computed from tracking data and can replace or supplement
    the event-based y3 and y4 labels.

    Args:
        corners_df: Corners dataset
        tracking_dict: Dict of match_id -> tracking DataFrame
        events_dict: Dict of match_id -> events DataFrame
        player_team_maps: Dict of match_id -> (player_id -> team_id)

    Returns:
        Updated corners_df with y3_tracking and y4_tracking columns
    """
    y3_tracking = []
    y4_tracking = []

    for corner in corners_df.iter_rows(named=True):
        match_id = corner["match_id"]

        if match_id not in tracking_dict:
            y3_tracking.append(0.0)
            y4_tracking.append(0.0)
            continue

        tracking_df = tracking_dict[match_id]
        events_df = events_dict.get(match_id, None)
        player_team_map = player_team_maps.get(match_id, None)

        try:
            metrics = compute_counter_risk_from_tracking(
                corner, tracking_df, events_df, player_team_map
            )
            y3_tracking.append(float(metrics.has_counter))
            y4_tracking.append(float(metrics.counter_xg))
        except Exception as e:
            print(f"Warning: Failed to compute counter risk for corner {corner.get('corner_id')}: {e}")
            y3_tracking.append(0.0)
            y4_tracking.append(0.0)

    return corners_df.with_columns([
        pl.Series("y3_tracking", y3_tracking),
        pl.Series("y4_tracking", y4_tracking)
    ])


if __name__ == "__main__":
    print("CTI Counter-Attack Risk Module (Tracking-Based)")
    print("=" * 70)
    print("\nThis module computes y3 and y4 from tracking data to avoid y4=0 issue.")
    print("\nKey metrics:")
    print("  - Space control: How much space counter-attackers control")
    print("  - Defensive compactness: How spread out defenders are")
    print("  - Numerical advantage: Attackers vs defenders ahead of ball")
    print("  - Transition speed: How fast the counter develops")
    print("\nUsage:")
    print("  from cti_counter_risk_tracking import add_tracking_based_counter_labels")
    print("  corners_df = add_tracking_based_counter_labels(corners, tracking, events, team_maps)")
