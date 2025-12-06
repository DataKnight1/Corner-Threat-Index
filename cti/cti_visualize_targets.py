"""
Author: Tiago
Date: 2025-12-04
Description: Visualize what each target variable (y1-y5) measures using tracking data. Creates static images and GIFs.
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Tuple, Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cti_paths import DATA_2024, DATA_OUT_DIR
from cti_corner_extraction import load_tracking_full, load_events_basic


def draw_pitch(ax, half=False):
    """
    Draw a soccer pitch on the given axes using SkillCorner coordinate system.

    SkillCorner coordinates:
    - Origin (0, 0) at center of pitch
    - x: -52.5 (left goal) to +52.5 (right goal)
    - y: -34 (bottom) to +34 (top)
    """
    # Pitch dimensions (SkillCorner system)
    if half:
        x_min, x_max = 0, 52.5
    else:
        x_min, x_max = -52.5, 52.5

    y_min, y_max = -34, 34

    # Pitch outline
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            color='white', linewidth=2)

    # Center line
    if not half:
        ax.plot([0, 0], [y_min, y_max], color='white', linewidth=1)

        # Center circle
        center_circle = plt.Circle((0, 0), 9.15,
                                  color='white', fill=False, linewidth=1)
        ax.add_patch(center_circle)

    # Penalty areas (16.5m from goal line, 40.32m wide)
    penalty_y_min = -20.16
    penalty_y_max = 20.16

    if not half:
        # Left penalty area
        ax.plot([-52.5, -36, -36, -52.5],
                [penalty_y_min, penalty_y_min, penalty_y_max, penalty_y_max],
                color='white', linewidth=1)
        # Left 6-yard box (5.5m from goal line, 18.32m wide)
        ax.plot([-52.5, -47, -47, -52.5],
                [-9.16, -9.16, 9.16, 9.16],
                color='white', linewidth=1)

    # Right penalty area
    ax.plot([52.5, 36, 36, 52.5] if not half else [52.5, 36, 36, 52.5],
            [penalty_y_min, penalty_y_min, penalty_y_max, penalty_y_max],
            color='white', linewidth=1)
    # Right 6-yard box
    ax.plot([52.5, 47, 47, 52.5] if not half else [52.5, 47, 47, 52.5],
            [-9.16, -9.16, 9.16, 9.16],
            color='white', linewidth=1)

    ax.set_xlim(x_min - 2, x_max + 2)
    ax.set_ylim(y_min - 2, y_max + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#2d5c2e')


def get_corner_window_data(
    corner: dict,
    tracking_df: pl.DataFrame,
    events_df: pl.DataFrame,
    window_start: float,
    window_end: float
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Get tracking and events data for a specific time window.

    Args:
        corner: Corner metadata dict
        tracking_df: Full tracking data
        events_df: Full events data
        window_start: Start time in seconds (relative to corner)
        window_end: End time in seconds (relative to corner)

    Returns:
        (tracking_window, events_window)
    """
    frame_start = corner['frame_start']
    period = corner['period']
    fps = 25

    frame_window_start = frame_start + int(window_start * fps)
    frame_window_end = frame_start + int(window_end * fps)

    # Get tracking in window
    tracking_window = tracking_df.filter(
        (pl.col('frame') >= frame_window_start) &
        (pl.col('frame') <= frame_window_end) &
        (pl.col('period') == period)
    )

    # Get events in window (convert frames to time for events)
    events_window = events_df.filter(
        (pl.col('frame_start') >= frame_window_start) &
        (pl.col('frame_start') <= frame_window_end) &
        (pl.col('period') == period)
    )

    return tracking_window, events_window


def create_y1_visualization(
    corner: dict,
    tracking_df: pl.DataFrame,
    events_df: pl.DataFrame,
    output_dir: Path,
    corner_idx: int = 0
):
    """
    y1: Shot probability in 0-10s window
    Shows attacking team players and shots taken
    """
    print(f"  Creating y1 (Shot Probability) visualization...")

    tracking_window, events_window = get_corner_window_data(
        corner, tracking_df, events_df, 0, 10
    )

    # Find shots in window
    shot_events = events_window.filter(
        (pl.col('end_type') == 'shot') &
        (pl.col('team_id') == corner['team_id'])
    )

    y1_value = 1.0 if len(shot_events) > 0 else 0.0

    # Static image: show initial and shot moments
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (title, frame_offset) in enumerate([
        ("Corner Delivery (0s)", 0),
        ("Shot Moment (if any)", 5)
    ]):
        ax = axes[idx]
        draw_pitch(ax)

        frame_num = corner['frame_start'] + int(frame_offset * 25)
        frame_data = tracking_window.filter(pl.col('frame') == frame_num)

        if len(frame_data) > 0:
            # Plot players
            for row in frame_data.iter_rows(named=True):
                if row.get('is_ball', False):
                    ax.scatter(row['x_m'], row['y_m'], c='white', s=100,
                              marker='o', edgecolors='black', linewidths=2, zorder=10)
                else:
                    color = 'red' if row.get('team_id') == corner['team_id'] else 'blue'
                    ax.scatter(row['x_m'], row['y_m'], c=color, s=80, alpha=0.7)

        # Mark shot positions
        if len(shot_events) > 0:
            for shot in shot_events.iter_rows(named=True):
                shot_x = shot.get('x_start', 0)
                shot_y = shot.get('y_start', 0)
                ax.scatter(shot_x, shot_y, c='yellow', s=300, marker='*',
                          edgecolors='red', linewidths=2, zorder=15,
                          label='Shot')

        # Add team legend
        ax.scatter([], [], c='red', s=80, alpha=0.7, label='Attacking Team')
        ax.scatter([], [], c='blue', s=80, alpha=0.7, label='Defending Team')
        ax.scatter([], [], c='white', s=100, marker='o', edgecolors='black', linewidths=2, label='Ball')

        ax.set_title(title, fontsize=14, color='white', pad=10)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.8)

    fig.suptitle(f'y1: Shot Probability (0-10s)\nValue: {y1_value:.0f} ({"Shot taken" if y1_value == 1 else "No shot"})',
                 fontsize=16, color='white', y=0.98)
    fig.patch.set_facecolor('#1a1a1a')
    plt.tight_layout()

    output_path = output_dir / f'y1_shot_probability_{corner_idx:03d}.png'
    plt.savefig(output_path, dpi=150, facecolor='#1a1a1a')
    plt.close()

    # Create GIF
    create_tracking_gif(
        tracking_window, corner, output_dir / f'y1_shot_probability_{corner_idx:03d}.gif',
        title=f'y1: Shot Window (0-10s) | Value: {y1_value:.0f}',
        highlight_events=shot_events,
        team_id=corner['team_id']
    )

    return output_path


def create_y2_visualization(
    corner: dict,
    tracking_df: pl.DataFrame,
    events_df: pl.DataFrame,
    output_dir: Path,
    corner_idx: int = 0
):
    """
    y2: Max xG in 0-10s window
    Shows shot quality (xG) for attacking team
    """
    print(f"  Creating y2 (Max xG) visualization...")

    tracking_window, events_window = get_corner_window_data(
        corner, tracking_df, events_df, 0, 10
    )

    # Find max xG
    team_events = events_window.filter(pl.col('team_id') == corner['team_id'])
    if 'xthreat' in team_events.columns:
        xg_values = team_events.select(pl.col('xthreat').drop_nulls())
        y2_value = float(xg_values.max().item()) if xg_values.height > 0 else 0.0
    else:
        y2_value = 0.0

    # Static image
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    draw_pitch(ax)

    # Plot positions at peak xG moment
    if y2_value > 0 and 'xthreat' in team_events.columns:
        max_xg_event = team_events.filter(pl.col('xthreat') == y2_value).row(0, named=True)
        frame_num = max_xg_event.get('frame_start', corner['frame_start'])

        frame_data = tracking_window.filter(pl.col('frame') == frame_num)

        for row in frame_data.iter_rows(named=True):
            if row.get('is_ball', False):
                ax.scatter(row['x_m'], row['y_m'], c='white', s=100,
                          marker='o', edgecolors='black', linewidths=2, zorder=10)
            else:
                color = 'red' if row.get('team_id') == corner['team_id'] else 'blue'
                ax.scatter(row['x_m'], row['y_m'], c=color, s=80, alpha=0.7)

        # Mark max xG position
        event_x = max_xg_event.get('x_start', 0)
        event_y = max_xg_event.get('y_start', 0)
        ax.scatter(event_x, event_y, c='gold', s=400, marker='*',
                  edgecolors='orange', linewidths=3, zorder=15,
                  label=f'Max xG: {y2_value:.3f}')

    # Add team legend
    ax.scatter([], [], c='red', s=80, alpha=0.7, label='Attacking Team')
    ax.scatter([], [], c='blue', s=80, alpha=0.7, label='Defending Team')
    ax.scatter([], [], c='white', s=100, marker='o', edgecolors='black', linewidths=2, label='Ball')

    ax.set_title(f'y2: Maximum Expected Goals (0-10s)\nValue: {y2_value:.3f}',
                fontsize=16, color='white', pad=10)
    ax.legend(loc='upper left', fontsize=12)
    fig.patch.set_facecolor('#1a1a1a')
    plt.tight_layout()

    output_path = output_dir / f'y2_max_xg_{corner_idx:03d}.png'
    plt.savefig(output_path, dpi=150, facecolor='#1a1a1a')
    plt.close()

    # Create GIF
    create_tracking_gif(
        tracking_window, corner, output_dir / f'y2_max_xg_{corner_idx:03d}.gif',
        title=f'y2: Max xG Window (0-10s) | Value: {y2_value:.3f}',
        team_id=corner['team_id']
    )

    return output_path


def create_y3_visualization(
    corner: dict,
    tracking_df: pl.DataFrame,
    events_df: pl.DataFrame,
    output_dir: Path,
    corner_idx: int = 0
):
    """
    y3: Counter-attack probability in 0-7s window
    Shows opponent's counter-attack (ball crossing midfield)
    """
    print(f"  Creating y3 (Counter Probability) visualization...")

    tracking_window, events_window = get_corner_window_data(
        corner, tracking_df, events_df, 0, 7
    )

    # Find opponent shots in counter window
    opp_shots = events_window.filter(
        (pl.col('end_type') == 'shot') &
        (pl.col('team_id') != corner['team_id'])
    )

    y3_value = 1.0 if len(opp_shots) > 0 else 0.0

    # Detect midfield crossing (y3 logic) - SkillCorner coordinates
    fps = 25
    ball_tracking = tracking_window.filter(pl.col('is_ball') == True).sort('frame')

    y3_detected = 0
    midfield_x = 0.0  # SkillCorner: midfield is at x=0
    corner_x = corner.get('x_start', 0)

    if len(ball_tracking) >= 2:
        start_x = ball_tracking.row(0, named=True).get('x_m', 0.0)
        end_x = ball_tracking.row(-1, named=True).get('x_m', 0.0)

        if corner_x < midfield_x:
            # Corner at left → counter goes right (toward positive x)
            if start_x < midfield_x and end_x >= midfield_x:
                y3_detected = 1
        else:
            # Corner at right → counter goes left (toward negative x)
            if start_x >= midfield_x and end_x < midfield_x:
                y3_detected = 1

    y3_value = float(y3_detected)

    # Static image - show ball trajectory with midfield line
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    draw_pitch(ax)

    # Highlight midfield line (x=0 in SkillCorner)
    ax.axvline(x=0, color='yellow', linewidth=3, linestyle='--',
              alpha=0.7, label='Midfield Line (x=0)', zorder=5)

    # Plot ball trajectory
    if len(ball_tracking) > 0:
        ball_x = [row.get('x_m', 0) for row in ball_tracking.iter_rows(named=True)]
        ball_y = [row.get('y_m', 0) for row in ball_tracking.iter_rows(named=True)]
        ax.plot(ball_x, ball_y, 'lime', linewidth=3, alpha=0.8,
               label='Ball Path', zorder=8)

        # Start and end markers
        ax.scatter(ball_x[0], ball_y[0], c='green', s=200, marker='o',
                  edgecolors='white', linewidths=2, zorder=10, label='Start (0s)')
        ax.scatter(ball_x[-1], ball_y[-1], c='red', s=200, marker='X',
                  edgecolors='white', linewidths=2, zorder=10, label='End (7s)')

    # Add team legend (for GIF reference)
    ax.scatter([], [], c='blue', s=80, alpha=0.7, label='Attacking Team (Corner Takers)')
    ax.scatter([], [], c='red', s=80, alpha=0.7, label='Defending Team (Opponent)')

    status_text = "COUNTER DETECTED!" if y3_value == 1 else "No counter (ball didn't cross midfield)"
    ax.set_title(f'y3: Counter-Attack Probability (0-7s)\nValue: {y3_value:.0f} | {status_text}',
                fontsize=16, color='white', pad=10)
    ax.legend(loc='upper left', fontsize=12)
    fig.patch.set_facecolor('#1a1a1a')
    plt.tight_layout()

    output_path = output_dir / f'y3_counter_probability_{corner_idx:03d}.png'
    plt.savefig(output_path, dpi=150, facecolor='#1a1a1a')
    plt.close()

    # Create GIF
    create_tracking_gif(
        tracking_window, corner, output_dir / f'y3_counter_probability_{corner_idx:03d}.gif',
        title=f'y3: Counter Window (0-7s) | Value: {y3_value:.0f}',
        team_id=corner['team_id'],
        flip_colors=True  # Highlight defending team
    )

    return output_path


def create_y4_visualization(
    corner: dict,
    tracking_df: pl.DataFrame,
    events_df: pl.DataFrame,
    output_dir: Path,
    corner_idx: int = 0
):
    """
    y4: Max counter xG in 10-25s window
    Shows quality of opponent's counter-attack chances
    """
    print(f"  Creating y4 (Max Counter xG) visualization...")

    tracking_window, events_window = get_corner_window_data(
        corner, tracking_df, events_df, 10, 25
    )

    # Find max opponent xG
    opp_events = events_window.filter(pl.col('team_id') != corner['team_id'])
    if 'xthreat' in opp_events.columns:
        xg_values = opp_events.select(pl.col('xthreat').drop_nulls())
        y4_value = float(xg_values.max().item()) if xg_values.height > 0 else 0.0
    else:
        y4_value = 0.0

    # Static image
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    draw_pitch(ax)

    # Plot positions at peak counter xG moment
    if y4_value > 0 and 'xthreat' in opp_events.columns:
        max_xg_event = opp_events.filter(pl.col('xthreat') == y4_value).row(0, named=True)
        frame_num = max_xg_event.get('frame_start', corner['frame_start'] + 250)

        frame_data = tracking_window.filter(pl.col('frame') == frame_num)

        for row in frame_data.iter_rows(named=True):
            if row.get('is_ball', False):
                ax.scatter(row['x_m'], row['y_m'], c='white', s=100,
                          marker='o', edgecolors='black', linewidths=2, zorder=10)
            else:
                # Defending team = red (countering)
                color = 'blue' if row.get('team_id') == corner['team_id'] else 'red'
                ax.scatter(row['x_m'], row['y_m'], c=color, s=80, alpha=0.7)

        # Mark max counter xG position
        event_x = max_xg_event.get('x_start', 0)
        event_y = max_xg_event.get('y_start', 0)
        ax.scatter(event_x, event_y, c='gold', s=400, marker='*',
                  edgecolors='orange', linewidths=3, zorder=15,
                  label=f'Max Counter xG: {y4_value:.3f}')

    # Add team legend
    ax.scatter([], [], c='blue', s=80, alpha=0.7, label='Attacking Team (Corner Takers)')
    ax.scatter([], [], c='red', s=80, alpha=0.7, label='Defending Team (Countering)')
    ax.scatter([], [], c='white', s=100, marker='o', edgecolors='black', linewidths=2, label='Ball')

    ax.set_title(f'y4: Maximum Counter Expected Goals (10-25s)\nValue: {y4_value:.3f}',
                fontsize=16, color='white', pad=10)
    ax.legend(loc='upper left', fontsize=12)
    fig.patch.set_facecolor('#1a1a1a')
    plt.tight_layout()

    output_path = output_dir / f'y4_max_counter_xg_{corner_idx:03d}.png'
    plt.savefig(output_path, dpi=150, facecolor='#1a1a1a')
    plt.close()

    # Create GIF
    create_tracking_gif(
        tracking_window, corner, output_dir / f'y4_max_counter_xg_{corner_idx:03d}.gif',
        title=f'y4: Max Counter xG Window (10-25s) | Value: {y4_value:.3f}',
        team_id=corner['team_id'],
        flip_colors=True
    )

    return output_path


def create_y5_visualization(
    corner: dict,
    tracking_df: pl.DataFrame,
    events_df: pl.DataFrame,
    output_dir: Path,
    corner_idx: int = 0
):
    """
    y5: Territory change (Delta xT)
    Shows ball movement and field position gain/loss
    """
    print(f"  Creating y5 (Territory Change) visualization...")

    tracking_window, _ = get_corner_window_data(
        corner, tracking_df, events_df, 0, 15
    )

    frame_start = corner['frame_start']
    period = corner['period']
    fps = 25

    # Get ball positions at start and end
    ball_start = tracking_df.filter(
        (pl.col('frame') >= frame_start - 5) &
        (pl.col('frame') <= frame_start + 5) &
        (pl.col('period') == period) &
        (pl.col('is_ball') == True)
    )

    ball_end = tracking_df.filter(
        (pl.col('frame') >= frame_start + 15*fps - 5) &
        (pl.col('frame') <= frame_start + 15*fps + 5) &
        (pl.col('period') == period) &
        (pl.col('is_ball') == True)
    )

    y5_value = 0.0
    start_x, end_x = 52.5, 52.5

    if len(ball_start) > 0 and len(ball_end) > 0:
        start_x = ball_start.row(0, named=True).get('x_m', 52.5)
        end_x = ball_end.row(0, named=True).get('x_m', 52.5)

        # Determine attacking direction
        attacking_right = start_x > 52.5

        if attacking_right:
            y5_value = (end_x - start_x) / 105.0  # Normalized
        else:
            y5_value = (start_x - end_x) / 105.0

    # Static image showing ball trajectory
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    draw_pitch(ax)

    # Plot ball positions over time
    ball_tracking = tracking_window.filter(pl.col('is_ball') == True).sort('frame')

    if len(ball_tracking) > 0:
        ball_x = ball_tracking.select(pl.col('x_m')).to_numpy().flatten()
        ball_y = ball_tracking.select(pl.col('y_m')).to_numpy().flatten()

        # Plot trajectory
        ax.plot(ball_x, ball_y, 'yellow', linewidth=2, alpha=0.7, label='Ball path')

        # Start position
        ax.scatter(ball_x[0], ball_y[0], c='green', s=200, marker='o',
                  edgecolors='white', linewidths=2, zorder=10, label='Start')

        # End position
        ax.scatter(ball_x[-1], ball_y[-1], c='red', s=200, marker='X',
                  edgecolors='white', linewidths=2, zorder=10, label='End')

        # Arrow showing direction
        if len(ball_x) > 10:
            mid_idx = len(ball_x) // 2
            dx = ball_x[mid_idx+5] - ball_x[mid_idx]
            dy = ball_y[mid_idx+5] - ball_y[mid_idx]
            ax.arrow(ball_x[mid_idx], ball_y[mid_idx], dx, dy,
                    head_width=3, head_length=2, fc='yellow', ec='yellow',
                    alpha=0.7, zorder=9)

    # Add team legend (for GIF reference)
    ax.scatter([], [], c='red', s=80, alpha=0.7, label='Attacking Team')
    ax.scatter([], [], c='blue', s=80, alpha=0.7, label='Defending Team')

    direction = "Forward" if y5_value > 0 else "Backward" if y5_value < 0 else "No change"
    ax.set_title(f'y5: Territory Change (Delta xT)\nValue: {y5_value:.3f} ({direction})',
                fontsize=16, color='white', pad=10)
    ax.legend(loc='upper left', fontsize=12)
    fig.patch.set_facecolor('#1a1a1a')
    plt.tight_layout()

    output_path = output_dir / f'y5_territory_change_{corner_idx:03d}.png'
    plt.savefig(output_path, dpi=150, facecolor='#1a1a1a')
    plt.close()

    # Create GIF showing ball movement
    create_tracking_gif(
        tracking_window, corner, output_dir / f'y5_territory_change_{corner_idx:03d}.gif',
        title=f'y5: Territory Change (0-15s) | Value: {y5_value:.3f}',
        team_id=corner['team_id'],
        show_ball_trail=True
    )

    return output_path


def create_tracking_gif(
    tracking_window: pl.DataFrame,
    corner: dict,
    output_path: Path,
    title: str = "Corner Tracking",
    highlight_events: Optional[pl.DataFrame] = None,
    team_id: Optional[int] = None,
    flip_colors: bool = False,
    show_ball_trail: bool = False,
    fps: int = 5
):
    """
    Create animated GIF from tracking data.

    Args:
        tracking_window: Tracking data for the window
        corner: Corner metadata
        output_path: Path to save GIF
        title: Title for the animation
        highlight_events: Events to highlight (e.g., shots)
        team_id: Team ID for coloring
        flip_colors: If True, highlight defending team instead
        show_ball_trail: Show ball trajectory trail
        fps: Frames per second for GIF
    """
    if len(tracking_window) == 0:
        print(f"    No tracking data for GIF, skipping...")
        return

    # Get unique frames
    frames = sorted(tracking_window.select(pl.col('frame').unique()).to_numpy().flatten())

    # Limit to reasonable number of frames for GIF size
    if len(frames) > 50:
        frames = frames[::max(1, len(frames)//50)]

    fig, ax = plt.subplots(figsize=(12, 8))

    ball_trail_x = []
    ball_trail_y = []

    def update(frame_num):
        ax.clear()
        draw_pitch(ax)

        # Get data for this frame
        frame_data = tracking_window.filter(pl.col('frame') == frame_num)

        if len(frame_data) > 0:
            for row in frame_data.iter_rows(named=True):
                if row.get('is_ball', False):
                    ball_x, ball_y = row['x_m'], row['y_m']
                    ax.scatter(ball_x, ball_y, c='white', s=100,
                              marker='o', edgecolors='black', linewidths=2, zorder=10)

                    if show_ball_trail:
                        ball_trail_x.append(ball_x)
                        ball_trail_y.append(ball_y)
                else:
                    if team_id is not None:
                        if flip_colors:
                            # Highlight defending team (opponent)
                            color = 'blue' if row.get('team_id') == team_id else 'red'
                            label_attacking = 'blue' if row.get('team_id') == team_id else None
                            label_defending = 'red' if row.get('team_id') != team_id else None
                        else:
                            # Highlight attacking team (corner takers)
                            color = 'red' if row.get('team_id') == team_id else 'blue'
                            label_attacking = 'red' if row.get('team_id') == team_id else None
                            label_defending = 'blue' if row.get('team_id') != team_id else None
                    else:
                        color = 'gray'
                        label_attacking = None
                        label_defending = None
                    ax.scatter(row['x_m'], row['y_m'], c=color, s=80, alpha=0.7)

        # Show ball trail
        if show_ball_trail and len(ball_trail_x) > 1:
            ax.plot(ball_trail_x, ball_trail_y, 'yellow', linewidth=2, alpha=0.5)

        # Highlight events
        if highlight_events is not None and len(highlight_events) > 0:
            for event in highlight_events.iter_rows(named=True):
                if event.get('frame_start', 0) <= frame_num:
                    event_x = event.get('x_start', 0)
                    event_y = event.get('y_start', 0)
                    ax.scatter(event_x, event_y, c='yellow', s=300, marker='*',
                              edgecolors='red', linewidths=2, zorder=15, alpha=0.8)

        # Add legend for team colors
        if team_id is not None:
            # Create dummy scatter plots for legend
            if flip_colors:
                ax.scatter([], [], c='red', s=80, alpha=0.7, label='Defending Team (Opponent)')
                ax.scatter([], [], c='blue', s=80, alpha=0.7, label='Attacking Team (Corner Takers)')
            else:
                ax.scatter([], [], c='red', s=80, alpha=0.7, label='Attacking Team (Corner Takers)')
                ax.scatter([], [], c='blue', s=80, alpha=0.7, label='Defending Team (Opponent)')
            ax.scatter([], [], c='white', s=100, marker='o', edgecolors='black', linewidths=2, label='Ball')
            ax.legend(loc='upper right', fontsize=10, framealpha=0.8)

        # Frame time
        time_s = (frame_num - corner['frame_start']) / 25.0
        ax.set_title(f'{title}\nTime: {time_s:.1f}s', fontsize=14, color='white', pad=10)
        fig.patch.set_facecolor('#1a1a1a')

    anim = FuncAnimation(fig, update, frames=frames, interval=1000//fps, repeat=True)

    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"    Saved GIF: {output_path.name}")


def visualize_all_targets_for_corner(
    corner: dict,
    tracking_df: pl.DataFrame,
    events_df: pl.DataFrame,
    output_dir: Path,
    corner_idx: int = 0
):
    """
    Create all visualizations for a single corner.

    Returns:
        dict with paths to all created files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nVisualizing corner {corner_idx} (match {corner['match_id']}):")

    paths = {}

    # Create each visualization
    paths['y1_static'] = create_y1_visualization(corner, tracking_df, events_df, output_dir, corner_idx)
    paths['y2_static'] = create_y2_visualization(corner, tracking_df, events_df, output_dir, corner_idx)
    paths['y3_static'] = create_y3_visualization(corner, tracking_df, events_df, output_dir, corner_idx)
    paths['y4_static'] = create_y4_visualization(corner, tracking_df, events_df, output_dir, corner_idx)
    paths['y5_static'] = create_y5_visualization(corner, tracking_df, events_df, output_dir, corner_idx)

    print(f"  All visualizations created for corner {corner_idx}!")

    return paths


if __name__ == "__main__":
    # Test with a single corner
    from cti_corner_extraction import load_corners_basic

    print("Loading data for visualization test...")

    # Load corners
    corners_df = load_corners_basic(max_matches=1, verbose=True)

    if corners_df.height == 0:
        print("No corners found!")
        sys.exit(1)

    # Get first corner with good data
    corner = corners_df.row(0, named=True)
    match_id = corner['match_id']

    # Load match data
    tracking_df = load_tracking_full(match_id, sort_rows=False)
    events_df = load_events_basic(match_id)

    # Create visualizations
    output_dir = DATA_OUT_DIR / "target_visualizations"

    visualize_all_targets_for_corner(
        corner, tracking_df, events_df, output_dir, corner_idx=0
    )

    print(f"\nDone! Visualizations saved to: {output_dir}")
