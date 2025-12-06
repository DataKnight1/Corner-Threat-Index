"""
Author: Tiago
Date: 2025-12-04
Description: Generate tracking-based visualizations for actual corner kicks. Shows real player positions and ball movement overlaid with model predictions.
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from pathlib import Path
from PIL import Image
import io
import base64
from typing import Dict, Tuple, Optional, List
import matplotlib.animation as animation


def generate_corner_tracking_visualization(
    corner_data: dict,
    tracking_df: pl.DataFrame,
    model_predictions: dict,
    xt_surface: Optional[np.ndarray] = None,
    output_format: str = 'base64',
    figsize: Tuple[int, int] = (10, 7)
) -> str:
    """
    Generate a static visualization showing actual corner tracking data with model predictions.

    Args:
        corner_data: Dict with keys: match_id, frame_start, period, team_id
        tracking_df: Tracking data for the match
        model_predictions: Dict with y1-y5 predictions and CTI score
        xt_surface: Optional xT surface for background
        output_format: 'base64' or 'path'
        figsize: Figure size (width, height)

    Returns:
        Base64 encoded image or file path
    """

    frame_start = corner_data['frame_start']
    period = corner_data['period']
    team_id = corner_data['team_id']

    # Extract 10 seconds of tracking data (250 frames at 25fps)
    window_frames = 250
    tracking_window = tracking_df.filter(
        (pl.col('period') == period) &
        (pl.col('frame') >= frame_start) &
        (pl.col('frame') <= frame_start + window_frames)
    )

    if tracking_window.height == 0:
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    # Draw pitch
    pitch = Pitch(
        pitch_type='custom', pitch_length=105, pitch_width=68, half=True,
        pitch_color='white', line_color='#333', linewidth=1.5
    )
    pitch.draw(ax=ax)
    ax.set_facecolor('white')

    # Add xT surface background
    if xt_surface is not None:
        try:
            n_x, n_y = xt_surface.shape
            x_bins = np.linspace(52.5, 105, n_x + 1)
            y_bins = np.linspace(0, 68, n_y + 1)
            ax.pcolormesh(x_bins, y_bins, xt_surface.T,
                         cmap='Reds', alpha=0.15, zorder=1,
                         vmin=0, vmax=xt_surface.max())
        except:
            pass

    # Get ball trajectory
    ball_data = tracking_window.filter(pl.col('is_ball') == True)
    if ball_data.height > 0:
        ball_x = ball_data['x_m'].to_numpy() + 52.5  # Convert to standard coords
        ball_y = ball_data['y_m'].to_numpy() + 34.0

        # Draw ball trajectory
        ax.plot(ball_x, ball_y, 'o-', color='#FFD700', linewidth=2,
               markersize=4, alpha=0.7, label='Ball trajectory', zorder=10)

        # Highlight start and end
        ax.scatter([ball_x[0]], [ball_y[0]], s=200, c='green',
                  edgecolors='white', linewidths=2, zorder=15,
                  label='Corner kick', marker='*')
        ax.scatter([ball_x[-1]], [ball_y[-1]], s=150, c='red',
                  edgecolors='white', linewidths=2, zorder=15,
                  label='End position', marker='X')

    # Get player positions at key moments
    # Show initial positions (frame_start) and final positions (frame_start + window_frames)
    initial_frame = tracking_window.filter(pl.col('frame') == frame_start)
    final_frame = tracking_window.filter(pl.col('frame') == frame_start + window_frames)

    # Plot players at start (lighter color)
    if initial_frame.height > 0:
        players_init = initial_frame.filter(pl.col('is_ball') == False)
        if players_init.height > 0:
            px_init = players_init['x_m'].to_numpy() + 52.5
            py_init = players_init['y_m'].to_numpy() + 34.0

            # Try to color by team
            if 'team_id' in players_init.columns:
                attacking_mask = players_init['team_id'].to_numpy() == team_id
                ax.scatter(px_init[attacking_mask], py_init[attacking_mask],
                          s=100, c='#4a90e2', alpha=0.4, edgecolors='white',
                          linewidths=1, zorder=5, label='Attacking (initial)')
                ax.scatter(px_init[~attacking_mask], py_init[~attacking_mask],
                          s=100, c='#e74c3c', alpha=0.4, edgecolors='white',
                          linewidths=1, zorder=5, label='Defending (initial)')
            else:
                ax.scatter(px_init, py_init, s=100, c='gray', alpha=0.4,
                          edgecolors='white', linewidths=1, zorder=5)

    # Plot players at end (darker color)
    if final_frame.height > 0:
        players_final = final_frame.filter(pl.col('is_ball') == False)
        if players_final.height > 0:
            px_final = players_final['x_m'].to_numpy() + 52.5
            py_final = players_final['y_m'].to_numpy() + 34.0

            if 'team_id' in players_final.columns:
                attacking_mask = players_final['team_id'].to_numpy() == team_id
                ax.scatter(px_final[attacking_mask], py_final[attacking_mask],
                          s=120, c='#4a90e2', alpha=0.9, edgecolors='white',
                          linewidths=2, zorder=8, label='Attacking (final)')
                ax.scatter(px_final[~attacking_mask], py_final[~attacking_mask],
                          s=120, c='#e74c3c', alpha=0.9, edgecolors='white',
                          linewidths=2, zorder=8, label='Defending (final)')
            else:
                ax.scatter(px_final, py_final, s=120, c='darkgray', alpha=0.9,
                          edgecolors='white', linewidths=2, zorder=8)

    # Add model predictions text box
    pred_text = f"""Model Predictions:
Shot Prob: {model_predictions.get('y1', 0):.1%}
Counter Risk: {model_predictions.get('y3', 0):.1%}
ΔxT: {model_predictions.get('y5', 0):.4f}
CTI: {model_predictions.get('cti', 0):.4f}"""

    ax.text(0.02, 0.98, pred_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontfamily='monospace')

    # Legend
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Title
    ax.set_title('Corner Kick Analysis - Tracking Data with Model Predictions',
                fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()

    # Convert to base64
    if output_format == 'base64':
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='white', bbox_inches='tight')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        buf.close()
        return f"data:image/png;base64,{img_data}"
    else:
        output_path = Path(output_format)
        plt.savefig(output_path, dpi=150, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        return str(output_path)


def generate_corner_tracking_gif(
    corner_data: dict,
    tracking_df: pl.DataFrame,
    model_predictions: dict,
    xt_surface: Optional[np.ndarray] = None,
    duration: int = 10,
    fps: int = 5,
    figsize: Tuple[int, int] = (10, 7)
) -> Optional[str]:
    """
    Generate an animated GIF showing actual corner tracking data with model predictions.

    Args:
        corner_data: Dict with keys: match_id, frame_start, period, team_id
        tracking_df: Tracking data for the match
        model_predictions: Dict with y1-y5 predictions and CTI score
        xt_surface: Optional xT surface for background
        duration: Duration in seconds to visualize
        fps: Frames per second for the GIF (lower = smaller file)
        figsize: Figure size (width, height)

    Returns:
        Base64 encoded GIF image or None if failed
    """

    frame_start = corner_data['frame_start']
    period = corner_data['period']
    team_id = corner_data['team_id']

    # Extract tracking data for duration (at 25fps tracking rate)
    tracking_fps = 25
    total_frames = int(duration * tracking_fps)

    tracking_window = tracking_df.filter(
        (pl.col('period') == period) &
        (pl.col('frame') >= frame_start) &
        (pl.col('frame') <= frame_start + total_frames)
    )

    if tracking_window.height == 0:
        return None

    # Sample frames for GIF (reduce to target fps)
    frame_step = tracking_fps // fps
    unique_frames = sorted(tracking_window['frame'].unique().to_list())
    sampled_frames = unique_frames[::frame_step]

    if len(sampled_frames) == 0:
        return None

    # Prepare data
    ball_data = tracking_window.filter(pl.col('is_ball') == True)
    player_data = tracking_window.filter(pl.col('is_ball') == False)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    # Initialize pitch
    pitch = Pitch(
        pitch_type='custom', pitch_length=105, pitch_width=68, half=True,
        pitch_color='white', line_color='#333', linewidth=1.5
    )
    pitch.draw(ax=ax)
    ax.set_facecolor('white')

    # Add xT surface background (static)
    if xt_surface is not None:
        try:
            n_x, n_y = xt_surface.shape
            x_bins = np.linspace(52.5, 105, n_x + 1)
            y_bins = np.linspace(0, 68, n_y + 1)
            ax.pcolormesh(x_bins, y_bins, xt_surface.T,
                         cmap='Reds', alpha=0.15, zorder=1,
                         vmin=0, vmax=xt_surface.max())
        except:
            pass

    # Add model predictions text box (static)
    pred_text = f"""Model Predictions:
Shot Prob: {model_predictions.get('y1', 0):.1%}
Counter Risk: {model_predictions.get('y3', 0):.1%}
ΔxT: {model_predictions.get('y5', 0):.4f}
CTI: {model_predictions.get('cti', 0):.4f}"""

    ax.text(0.02, 0.98, pred_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontfamily='monospace', zorder=20)

    # Initialize plot elements
    ball_scatter = ax.scatter([], [], s=150, c='#FFD700', edgecolors='white',
                             linewidths=2, zorder=15, marker='o', label='Ball')
    ball_trail, = ax.plot([], [], 'o-', color='#FFD700', linewidth=1,
                         markersize=2, alpha=0.3, zorder=10)

    attacking_scatter = ax.scatter([], [], s=100, c='#4a90e2', alpha=0.9,
                                  edgecolors='white', linewidths=2, zorder=8,
                                  label='Attacking')
    defending_scatter = ax.scatter([], [], s=100, c='#e74c3c', alpha=0.9,
                                  edgecolors='white', linewidths=2, zorder=8,
                                  label='Defending')

    # Time display
    time_text = ax.text(0.98, 0.02, '', transform=ax.transAxes,
                       fontsize=11, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_title('Corner Kick Analysis - Real-time Tracking',
                fontsize=12, fontweight='bold', pad=10)

    # Animation function
    ball_trail_x = []
    ball_trail_y = []

    def animate(i):
        frame_num = sampled_frames[i]
        elapsed_time = (frame_num - frame_start) / tracking_fps

        # Get data for this frame
        frame_data = tracking_window.filter(pl.col('frame') == frame_num)

        if frame_data.height > 0:
            # Ball position
            ball_frame = frame_data.filter(pl.col('is_ball') == True)
            if ball_frame.height > 0:
                ball_x = ball_frame['x_m'].to_numpy()[0] + 52.5
                ball_y = ball_frame['y_m'].to_numpy()[0] + 34.0
                ball_scatter.set_offsets([[ball_x, ball_y]])

                # Update ball trail
                ball_trail_x.append(ball_x)
                ball_trail_y.append(ball_y)
                ball_trail.set_data(ball_trail_x, ball_trail_y)

            # Player positions
            players_frame = frame_data.filter(pl.col('is_ball') == False)
            if players_frame.height > 0:
                px = players_frame['x_m'].to_numpy() + 52.5
                py = players_frame['y_m'].to_numpy() + 34.0

                if 'team_id' in players_frame.columns:
                    attacking_mask = players_frame['team_id'].to_numpy() == team_id
                    attacking_scatter.set_offsets(np.column_stack([px[attacking_mask], py[attacking_mask]]))
                    defending_scatter.set_offsets(np.column_stack([px[~attacking_mask], py[~attacking_mask]]))
                else:
                    attacking_scatter.set_offsets(np.column_stack([px, py]))
                    defending_scatter.set_offsets([])

        # Update time
        time_text.set_text(f't = {elapsed_time:.1f}s')

        return ball_scatter, ball_trail, attacking_scatter, defending_scatter, time_text

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(sampled_frames),
                                  interval=1000//fps, blit=True, repeat=True)

    # Save as GIF
    try:
        buf = io.BytesIO()
        anim.save(buf, format='gif', writer='pillow', fps=fps, dpi=100)
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        buf.close()
        return f"data:image/gif;base64,{img_data}"
    except Exception as e:
        print(f"Warning: Failed to generate GIF: {e}")
        plt.close(fig)
        return None


def generate_team_corner_samples(
    team_id: int,
    team_name: str,
    corners_df: pl.DataFrame,
    tracking_dict: Dict[int, pl.DataFrame],
    predictions_df: pl.DataFrame,
    xt_surface: Optional[np.ndarray] = None,
    max_corners: int = 5,
    use_gif: bool = True
) -> Dict[int, str]:
    """
    Generate tracking visualizations for a sample of corners from a team.

    Args:
        team_id: Team ID
        team_name: Team name
        corners_df: All corners data
        tracking_dict: Dict of match_id -> tracking DataFrame
        predictions_df: Model predictions for all corners
        xt_surface: Optional xT surface
        max_corners: Maximum number of corners to visualize
        use_gif: If True, generate animated GIFs; if False, static images

    Returns:
        Dict mapping corner_id -> base64 image/gif
    """

    # Filter corners for this team
    team_corners = corners_df.filter(pl.col('team_id') == team_id)

    if team_corners.height == 0:
        return {}

    # Sample up to max_corners
    if team_corners.height > max_corners:
        # Select diverse corners (high CTI, low CTI, medium)
        # First check if 'cti_e' column exists
        if 'cti_e' in team_corners.columns:
            team_corners = team_corners.sort('cti_e', descending=True)
        elif 'cti' in team_corners.columns:
            # Join with predictions to get CTI
            team_corners = team_corners.join(
                predictions_df.select(['corner_id', 'cti']),
                on='corner_id',
                how='left'
            ).sort('cti', descending=True)

        indices = [
            0,  # Highest CTI
            team_corners.height // 4,  # Upper quartile
            team_corners.height // 2,  # Median
            3 * team_corners.height // 4,  # Lower quartile
            team_corners.height - 1  # Lowest CTI
        ]
        team_corners = team_corners[indices[:max_corners]]

    corner_visualizations = {}

    for corner_row in team_corners.iter_rows(named=True):
        corner_id = corner_row['corner_id']
        match_id = corner_row['match_id']

        # Check if we have tracking data
        if match_id not in tracking_dict:
            continue

        tracking_df = tracking_dict[match_id]

        # Get model predictions
        pred_row = predictions_df.filter(pl.col('corner_id') == corner_id)
        if pred_row.height == 0:
            continue

        pred_dict = {
            'y1': pred_row['y1'][0],
            'y2': pred_row['y2'][0],
            'y3': pred_row['y3'][0],
            'y4': pred_row['y4'][0],
            'y5': pred_row['y5'][0],
            'cti': pred_row['cti'][0]
        }

        # Generate visualization
        try:
            if use_gif:
                img_base64 = generate_corner_tracking_gif(
                    corner_row, tracking_df, pred_dict, xt_surface,
                    duration=10, fps=5
                )
            else:
                img_base64 = generate_corner_tracking_visualization(
                    corner_row, tracking_df, pred_dict, xt_surface
                )

            if img_base64:
                corner_visualizations[corner_id] = img_base64
                print(f"  [OK] Generated {'GIF' if use_gif else 'image'} for corner {corner_id}")
        except Exception as e:
            print(f"Warning: Failed to generate tracking viz for corner {corner_id}: {e}")
            continue

    return corner_visualizations


if __name__ == "__main__":
    print("CTI Tracking Visualization Module")
    print("=" * 70)
    print("\nThis module generates visualizations of actual corner kicks")
    print("with tracking data overlaid with model predictions.")
    print("\nUsage:")
    print("  from cti_tracking_visualization import generate_corner_tracking_visualization")
    print("  img = generate_corner_tracking_visualization(corner, tracking, predictions, xt)")
