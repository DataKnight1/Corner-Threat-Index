"""
Author: Tiago
Date: 2025-12-04
Description: Generate side-by-side pitch comparisons showing tactical patterns and model predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from pathlib import Path
import io
import base64
from PIL import Image


def generate_team_comparison_visual(
    team_name: str,
    nmf_model,
    zone_models,
    team_features: np.ndarray,
    team_metrics: dict,
    xt_surface=None,
    team_corner_positions=None
):
    """
    Generate side-by-side comparison:
    Left: NMF tactical pattern
    Right: Model prediction visualization

    Args:
        team_name: Team name
        nmf_model: NMF model
        zone_models: GMM zone models
        team_features: Team's NMF feature vector
        team_metrics: Dict with y1_avg, y3_avg, y5_avg, cti_avg
        xt_surface: Optional xT surface
        team_corner_positions: Dict with avg_x, avg_y, attacking_side for team's corners

    Returns:
        Base64 encoded image
    """

    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')

    # LEFT PITCH: NMF Tactical Pattern
    init_zones = plot_nmf_pattern(ax1, nmf_model, zone_models, team_features, xt_surface)
    ax1.set_title(f'{team_name} - Tactical Pattern (NMF)',
                 fontsize=14, fontweight='bold', pad=15)

    # RIGHT PITCH: Model Predictions Visualization
    plot_prediction_overlay(ax2, team_metrics, xt_surface, team_corner_positions, init_zones)
    ax2.set_title(f'{team_name} - Model Predictions',
                 fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout(pad=2.0)

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor='white', bbox_inches='tight')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    buf.close()

    return f"data:image/png;base64,{img_data}"


def plot_nmf_pattern(ax, nmf_model, zone_models, team_features, xt_surface=None):
    """Plot NMF tactical pattern on ax and return initial zone positions"""

    # Initialize pitch
    pitch = Pitch(
        pitch_type='custom', pitch_length=105, pitch_width=68, half=True,
        pitch_color='#167d3b', line_color='white', linewidth=1.5
    )
    pitch.draw(ax=ax)
    ax.set_facecolor('#313332')

    # Add xT surface
    if xt_surface is not None:
        try:
            n_x, n_y = xt_surface.shape
            x_bins = np.linspace(52.5, 105, n_x + 1)
            y_bins = np.linspace(0, 68, n_y + 1)
            ax.pcolormesh(x_bins, y_bins, xt_surface.T,
                         cmap='Reds', alpha=0.2, zorder=1,
                         vmin=0, vmax=xt_surface.max())
        except:
            pass

    # Get top feature
    top_feature_idx = np.argmax(team_features)
    H = nmf_model.H
    topic = H[top_feature_idx].reshape(6, 7)

    # Zone coordinates
    init_means = zone_models.gmm_init.means_
    tgt_means_all = zone_models.gmm_tgt.means_
    tgt_ids = zone_models.active_tgt_ids
    tgt_means = tgt_means_all[tgt_ids]

    def sc_to_std(arr):
        return np.column_stack((arr[:, 0] + 52.5, arr[:, 1] + 34.0))

    init_std = sc_to_std(init_means)
    tgt_std = sc_to_std(tgt_means)

    # Get top runs
    runs = []
    for i in range(6):
        for j in range(7):
            w = float(topic[i, j])
            if w >= 0.03:
                runs.append((w, i, j))

    if runs:
        runs.sort(reverse=True)
        runs = runs[:8]
        wmax = runs[0][0] if runs[0][0] > 0 else 1.0

        # Draw arrows
        for w, i, j in runs:
            rel = max(0.0, min(1.0, w / wmax))
            lw = 2.0 + 4.5 * rel
            x0, y0 = init_std[i]
            x1, y1 = tgt_std[j]
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                       arrowprops=dict(arrowstyle='-|>', lw=lw,
                                     color='#ff6b6b', alpha=0.75,
                                     shrinkA=0, shrinkB=0,
                                     connectionstyle='arc3,rad=-0.25'))

    # Draw initial zone dots
    ax.scatter(init_std[:, 0], init_std[:, 1], s=120, c='#4a90e2',
              zorder=15, edgecolors='white', linewidths=2.5, alpha=1.0)

    return init_std


def plot_prediction_overlay(ax, team_metrics, xt_surface=None, team_corner_positions=None, init_zones=None):
    """Plot model predictions as visual overlays on pitch with dynamic positioning"""

    # Initialize pitch
    pitch = Pitch(
        pitch_type='custom', pitch_length=105, pitch_width=68, half=True,
        pitch_color='white', line_color='#333', linewidth=1.5
    )
    pitch.draw(ax=ax)
    ax.set_facecolor('white')

    # Add xT surface
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

    # Extract metrics
    y1_avg = team_metrics.get('y1_avg', 0.0)
    y3_avg = team_metrics.get('y3_avg', 0.0)
    y5_avg = team_metrics.get('y5_avg', 0.0)
    cti_avg = team_metrics.get('cti_avg', 0.0)

    # Calculate dynamic positions based on ball location and xT gradients
    if team_corner_positions is not None:
        avg_x = team_corner_positions.get('avg_x', 52.5)
        avg_y = team_corner_positions.get('avg_y', 34.0)
        # Convert to standard coordinates
        ball_x = avg_x + 52.5
        ball_y = avg_y + 34.0
    else:
        # Fallback to corner flag position
        ball_x = 52.5
        ball_y = 34.0

    # Calculate shot zone position using xT gradient (highest threat area)
    if xt_surface is not None:
        # Find the highest xT area (likely near goal)
        max_xt_idx = np.unravel_index(np.argmax(xt_surface), xt_surface.shape)
        shot_zone_x = 52.5 + (max_xt_idx[0] / 40.0) * 52.5
        shot_zone_y = (max_xt_idx[1] / 40.0) * 68.0
    else:
        # Fallback: penalty spot area
        shot_zone_x = 94.0
        shot_zone_y = 34.0

    # Visualize shot probability (y1) - dynamic circle based on xT surface
    shot_radius = 5 + 15 * y1_avg  # 5-20 meter radius
    shot_circle = plt.Circle((shot_zone_x, shot_zone_y), shot_radius,
                            color='#4CAF50', alpha=0.3, zorder=5,
                            label=f'Shot Zone (P={y1_avg:.1%})')
    ax.add_patch(shot_circle)

    # Counter-attack risk zone - positioned at ball location (where defenders are exposed)
    counter_width = 10 + 30 * y3_avg  # 10-40 meters width
    counter_height = 8 + 12 * y3_avg  # 8-20 meters height
    counter_rect = plt.Rectangle((ball_x - counter_height/2, ball_y - counter_width/2),
                                 counter_height, counter_width,
                                 color='#e74c3c', alpha=0.25, zorder=4,
                                 label=f'Counter Risk (P={y3_avg:.1%})')
    ax.add_patch(counter_rect)

    # Add player position dots from initial zones (same as left pitch)
    if init_zones is not None:
        ax.scatter(init_zones[:, 0], init_zones[:, 1], s=120, c='#4a90e2',
                  zorder=15, edgecolors='white', linewidths=2.5, alpha=1.0)

    # CTI score display
    cti_color = '#4CAF50' if cti_avg > 0.2 else '#FFA500' if cti_avg > 0.1 else '#e74c3c'
    ax.text(0.95, 0.05, f'CTI = {cti_avg:.4f}',
           transform=ax.transAxes, fontsize=16, fontweight='bold',
           ha='right', va='bottom', color=cti_color,
           bbox=dict(boxstyle='round', facecolor='white',
                    edgecolor=cti_color, linewidth=3, alpha=0.9))

    # Add legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Add explanation text
    explanation = "Dynamic model predictions:\n• Green = Shot danger (xT peak)\n• Red = Counter risk (ball position)\n• Blue = Player positions"
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


if __name__ == "__main__":
    print("CTI Comparison Visuals Generator")
    print("=" * 70)
    print("\nGenerates side-by-side pitch comparisons showing:")
    print("  Left: NMF tactical patterns")
    print("  Right: Model predictions visualized")
