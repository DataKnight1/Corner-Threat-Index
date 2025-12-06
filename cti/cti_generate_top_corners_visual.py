"""
Author: Tiago
Date: 2025-12-04
Description: Generate visual dashboard of top corners for each team with plots and logos.
"""
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'cti_data'
OUT_FIG = ROOT / 'cti_outputs'
LOGO_DIR = ROOT / 'assets' / 'team_logos'

# Load data
print("Loading data...")
predictions = pl.read_csv(DATA_DIR / 'predictions.csv')
corners_df = pl.scan_parquet(DATA_DIR / 'corners_dataset.parquet').collect()
team_cti = pl.read_csv(DATA_DIR / 'team_cti_detailed.csv')

# Load team name mapping
try:
    teams_meta = pl.read_parquet(ROOT.parent / 'data' / 'meta' / 'teams.parquet')
    team_name_map = {row['team_id']: row['name'] for row in teams_meta.iter_rows(named=True)}
except:
    team_name_map = {}

# Predefined team names
predefined_names = {
    2: "Liverpool Football Club", 3: "Arsenal Football Club", 31: "Manchester United",
    32: "Newcastle United", 37: "West Ham United", 39: "Aston Villa", 40: "Manchester City",
    41: "Everton", 44: "Tottenham Hotspur", 48: "Fulham", 49: "Chelsea",
    52: "Wolverhampton Wanderers", 58: "Southampton", 60: "Crystal Palace",
    62: "Leicester City", 63: "Bournemouth", 308: "Brighton and Hove Albion",
    747: "Nottingham Forest", 752: "Ipswich Town", 754: "Brentford FC"
}
team_name_map.update(predefined_names)

def get_team_logo(team_id):
    """Get team logo image or return None"""
    logo_path = LOGO_DIR / f"{team_id}.png"
    if logo_path.exists():
        try:
            return Image.open(logo_path)
        except:
            return None
    return None

def plot_corner_snapshot(ax, corner_data, show_title=True):
    """Plot a corner kick snapshot on given axis"""

    def transform_coords(x_vals, y_vals, attacking_side='left_to_right'):
        """Transform from centered coords [-52.5, 52.5] x [-34, 34] to [0, 105] x [0, 68]

        Args:
            x_vals: X coordinates in centered system
            y_vals: Y coordinates in centered system
            attacking_side: 'left_to_right' or 'right_to_left'
        """
        if x_vals is None or y_vals is None:
            return [], []
        if not isinstance(x_vals, (list, np.ndarray)) or not isinstance(y_vals, (list, np.ndarray)):
            return [], []
        if len(x_vals) == 0 or len(y_vals) == 0:
            return [], []

        # Transform: add 52.5 to x, add 34 to y (centered to standard coords)
        x_transformed = [x + 52.5 for x in x_vals]
        y_transformed = [y + 34.0 for y in y_vals]

        # Flip left_to_right attacks so they all appear as attacking the right goal
        # left_to_right = attacking FROM left (players on left) → flip to show on right
        # right_to_left = attacking FROM right (players already on right) → keep as-is
        if attacking_side == 'left_to_right':
            x_transformed = [105.0 - x for x in x_transformed]

        return x_transformed, y_transformed

    # Draw pitch (simplified - just penalty area)
    pitch_color = '#2d5f3f'
    line_color = 'white'

    ax.set_facecolor(pitch_color)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)

    # Penalty area (right side - attacking direction)
    penalty_box = patches.Rectangle((88.5, 13.84), 16.5, 40.32,
                                     linewidth=2, edgecolor=line_color,
                                     facecolor='none')
    ax.add_patch(penalty_box)

    # Goal area
    goal_box = patches.Rectangle((99.5, 24.84), 5.5, 18.32,
                                  linewidth=2, edgecolor=line_color,
                                  facecolor='none')
    ax.add_patch(goal_box)

    # Goal line
    ax.plot([105, 105], [30.34, 37.66], color=line_color, linewidth=3)

    # Corner arcs (both sides)
    corner_arc_1 = patches.Arc((105, 0), 2, 2, angle=0, theta1=90, theta2=180,
                               linewidth=2, edgecolor=line_color, facecolor='none')
    corner_arc_2 = patches.Arc((105, 68), 2, 2, angle=0, theta1=180, theta2=270,
                               linewidth=2, edgecolor=line_color, facecolor='none')
    ax.add_patch(corner_arc_1)
    ax.add_patch(corner_arc_2)

    # Extract and transform player positions
    try:
        # Get attacking direction
        attacking_side = corner_data.get('attacking_side', 'left_to_right')

        # Attacking players (red)
        att_x = corner_data.get('att_x', [])
        att_y = corner_data.get('att_y', [])
        att_x_t, att_y_t = transform_coords(att_x, att_y, attacking_side)
        if len(att_x_t) > 0:
            ax.scatter(att_x_t, att_y_t, c='red', s=100, alpha=0.8,
                      edgecolors='white', linewidth=1.5, zorder=10, label='Attack')

        # Defending players (blue)
        def_x = corner_data.get('def_x', [])
        def_y = corner_data.get('def_y', [])
        def_x_t, def_y_t = transform_coords(def_x, def_y, attacking_side)
        if len(def_x_t) > 0:
            ax.scatter(def_x_t, def_y_t, c='blue', s=100, alpha=0.8,
                      edgecolors='white', linewidth=1.5, zorder=10, label='Defense')

        # Ball position (yellow star) - also transform
        ball_x = corner_data.get('ball_x', 52.5)  # Default to right corner in centered coords
        ball_y = corner_data.get('ball_y', 34.0)
        ball_x_t, ball_y_t = transform_coords([ball_x], [ball_y], attacking_side)
        if len(ball_x_t) > 0:
            ax.scatter(ball_x_t[0], ball_y_t[0], c='yellow', s=200, marker='*',
                      edgecolors='black', linewidth=1.5, zorder=20, label='Ball')
    except Exception as e:
        print(f"  Warning: Could not plot players - {e}")

    if show_title:
        # Add CTI value as title
        cti_val = corner_data.get('cti', 0)
        ax.set_title(f"CTI: {cti_val:.3f}", fontsize=10, fontweight='bold', color='white')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    # Add small legend
    ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

def create_team_top_corners_dashboard(n_teams=10, corners_per_team=3):
    """Create dashboard showing top corners for top teams"""

    print(f"\nCreating top corners visualization for top {n_teams} teams...")

    # Get top teams
    top_teams = team_cti.head(n_teams)

    # Create figure
    fig = plt.figure(figsize=(20, 4 * n_teams))
    fig.suptitle('Top Corner Situations by Team - CTI Model Analysis',
                 fontsize=24, fontweight='bold', y=0.995)

    # Create grid: each team gets 1 row with logo + corners_per_team corner plots
    gs = fig.add_gridspec(n_teams, corners_per_team + 1,
                          width_ratios=[1] + [2]*corners_per_team,
                          hspace=0.4, wspace=0.3,
                          left=0.05, right=0.98, top=0.97, bottom=0.03)

    for idx, team_row in enumerate(top_teams.iter_rows(named=True)):
        team_id = team_row['team_id']
        team_name = team_name_map.get(team_id, f"Team {team_id}")
        team_cti_avg = team_row['cti_avg']
        n_corners = team_row['n_corners']

        print(f"  Processing {team_name} (CTI: {team_cti_avg:.3f}, {n_corners} corners)...")

        # Logo axis
        ax_logo = fig.add_subplot(gs[idx, 0])
        ax_logo.axis('off')

        # Try to load and display logo
        logo = get_team_logo(team_id)
        if logo:
            ax_logo.imshow(logo)
            ax_logo.set_title(f"#{idx+1}", fontsize=16, fontweight='bold', pad=10)
        else:
            # No logo - just show team info
            ax_logo.text(0.5, 0.6, f"#{idx+1}", ha='center', va='center',
                        fontsize=40, fontweight='bold', color='#2c3e50')
            ax_logo.text(0.5, 0.3, team_name, ha='center', va='center',
                        fontsize=10, fontweight='bold', color='#34495e', wrap=True)

        # Add team stats below logo
        stats_text = f"{team_name}\n"
        stats_text += f"CTI: {team_cti_avg:.3f}\n"
        stats_text += f"P(Shot): {team_row['y1_avg']:.1%}\n"
        stats_text += f"Corners: {n_corners:,}"

        ax_logo.text(0.5, -0.15, stats_text, ha='center', va='top',
                    fontsize=9, color='#2c3e50',
                    bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8),
                    transform=ax_logo.transAxes)

        # Get top corners for this team
        team_corners = corners_df.filter(pl.col('team_id') == team_id)
        team_predictions = predictions.filter(pl.col('team_id') == team_id)

        # Join to get CTI values
        team_data = team_corners.join(
            team_predictions.select(['corner_id', 'cti']),
            on='corner_id',
            how='inner'
        ).sort('cti', descending=True).head(corners_per_team)

        # Plot top corners
        for corner_idx in range(corners_per_team):
            ax_corner = fig.add_subplot(gs[idx, corner_idx + 1])

            if corner_idx < len(team_data):
                # Get row as dict (Polars way)
                corner_row = team_data.row(corner_idx, named=True)

                # Convert List columns to Python lists
                corner_dict = {
                    'att_x': corner_row['att_x'].to_list() if hasattr(corner_row['att_x'], 'to_list') else list(corner_row['att_x']) if corner_row['att_x'] is not None else [],
                    'att_y': corner_row['att_y'].to_list() if hasattr(corner_row['att_y'], 'to_list') else list(corner_row['att_y']) if corner_row['att_y'] is not None else [],
                    'def_x': corner_row['def_x'].to_list() if hasattr(corner_row['def_x'], 'to_list') else list(corner_row['def_x']) if corner_row['def_x'] is not None else [],
                    'def_y': corner_row['def_y'].to_list() if hasattr(corner_row['def_y'], 'to_list') else list(corner_row['def_y']) if corner_row['def_y'] is not None else [],
                    'ball_x': corner_row.get('ball_x', 105),
                    'ball_y': corner_row.get('ball_y', 68),
                    'attacking_side': corner_row.get('attacking_side', 'left_to_right'),
                    'cti': corner_row['cti']
                }

                plot_corner_snapshot(ax_corner, corner_dict, show_title=True)

                # Add rank label
                rank_text = f"Top {corner_idx + 1}"
                ax_corner.text(0.02, 0.98, rank_text,
                              transform=ax_corner.transAxes,
                              fontsize=9, fontweight='bold',
                              va='top', ha='left',
                              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            else:
                ax_corner.text(0.5, 0.5, 'No data', ha='center', va='center',
                              fontsize=12, color='gray')
                ax_corner.set_xticks([])
                ax_corner.set_yticks([])

    # Save figure
    output_path = OUT_FIG / 'team_top_corners_visual.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nOK Saved visualization: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    plt.close()

    return output_path

def create_compact_version(n_teams=20, corners_per_team=1):
    """Create compact version showing all teams with 1 corner each"""

    print(f"\nCreating compact version for all {n_teams} teams...")

    # Get all teams
    all_teams = team_cti.head(n_teams)

    # Create figure with grid layout
    n_cols = 4
    n_rows = (n_teams + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(20, 5 * n_rows))
    fig.suptitle('Top Corner for Each Team - CTI Rankings',
                 fontsize=22, fontweight='bold', y=0.995)

    gs = fig.add_gridspec(n_rows, n_cols,
                          hspace=0.5, wspace=0.3,
                          left=0.05, right=0.98, top=0.97, bottom=0.03)

    for idx, team_row in enumerate(all_teams.iter_rows(named=True)):
        row_idx = idx // n_cols
        col_idx = idx % n_cols

        team_id = team_row['team_id']
        team_name = team_name_map.get(team_id, f"Team {team_id}")
        team_cti_avg = team_row['cti_avg']

        print(f"  #{idx+1} {team_name} (CTI: {team_cti_avg:.3f})...")

        ax = fig.add_subplot(gs[row_idx, col_idx])

        # Get top corner for this team
        team_corners = corners_df.filter(pl.col('team_id') == team_id)
        team_predictions = predictions.filter(pl.col('team_id') == team_id)

        team_data = team_corners.join(
            team_predictions.select(['corner_id', 'cti']),
            on='corner_id',
            how='inner'
        ).sort('cti', descending=True).head(1)

        if len(team_data) > 0:
            # Get row as dict (Polars way)
            corner_row = team_data.row(0, named=True)

            # Convert List columns to Python lists
            corner_dict = {
                'att_x': corner_row['att_x'].to_list() if hasattr(corner_row['att_x'], 'to_list') else list(corner_row['att_x']) if corner_row['att_x'] is not None else [],
                'att_y': corner_row['att_y'].to_list() if hasattr(corner_row['att_y'], 'to_list') else list(corner_row['att_y']) if corner_row['att_y'] is not None else [],
                'def_x': corner_row['def_x'].to_list() if hasattr(corner_row['def_x'], 'to_list') else list(corner_row['def_x']) if corner_row['def_x'] is not None else [],
                'def_y': corner_row['def_y'].to_list() if hasattr(corner_row['def_y'], 'to_list') else list(corner_row['def_y']) if corner_row['def_y'] is not None else [],
                'ball_x': corner_row.get('ball_x', 105),
                'ball_y': corner_row.get('ball_y', 68),
                'attacking_side': corner_row.get('attacking_side', 'left_to_right'),
                'cti': corner_row['cti']
            }

            plot_corner_snapshot(ax, corner_dict, show_title=False)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # Add team info
        # Try to add logo
        logo = get_team_logo(team_id)
        if logo:
            # Create inset axis for logo
            ax_inset = ax.inset_axes([0.02, 0.75, 0.2, 0.2])
            ax_inset.imshow(logo)
            ax_inset.axis('off')

        # Team name and stats
        title_text = f"#{idx+1} {team_name}\nCTI: {team_cti_avg:.3f}"
        ax.set_title(title_text, fontsize=10, fontweight='bold', pad=10)

    # Save compact version
    output_path = OUT_FIG / 'all_teams_top_corner.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nOK Saved compact visualization: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    plt.close()

    return output_path

if __name__ == '__main__':
    print("="*60)
    print("Generating Top Corners Visualizations")
    print("="*60)

    # Check if logos directory exists
    if not LOGO_DIR.exists():
        print(f"\nWarning: Logo directory not found: {LOGO_DIR}")
        print("Continuing without logos...")
    else:
        n_logos = len(list(LOGO_DIR.glob('*.png')))
        print(f"Found {n_logos} team logos")

    # Create detailed version (top 10 teams, 3 corners each)
    # detailed_path = create_team_top_corners_dashboard(n_teams=10, corners_per_team=3)

    # Create compact version (all 20 teams, 1 corner each)
    # compact_path = create_compact_version(n_teams=20, corners_per_team=1)

    print("\n" + "="*60)
    print("DONE - Created 2 visualizations:")
    print(f"  1. Detailed: {detailed_path.name}")
    print(f"  2. Compact: {compact_path.name}")
    print("="*60)
