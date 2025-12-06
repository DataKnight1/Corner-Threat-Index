"""
Author: Tiago
Date: 2025-12-04
Description: Generate team-specific pitch visualizations with NMF features.
"""
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from mplsoccer import Pitch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import warnings
warnings.filterwarnings('ignore')

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'cti_data'
OUT_FIG = ROOT / 'cti_outputs'
LOGO_DIR = ROOT / 'assets'  # Logos are directly in assets folder with sanitized names

# Professional colors
BG_COLOR = '#313332'
PITCH_COLOR = '#167d3b'
ARROW_COLOR = '#ff6b6b'  # Light red for tactical patterns
DOT_COLOR = '#4a90e2'    # Keep blue for initial positions for contrast

# Team name mapping
team_name_map = {
    2: "Liverpool Football Club", 3: "Arsenal Football Club", 31: "Manchester United",
    32: "Newcastle United", 37: "West Ham United", 39: "Aston Villa", 40: "Manchester City",
    41: "Everton", 44: "Tottenham Hotspur", 48: "Fulham", 49: "Chelsea",
    52: "Wolverhampton Wanderers", 58: "Southampton", 60: "Crystal Palace",
    62: "Leicester City", 63: "Bournemouth", 308: "Brighton and Hove Albion",
    747: "Nottingham Forest", 752: "Ipswich Town", 754: "Brentford FC"
}

def sanitize_team_name(name: str) -> str:
    """Sanitize team name to match logo filename format"""
    return ''.join(ch.lower() for ch in str(name) if ch.isalnum())


def get_team_logo(team_name):
    """Get team logo image using sanitized name"""
    sanitized = sanitize_team_name(team_name)
    logo_path = LOGO_DIR / f"{sanitized}.png"
    if logo_path.exists():
        try:
            return Image.open(logo_path).convert('RGBA')
        except:
            return None
    return None


def load_nmf_and_zones():
    """Load NMF model and zone models"""
    import pickle
    from cti_gmm_zones import load_zone_models

    nmf_path = DATA_DIR / 'nmf_model.pkl'
    zones_path = DATA_DIR / 'gmm_zones.pkl'

    if not nmf_path.exists():
        raise FileNotFoundError(f"NMF model not found: {nmf_path}")
    if not zones_path.exists():
        raise FileNotFoundError(f"Zone models not found: {zones_path}")

    with open(nmf_path, 'rb') as f:
        nmf_model = pickle.load(f)

    zone_models = load_zone_models(zones_path)

    return nmf_model, zone_models


def load_xt_surface():
    """Load xThreat surface model"""
    import pickle

    xt_path = DATA_DIR / 'xt_surface.pkl'
    if not xt_path.exists():
        print(f"Warning: xT surface not found: {xt_path}")
        return None

    try:
        with open(xt_path, 'rb') as f:
            xt_surface = pickle.load(f)
        return xt_surface
    except Exception as e:
        print(f"Warning: Failed to load xT surface: {e}")
        return None


def plot_nmf_feature_on_pitch(ax, feature_id, nmf_model, zone_models,
                               xt_surface=None, weight_threshold=0.03, max_runs=8):
    """Draw NMF feature pattern on a pitch axes with improved visibility and xT heatmap"""
    H = nmf_model.H  # (n_topics, 42)

    # Initialize pitch with thicker lines for better visibility
    pitch = Pitch(
        pitch_type='custom', pitch_length=105, pitch_width=68, half=True,
        pitch_color=PITCH_COLOR, line_color='white', linewidth=1.5
    )
    pitch.draw(ax=ax)
    ax.set_facecolor(BG_COLOR)

    # Add xThreat surface heatmap as background
    if xt_surface is not None:
        try:
            # xT surface is (40, 40) for half-pitch attacking zone (52.5-105m x 0-68m)
            # First dimension: x-direction (52.5-105m)
            # Second dimension: y-direction (0-68m)
            # pcolormesh expects data as (n_y, n_x), so transpose
            n_x, n_y = xt_surface.shape
            x_bins = np.linspace(52.5, 105, n_x + 1)
            y_bins = np.linspace(0, 68, n_y + 1)

            # Plot heatmap with low alpha so it doesn't overwhelm the tactical patterns
            # Use warm colormap (Reds) for threat, subtle alpha
            im = ax.pcolormesh(x_bins, y_bins, xt_surface.T,
                              cmap='Reds', alpha=0.2, zorder=1,
                              vmin=0, vmax=xt_surface.max())
        except Exception as e:
            print(f"  Warning: Failed to plot xT surface: {e}")

    # Get zone means
    init_means = zone_models.gmm_init.means_  # Shape corrected coords
    tgt_means_all = zone_models.gmm_tgt.means_
    tgt_ids = zone_models.active_tgt_ids
    tgt_means = tgt_means_all[tgt_ids]

    # Convert to standard coords
    def sc_to_std(arr):
        return np.column_stack((arr[:, 0] + 52.5, arr[:, 1] + 34.0))

    init_std = sc_to_std(init_means)
    tgt_std = sc_to_std(tgt_means)

    # Get feature topic
    topic = H[feature_id].reshape(6, 7)

    # Select top runs to visualize FIRST
    runs = []
    for i in range(6):
        for j in range(7):
            w = float(topic[i, j])
            if w >= weight_threshold:
                runs.append((w, i, j))

    if runs:
        runs.sort(reverse=True)
        runs = runs[:max_runs]
        wmax = runs[0][0] if runs[0][0] > 0 else 1.0

        # Draw arrows first (so they appear behind dots)
        for w, i, j in runs:
            rel = max(0.0, min(1.0, w / wmax))
            lw = 2.0 + 4.5 * rel
            x0, y0 = init_std[i]
            x1, y1 = tgt_std[j]
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='-|>', lw=lw,
                                        color=ARROW_COLOR, alpha=0.75,
                                        shrinkA=0, shrinkB=0,
                                        connectionstyle='arc3,rad=-0.25'))

    # Draw initial zone dots LAST (on top of arrows) with MUCH better visibility
    # Larger size, thicker white edge, fully opaque
    ax.scatter(init_std[:, 0], init_std[:, 1], s=120, c=DOT_COLOR,
               zorder=15, edgecolors='white', linewidths=2.5, alpha=1.0)


def create_team_nmf_pitch_dashboard():
    """Create pitch visualization dashboard for each team's top NMF feature"""

    print("\nCreating team NMF pitch dashboard with logos...")

    # Load data
    team_cti_df = pl.read_csv(DATA_DIR / 'team_cti_detailed.csv')
    team_top_df = pl.read_csv(DATA_DIR / 'team_top_feature.csv')

    # Load NMF model and xT surface
    try:
        nmf_model, zone_models = load_nmf_and_zones()
        xt_surface = load_xt_surface()
    except Exception as e:
        print(f"Error loading NMF/zones: {e}")
        return None

    # Merge to get CTI rankings
    merged = team_top_df.join(
        team_cti_df.select(['team', 'cti_avg']),
        on='team', how='left'
    ).sort('cti_avg', descending=True).head(20)

    # Create figure - 4 rows x 5 columns = 20 teams
    fig = plt.figure(figsize=(26, 22))
    fig.patch.set_facecolor(BG_COLOR)

    # Title
    fig.text(0.5, 0.98, 'Team Corner Patterns - Top NMF Feature by Team',
             ha='center', va='top', fontsize=26, fontweight='bold',
             color='white')
    fig.text(0.5, 0.965, 'Tactical patterns showing player movement from corner kicks',
             ha='center', va='top', fontsize=14, color='white', alpha=0.8)

    # Create grid
    gs = fig.add_gridspec(4, 5, hspace=0.45, wspace=0.35,
                          left=0.05, right=0.97, top=0.94, bottom=0.05)

    # Find team_id mapping
    team_to_id = {v: k for k, v in team_name_map.items()}

    for idx, team_row in enumerate(merged.iter_rows(named=True)):
        row_idx = idx // 5
        col_idx = idx % 5

        team_name = team_row['team']
        team_id = team_to_id.get(team_name)
        feature_id = int(team_row['top_feature_id']) - 1  # Convert to 0-indexed
        top_weight = team_row['top_weight']
        n_corners = team_row['n_corners']

        # Create subplot
        ax = fig.add_subplot(gs[row_idx, col_idx])

        # Plot NMF feature pattern with xT surface
        try:
            plot_nmf_feature_on_pitch(ax, feature_id, nmf_model, zone_models, xt_surface)
        except Exception as e:
            print(f"  Warning: Failed to plot feature for {team_name}: {e}")
            ax.set_facecolor(BG_COLOR)
            ax.axis('off')
            continue

        # Add team logo using OffsetImage and AnnotationBbox (same method as working code)
        logo = get_team_logo(team_name)
        if logo:
            try:
                # Resize logo to consistent height
                aspect = logo.width / max(1, logo.height)
                target_h = 70  # Slightly larger than the 60 in the original
                target_w = max(1, int(target_h * aspect))
                logo_resized = logo.resize((target_w, target_h), Image.Resampling.LANCZOS)

                # Create OffsetImage
                oi = OffsetImage(np.asarray(logo_resized), zoom=1.0, zorder=20)

                # Create AnnotationBbox at top-left of axes
                ab = AnnotationBbox(oi, (0.10, 0.88), frameon=False,
                                   xycoords='axes fraction',
                                   box_alignment=(0.5, 0.5),
                                   zorder=20)
                ax.add_artist(ab)
                print(f"  Added logo for {team_name}")
            except Exception as e:
                print(f"  Warning: Failed to add logo for {team_name}: {e}")
        else:
            print(f"  No logo found for {team_name} (looking for {sanitize_team_name(team_name)}.png)")

        # Team name and metrics
        ax.text(0.5, -0.08, team_name, transform=ax.transAxes,
                ha='center', va='top', fontsize=12, fontweight='bold',
                color='white')

        ax.text(0.5, -0.15, f'Feature #{int(team_row["top_feature_id"])} • Weight: {top_weight:.3f}',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=10, color='white', alpha=0.85)

        ax.text(0.5, -0.21, f'{n_corners} corners analyzed',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=9, color='white', alpha=0.7)

        # Rank badge
        rank = idx + 1
        rank_color = '#FFD700' if rank <= 3 else '#C0C0C0' if rank <= 10 else '#CD7F32'
        circle = plt.Circle((0.90, 0.92), 0.045, transform=ax.transAxes,
                           color=rank_color, zorder=10, edgecolor='white', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(0.90, 0.92, f'{rank}', transform=ax.transAxes,
               ha='center', va='center', fontsize=11, fontweight='bold',
               color='black', zorder=11)

    # Legend with improved styling
    from matplotlib.lines import Line2D
    dot = Line2D([0], [0], marker='o', color='w', label='Initial position zone',
                 markerfacecolor=DOT_COLOR, markersize=10, markeredgecolor='white', markeredgewidth=2)
    arr = Line2D([0], [0], color=ARROW_COLOR, lw=3.5, label='Tactical pattern (player runs)',
                 marker='', linestyle='-')

    leg = fig.legend(handles=[dot, arr], loc='lower center', ncol=2,
                     frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.00))
    for txt in leg.get_texts():
        txt.set_color('white')

    # Save with higher DPI for better quality
    output_path = OUT_FIG / 'team_nmf_pitch_dashboard.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    print(f"\nOK Saved: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    plt.close()

    return output_path


if __name__ == '__main__':
    print("="*70)
    print("Generating Team Pitch Visualizations WITH BADGES")
    print("="*70)

    # Check logos
    if not LOGO_DIR.exists():
        print(f"\nError: Logo directory not found: {LOGO_DIR}")
    else:
        n_logos = len(list(LOGO_DIR.glob('*.png')))
        print(f"\nFound {n_logos} PNG files in {LOGO_DIR}")

    # Generate visualizations
    try:
        pitch_path = create_team_nmf_pitch_dashboard()
    except Exception as e:
        print(f"\nError creating pitch dashboard: {e}")
        import traceback
        traceback.print_exc()
        pitch_path = None

    print("\n" + "="*70)
    if pitch_path:
        print(f"DONE - Created: {pitch_path.name}")
    print("="*70)
