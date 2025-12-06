"""
Author: Tiago
Date: 2025-12-04
Description: Generate comprehensive professional HTML reliability report with interactive features.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Tuple
import base64
import pickle
from PIL import Image
import io


def sanitize_team_name(name: str) -> str:
    """
    Sanitize team name to match logo filename format (lowercase alphanumeric).

    :param name: Original team name.
    :return: Sanitized name string.
    """
    return ''.join(ch.lower() for ch in str(name) if ch.isalnum())


def compute_calibration_metrics(y_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Compute calibration metrics including Expected Calibration Error (ECE),
    Maximum Calibration Error (MCE), and Brier score.

    :param y_pred: Predicted probabilities.
    :param y_true: Binary ground truth labels.
    :param n_bins: Number of bins for calibration curve.
    :return: Dictionary containing calibration metrics and curve data.
    """
    from sklearn.calibration import calibration_curve
    from scipy.stats import pearsonr

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')

    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_totals = np.histogram(y_pred, bins=bin_edges)[0]
    non_empty_indices = np.where(bin_totals > 0)[0]
    n_returned_bins = len(prob_true)
    weights = np.zeros(n_returned_bins)

    for i in range(min(n_returned_bins, len(non_empty_indices))):
        bin_idx = non_empty_indices[i]
        weights[i] = bin_totals[bin_idx] / len(y_pred)

    ece = np.sum(weights * np.abs(prob_true - prob_pred))
    mce = np.max(np.abs(prob_true - prob_pred))
    brier = np.mean((y_pred - y_true) ** 2)

    # Correlation
    try:
        corr, _ = pearsonr(y_pred, y_true)
    except:
        corr = 0.0

    return {
        'ece': float(ece),
        'mce': float(mce),
        'brier': float(brier),
        'correlation': float(corr),
        'prob_true': prob_true.tolist(),
        'prob_pred': prob_pred.tolist(),
        'bin_counts': bin_totals.tolist()
    }


def load_xt_surface():
    """
    Load the xT surface model from the data directory.

    :return: The loaded xT surface object (numpy array) or None if not found.
    """
    import pickle
    from cti_paths import FINAL_PROJECT_DIR

    xt_path = FINAL_PROJECT_DIR / 'cti_data' / 'xt_surface.pkl'
    if not xt_path.exists():
        return None

    try:
        with open(xt_path, 'rb') as f:
            xt_surface = pickle.load(f)
        return xt_surface
    except Exception as e:
        print(f"Warning: Failed to load xT surface: {e}")
        return None


def generate_team_corner_animations(
    team_id: int,
    team_name: str,
    nmf_model,
    zone_models,
    team_features: np.ndarray,
    output_dir: Path,
    xt_surface=None,
    n_frames: int = 20
) -> str:
    """
    Generate an animated GIF showing the dominant corner pattern for a team.

    Visualizes the transition from initial delivery zones to target zones based on NMF features.

    :param team_id: Unique team identifier.
    :param team_name: Name of the team.
    :param nmf_model: Trained NMF model object.
    :param zone_models: GMM zone models.
    :param team_features: NMF feature weights for the team.
    :param output_dir: Directory to save the generated GIF.
    :param xt_surface: Optional xT surface for background heatmap.
    :param n_frames: Number of frames in the animation.
    :return: Base64 data URI of the generated GIF.
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    import matplotlib.patches as mpatches

    try:
        # Get top feature for this team
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

        if not runs:
            return None

        runs.sort(reverse=True)
        runs = runs[:8]
        wmax = runs[0][0] if runs[0][0] > 0 else 1.0

        # Create animation frames
        frames = []
        for frame_idx in range(n_frames):
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')

            pitch = Pitch(
                pitch_type='custom', pitch_length=105, pitch_width=68, half=True,
                pitch_color='white', line_color='#333', linewidth=1.5
            )
            pitch.draw(ax=ax)
            ax.set_facecolor('white')

            # Add xThreat surface heatmap as background
            if xt_surface is not None:
                try:
                    # Transpose xT surface for pcolormesh
                    n_x, n_y = xt_surface.shape
                    x_bins = np.linspace(52.5, 105, n_x + 1)
                    y_bins = np.linspace(0, 68, n_y + 1)
                    ax.pcolormesh(x_bins, y_bins, xt_surface.T,
                                 cmap='Reds', alpha=0.15, zorder=1,
                                 vmin=0, vmax=xt_surface.max())
                except:
                    pass

            # Progress ratio for animation
            progress = frame_idx / (n_frames - 1)

            # Draw arrows with animation
            for w, i, j in runs:
                rel = max(0.0, min(1.0, w / wmax))
                lw = 2.0 + 4.0 * rel
                x0, y0 = init_std[i]
                x1, y1 = tgt_std[j]

                # Animate from start to end
                x_curr = x0 + (x1 - x0) * progress
                y_curr = y0 + (y1 - y0) * progress

                if progress < 1.0:
                    ax.plot([x0, x_curr], [y0, y_curr],
                           color='#333', lw=lw, alpha=0.7, zorder=5)
                    ax.scatter([x_curr], [y_curr], s=100, c='#333',
                             zorder=10, edgecolors='white', linewidths=2)
                else:
                    # Final frame shows arrow
                    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                               arrowprops=dict(arrowstyle='-|>', lw=lw,
                                             color='#333', alpha=0.7))

            # Initial positions
            ax.scatter(init_std[:, 0], init_std[:, 1], s=100, c='white',
                      zorder=15, edgecolors='#333', linewidths=2.5)

            # Title
            ax.set_title(f'{team_name} - Corner Pattern', fontsize=12, fontweight='bold', pad=10)

            plt.tight_layout()

            # Save frame to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, facecolor='white', bbox_inches='tight')
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            plt.close(fig)
            buf.close()

        # Save as GIF
        sanitized = sanitize_team_name(team_name)
        gif_path = output_dir / f'team_{sanitized}_pattern.gif'
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )

        # Convert to base64
        with open(gif_path, 'rb') as f:
            gif_data = base64.b64encode(f.read()).decode()

        return f"data:image/gif;base64,{gif_data}"

    except Exception as e:
        print(f"Warning: Failed to generate animation for {team_name}: {e}")
        return None


def create_enhanced_reliability_html(
    predictions_df: pl.DataFrame,
    team_cti_df: pl.DataFrame,
    team_top_feature_df: pl.DataFrame,
    output_path: Path,
    nmf_model=None,
    zone_models=None,
    title: str = "CTI Model"
):
    """
    Create a comprehensive HTML report with performance metrics, interactive team analysis,
    reliability curves, and scatter plots.

    :param predictions_df: DataFrame containing model predictions (y1-y5) and empirical baselines.
    :param team_cti_df: DataFrame containing team-level CTI statistics.
    :param team_top_feature_df: DataFrame containing top tactical features per team.
    :param output_path: Path to save the HTML report.
    :param nmf_model: Optional NMF model for visualizations.
    :param zone_models: Optional GMM zone models for visuals.
    :param title: Title of the HTML report.
    """

    from cti_paths import FINAL_PROJECT_DIR

    # Compute metrics for all outputs
    y1_pred = predictions_df['y1'].to_numpy()
    y1_true = predictions_df['y1_e'].to_numpy()
    y2_pred = predictions_df['y2'].to_numpy()
    y2_true = predictions_df['y2_e'].to_numpy()
    y3_pred = predictions_df['y3'].to_numpy()
    y3_true = predictions_df['y3_e'].to_numpy()
    y4_pred = predictions_df['y4'].to_numpy()
    y4_true = predictions_df['y4_e'].to_numpy()
    y5_pred = predictions_df['y5'].to_numpy()
    y5_true = predictions_df['y5_e'].to_numpy()
    cti_pred = predictions_df['cti'].to_numpy()
    cti_true = predictions_df['cti_e'].to_numpy()

    y1_metrics = compute_calibration_metrics(y1_pred, y1_true)
    y2_metrics = {'correlation': float(np.corrcoef(y2_pred, y2_true)[0, 1])}
    y3_metrics = compute_calibration_metrics(y3_pred, y3_true)
    y4_metrics = {'correlation': float(np.corrcoef(y4_pred, y4_true)[0, 1])}
    y5_metrics = {'correlation': float(np.corrcoef(y5_pred, y5_true)[0, 1])}
    cti_metrics = {'correlation': float(np.corrcoef(cti_pred, cti_true)[0, 1])}

    # Load team logos
    assets_dir = FINAL_PROJECT_DIR / 'assets'
    team_logos = {}
    team_name_map = {
        2: "Liverpool Football Club", 3: "Arsenal Football Club", 31: "Manchester United",
        32: "Newcastle United", 37: "West Ham United", 39: "Aston Villa", 40: "Manchester City",
        41: "Everton", 44: "Tottenham Hotspur", 48: "Fulham", 49: "Chelsea",
        52: "Wolverhampton Wanderers", 58: "Southampton", 60: "Crystal Palace",
        62: "Leicester City", 63: "Bournemouth", 308: "Brighton and Hove Albion",
        747: "Nottingham Forest", 752: "Ipswich Town", 754: "Brentford FC"
    }

    for team_id, team_name in team_name_map.items():
        sanitized = sanitize_team_name(team_name)
        logo_path = assets_dir / f"{sanitized}.png"
        if logo_path.exists():
            with open(logo_path, 'rb') as f:
                logo_data = base64.b64encode(f.read()).decode()
                team_logos[team_name] = f"data:image/png;base64,{logo_data}"

    output_dir = output_path.parent

    def load_png_base64(path: Path) -> str:
        """Return base64-encoded data URI for a PNG if it exists, else empty string."""
        if path.exists():
            with open(path, 'rb') as f:
                return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
        print(f"Warning: Missing image for HTML report: {path}")
        return ""

    cti_table_v2_img = load_png_base64(output_dir / 'team_cti_table_v2.png')
    offense_counter_img = load_png_base64(output_dir / 'team_offense_vs_counter_presentation.png')

    # Load team_cti_v2.csv
    team_cti_v2_path = FINAL_PROJECT_DIR / 'cti_data' / 'team_cti_v2.csv'
    team_cti_v2_df = None
    if team_cti_v2_path.exists():
        team_cti_v2_df = pl.read_csv(team_cti_v2_path)

    # Generate team performance rows
    team_performance_rows = ""
    if team_cti_v2_df is not None:
        # Sort by CTI average
        sorted_df = team_cti_v2_df.sort('cti_avg', descending=True)
        for idx, row in enumerate(sorted_df.iter_rows(named=True)):
            team_name = row['team']
            cti_val = row['cti_avg']
            p_shot = row['p_shot']
            counter_risk = row['counter_risk']
            delta_xt = row['delta_xt']
            n_corners = row['n_corners']

            # Determine CTI badge class (Scaled 0-100)
            if cti_val >= 6.5:
                badge_class = "excellent"
            elif cti_val >= 5.5:
                badge_class = "good"
            elif cti_val >= 4.5:
                badge_class = "average"
            else:
                badge_class = "poor"

            # Get team logo
            logo_html = ""
            if team_name in team_logos:
                logo_html = f'<img src="{team_logos[team_name]}" alt="{team_name}" class="team-logo">'

            display_name = team_name.replace(" Football Club", "").replace(" and Hove Albion", "")

            # Generate tooltip content
            tooltip_content = f"""
                <div class="tooltip-content">
                    <strong>{display_name}</strong><br>
                    <span style="color: #ddd;">Corners: {n_corners}</span><br>
                    <span style="color: #4ade80;">Shot Prob: {p_shot*100:.1f}%</span><br>
                    <span style="color: #f87171;">Counter Risk: {counter_risk*100:.2f}%</span><br>
                    <span style="color: #fbbf24;">ΔxT: {delta_xt:.4f}</span>
                </div>
            """

            team_performance_rows += f"""
                <tr class="team-row">
                    <td class="number-cell" style="text-align: center;">{idx + 1}</td>
                    <td>
                        <div class="team-logo-container">
                            {logo_html}
                            {tooltip_content}
                        </div>
                    </td>
                    <td style="font-weight: 600; text-align: left;">{display_name}</td>
                    <td class="number-cell" style="text-align: center;">
                        <span class="cti-badge-{badge_class}">{cti_val:.5f}</span>
                    </td>
                    <td class="number-cell" style="text-align: center;">{p_shot*100:.1f}%</td>
                    <td class="number-cell" style="text-align: center;">{counter_risk*100:.2f}%</td>
                    <td class="number-cell" style="text-align: center;">{delta_xt:.4f}</td>
                    <td class="number-cell" style="text-align: center;">{n_corners}</td>
                </tr>
            """

    # Generate team comparison visuals (side-by-side pitches) if models available
    team_animations = {}
    if nmf_model and zone_models:
        print("\nGenerating team comparison visuals (tactical pattern + predictions)...")
        animations_dir = output_path.parent / 'team_animations'
        animations_dir.mkdir(exist_ok=True)

        # Load xT surface
        xt_surface = load_xt_surface()

        # Load team features
        W_path = FINAL_PROJECT_DIR / 'cti_data' / 'nmf_model.pkl'
        with open(W_path, 'rb') as f:
            nmf_data = pickle.load(f)
            W = nmf_data.W if hasattr(nmf_data, 'W') else None

        # Load corners dataset to get average ball positions per team
        corners_path = FINAL_PROJECT_DIR / 'cti_data' / 'corners_dataset.parquet'
        team_corner_positions = {}
        if corners_path.exists():
            corners_df_full = pl.read_parquet(corners_path)
            for team_id in team_cti_df['team_id'].unique():
                team_corners = corners_df_full.filter(pl.col('team_id') == team_id)
                if team_corners.height > 0:
                    avg_x = team_corners['x_start'].mean()
                    avg_y = team_corners['y_start'].mean()
                    team_corner_positions[team_id] = {'avg_x': avg_x, 'avg_y': avg_y}

        # Import comparison visual generator
        from cti_generate_comparison_visuals import generate_team_comparison_visual

        if W is not None:
            for idx, row in enumerate(team_cti_df.iter_rows(named=True)):
                team_id = row['team_id']
                team_name = team_name_map.get(team_id)
                if team_name and idx < len(W):
                    # Build metrics dict
                    team_metrics = {
                        'y1_avg': row['y1_avg'],
                        'y3_avg': row['y3_avg'],
                        'y5_avg': row['y5_avg'],
                        'cti_avg': row['cti_avg']
                    }

                    # Get team corner positions
                    team_pos = team_corner_positions.get(team_id, None)

                    # Generate side-by-side comparison
                    img_data = generate_team_comparison_visual(
                        team_name, nmf_model, zone_models,
                        W[idx], team_metrics, xt_surface, team_pos
                    )
                    if img_data:
                        team_animations[team_name] = img_data
                        print(f"  [OK] Generated comparison for {team_name}")

    # Build team options for dropdown
    team_options = ''.join([
        f'<option value="{sanitize_team_name(name)}">{name.replace(" Football Club", "")}</option>'
        for name in sorted(team_logos.keys())
    ])

    # Build team data JSON for JavaScript
    team_data_json = "{\n"
    for team_name in sorted(team_logos.keys()):
        sanitized = sanitize_team_name(team_name)
        team_row = team_cti_df.filter(pl.col('team_id') == [k for k, v in team_name_map.items() if v == team_name][0])
        if len(team_row) > 0:
            row = team_row.row(0, named=True)
            cti_val = row['cti_avg']
            y1_val = row['y1_avg']
            y3_val = row['y3_avg']
            y5_val = row['y5_avg']
            corners = row['n_corners']

            team_data_json += f"""
    '{sanitized}': {{
        name: '{team_name.replace(" Football Club", "")}',
        logo: '{team_logos.get(team_name, '')}',
        animation: '{team_animations.get(team_name, '')}',
        cti: {cti_val:.4f},
        shot_prob: {y1_val:.3f},
        counter_prob: {y3_val:.3f},
        delta_xt: {y5_val:.4f},
        corners: {corners}
    }},"""
    team_data_json += "\n}"

    # Generate team table rows
    team_table_rows = ""
    for idx, row in enumerate(team_top_feature_df.iter_rows(named=True)):
        team_name = row['team']
        n_corners = row['n_corners']
        top_feature_id = row['top_feature_id']
        top_weight = row['top_weight']
        xt_total = row['xt_total']
        xt_avg = row['xt_avg']

        logo_html = ""
        if team_name in team_logos:
            logo_html = f'<img src="{team_logos[team_name]}" alt="{team_name}">'

        display_name = team_name.replace(" Football Club", "").replace(" and Hove Albion", "")

        team_table_rows += f"""
                    <tr>
                        <td class="number-cell" style="text-align: center;">{idx + 1}</td>
                        <td>
                            <div class="team-cell">
                                {logo_html}
                                <span style="font-weight: 600;">{display_name}</span>
                            </div>
                        </td>
                        <td class="number-cell" style="text-align: center;">{n_corners}</td>
                        <td class="number-cell highlight-cell" style="text-align: center;">#{int(top_feature_id)}</td>
                        <td class="number-cell" style="text-align: center;">{top_weight:.4f}</td>
                        <td class="number-cell" style="text-align: center;">{xt_total:.2f}</td>
                        <td class="number-cell" style="text-align: center;">{xt_avg:.4f}</td>
                    </tr>"""

    visuals_cards = ""
    if cti_table_v2_img:
        visuals_cards += f"""
                <div class="plot-container">
                    <h3 style="margin-bottom: 12px;">Team CTI Ranking (v2)</h3>
                    <img src="{cti_table_v2_img}" alt="Team CTI Table v2"
                        style="width: 100%; border: 1px solid #ccc; border-radius: 6px;">
                    <p style="margin-top: 10px; color: #555; font-size: 0.95em;">
                        Goal-weighted CTI table with shot rate, counter risk, and xT volume (corrected).
                    </p>
                </div>"""

    if offense_counter_img:
        visuals_cards += f"""
                <div class="plot-container">
                    <h3 style="margin-bottom: 12px;">Offense vs Counter Risk (Badged)</h3>
                    <img src="{offense_counter_img}" alt="Offense vs Counter Risk Quadrants"
                        style="width: 100%; border: 1px solid #ccc; border-radius: 6px;">
                    <p style="margin-top: 10px; color: #555; font-size: 0.95em;">
                        Quadrant view with club badges showing offensive threat vs counter-attack exposure.
                    </p>
                </div>"""

    # Executive visuals removed to avoid duplication with detailed sections
    visuals_block = ""

    # Create HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            background: #ffffff;
            color: #1a1a1a;
            line-height: 1.6;
        }}

        .header {{
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            border-bottom: 4px solid #000;
        }}

        .header h1 {{
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
            letter-spacing: -0.5px;
        }}

        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
            font-weight: 300;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 20px;
        }}

        .section {{
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}

        .section-title {{
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 3px solid #1a1a1a;
            color: #1a1a1a;
        }}

        .variable-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}

        .variable-card {{
            background: #fafafa;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            padding: 20px;
            transition: all 0.3s ease;
        }}

        .variable-card:hover {{
            border-color: #333;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .variable-card h3 {{
            font-size: 1.3em;
            margin-bottom: 10px;
            color: #1a1a1a;
            font-weight: 600;
        }}

        .variable-card .formula {{
            background: white;
            border: 1px solid #d0d0d0;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            margin: 10px 0;
            color: #333;
        }}

        .variable-card .description {{
            color: #555;
            font-size: 0.95em;
            line-height: 1.5;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}

        .metric-box {{
            background: #f5f5f5;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            padding: 15px;
            text-align: center;
        }}

        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 8px;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}

        .metric-value {{
            font-size: 1.8em;
            font-weight: 700;
            color: #1a1a1a;
            font-family: 'Courier New', monospace;
        }}

        .metric-value.good {{ color: #2d7a2d; }}
        .metric-value.warning {{ color: #cc8800; }}
        .metric-value.bad {{ color: #cc0000; }}

        .plot-container {{
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            padding: 20px;
            margin: 20px 0;
        }}

        .info-box {{
            background: #f8f8f8;
            border-left: 5px solid #1a1a1a;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}

        .info-box h4 {{
            font-size: 1.2em;
            margin-bottom: 10px;
            font-weight: 600;
            color: #1a1a1a;
        }}

        .info-box p, .info-box li {{
            color: #444;
            margin: 8px 0;
        }}

        .team-selector {{
            background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%);
            border: 3px solid #333;
            border-radius: 8px;
            padding: 30px;
            margin: 30px 0;
        }}

        .team-selector h3 {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #1a1a1a;
            font-weight: 700;
        }}

        .team-selector select {{
            width: 100%;
            padding: 15px;
            font-size: 1.1em;
            border: 2px solid #333;
            border-radius: 6px;
            background: white;
            color: #1a1a1a;
            font-weight: 500;
            cursor: pointer;
        }}

        .team-display {{
            margin-top: 30px;
            padding: 20px;
            background: white;
            border: 2px solid #333;
            border-radius: 6px;
            display: none;
        }}

        .team-display.active {{
            display: block;
        }}

        .team-header {{
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }}

        .team-header img {{
            width: 80px;
            height: 80px;
            object-fit: contain;
        }}

        .team-header h4 {{
            font-size: 1.8em;
            font-weight: 700;
            color: #1a1a1a;
        }}

        .team-animation {{
            text-align: center;
            margin: 20px 0;
        }}

        .team-animation img {{
            max-width: 100%;
            border: 2px solid #333;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        .team-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}

        .team-stat {{
            background: #fafafa;
            border: 2px solid #e0e0e0;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}

        .team-stat-label {{
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
            text-transform: uppercase;
            font-weight: 600;
        }}

        .team-stat-value {{
            font-size: 1.5em;
            font-weight: 700;
            color: #1a1a1a;
        }}

        .logo-grid {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 15px;
            margin: 20px 0;
        }}

        .logo-grid img {{
            width: 60px;
            height: 60px;
            object-fit: contain;
            filter: grayscale(30%);
            transition: all 0.3s ease;
        }}

        .logo-grid img:hover {{
            filter: grayscale(0%);
            transform: scale(1.1);
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            overflow: hidden;
        }}

        .data-table thead {{
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            color: white;
        }}

        .data-table th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
            border-bottom: 3px solid #000;
        }}

        .data-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
            color: #333;
        }}

        .data-table tbody tr:hover {{
            background: #f0f0f0;
        }}

        /* Match hover color with tactical patterns table */
        .cti-table tbody tr:hover {{
            background: #f0f0f0;
        }}

        .data-table tbody tr:last-child td {{
            border-bottom: none;
        }}

        .data-table .team-cell {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .data-table .team-cell img {{
            width: 30px;
            height: 30px;
            object-fit: contain;
        }}

        .data-table .number-cell {{
            font-family: 'Courier New', monospace;
            font-weight: 600;
        }}

        .data-table .highlight-cell {{
            background: #f0f0f0;
            font-weight: 700;
        }}

        footer {{
            text-align: center;
            padding: 30px 20px;
            border-top: 3px solid #1a1a1a;
            margin-top: 50px;
            background: #f8f8f8;
        }}

        footer p {{
            color: #666;
            margin: 5px 0;
        }}

        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}

        /* Team Corner Performance Analysis Styles */
        .team-row {{
            transition: all 0.3s ease;
        }}

        .team-row:hover {{
            background: #f0f0f0 !important;
            transform: scale(1.01);
        }}

        .team-logo-container {{
            position: relative;
            display: inline-block;
            width: 40px;
            height: 40px;
        }}

        .team-logo {{
            width: 40px;
            height: 40px;
            object-fit: contain;
            transition: transform 0.3s ease;
            cursor: pointer;
        }}

        .team-logo:hover {{
            transform: scale(1.3);
        }}

        .tooltip-content {{
            visibility: hidden;
            opacity: 0;
            position: absolute;
            z-index: 1000;
            background: #1a1a1a;
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.9em;
            white-space: nowrap;
            left: 50px;
            top: 50%;
            transform: translateY(-50%);
            border: 2px solid #4ade80;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
            transition: opacity 0.3s ease, visibility 0.3s ease;
        }}

        .team-logo-container:hover .tooltip-content {{
            visibility: visible;
            opacity: 1;
        }}

        .cti-badge-excellent {{
            background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
            color: #000;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 1.1em;
            box-shadow: 0 2px 6px rgba(74, 222, 128, 0.4);
        }}

        .cti-badge-good {{
            background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 1.1em;
            box-shadow: 0 2px 6px rgba(96, 165, 250, 0.4);
        }}

        .cti-badge-average {{
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
            color: #000;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 1.1em;
            box-shadow: 0 2px 6px rgba(251, 191, 36, 0.4);
        }}

        .cti-badge-poor {{
            background: linear-gradient(135deg, #f87171 0%, #ef4444 100%);
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 1.1em;
            box-shadow: 0 2px 6px rgba(248, 113, 113, 0.4);
        }}

        .scatter-plot-container {{
            margin: 30px 0;
            text-align: center;
        }}

        .scatter-plot-container img {{
            max-width: 100%;
            border: 2px solid #333;
            border-radius: 8px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }}

        @media (max-width: 768px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
            .variable-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Comprehensive Reliability & Performance Analysis</p>
    </div>

    <div class="container">
        {visuals_block}

        <!-- Model Variables Explanation -->
        <div class="section">
            <h2 class="section-title">Understanding the Model Variables</h2>
            <p style="margin-bottom: 30px; font-size: 1.05em; color: #444;">
                The Corner Threat Index (CTI) captures multiple aspects of corner kick danger.
                Each component measures a different dimension of set-piece effectiveness.
            </p>

            <div class="variable-grid">
                <div class="variable-card">
                    <h3>y₁: Shot Probability</h3>
                    <div class="formula">P(Shot within 10 seconds)</div>
                    <div class="description">
                        Predicts whether the attacking team will take a shot within 10 seconds of the corner kick.
                        Binary outcome (yes/no) converted to a probability. Higher values indicate more
                        immediate offensive threat.
                    </div>
                </div>

                <div class="variable-card">
                    <h3>y₂: Expected Goals (xG)</h3>
                    <div class="formula">Max xG gain in sequence</div>
                    <div class="description">
                        Estimates shot quality using Expected Goals (xG) when a shot occurs. This continuous value
                        represents the likelihood that the shot will result in a goal. Tap-ins from 5 yards have
                        high xG, while long-range efforts have low xG.
                    </div>
                </div>

                <div class="variable-card">
                    <h3>y₃: Counter-Attack Risk</h3>
                    <div class="formula">P(Counter-attack shot by opponent)</div>
                    <div class="description">
                        Tracks the probability that the defending team will launch a counter-attack and take a shot.
                        Captures the defensive vulnerability that comes with committing players forward
                        for corner kicks.
                    </div>
                </div>

                <div class="variable-card">
                    <h3>y₄: Counter xG</h3>
                    <div class="formula">xG of counter-attack if it occurs</div>
                    <div class="description">
                        Measures the quality of the opponent's scoring chance when a counter-attack happens.
                        Fast breaks often lead to high-quality chances, so this risk must be factored into
                        the overall corner assessment.
                    </div>
                </div>

                <div class="variable-card">
                    <h3>y₅: Territorial Gain (ΔxT)</h3>
                    <div class="formula">Expected Threat change</div>
                    <div class="description">
                        Uses Expected Threat (xT) grids to measure how much the attacking team improves their
                        field position. Even without a shot, advancing the ball into dangerous areas creates
                        value. Positive values indicate successful territory gain.
                    </div>
                </div>

                <div class="variable-card" style="grid-column: span 2;">
                    <h3>CTI: Corner Threat Index</h3>
                    <div class="formula">CTI = y₁ × y₂ - 0.5 × y₃ × y₄ + 1.0 × y₅</div>
                    <div class="description">
                        Combines all components into a single metric representing the overall value of a corner.
                        The formula rewards offensive threat (y₁ × y₂) and territorial gains (y₅), while penalizing
                        counter-attack risk (y₃ × y₄). The coefficients (0.5 and 1.0) are tuned weights to
                        balance these competing factors. Higher CTI = more dangerous corner.
                    </div>
                </div>
            </div>
        </div>

        <!-- Overall Model Performance -->
        <div class="section">
            <h2 class="section-title">Overall Model Performance</h2>

            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">Total Corners</div>
                    <div class="metric-value">{len(y1_pred):,}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Teams Analyzed</div>
                    <div class="metric-value">{len(team_cti_df)}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">y₁ ECE</div>
                    <div class="metric-value {'good' if y1_metrics['ece'] < 0.05 else 'warning' if y1_metrics['ece'] < 0.10 else 'bad'}">{y1_metrics['ece']:.4f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">y₃ ECE</div>
                    <div class="metric-value {'good' if y3_metrics['ece'] < 0.05 else 'warning' if y3_metrics['ece'] < 0.10 else 'bad'}">{y3_metrics['ece']:.4f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">CTI Correlation</div>
                    <div class="metric-value {'good' if abs(cti_metrics['correlation']) > 0.7 else 'warning' if abs(cti_metrics['correlation']) > 0.5 else 'bad'}">{cti_metrics['correlation']:.3f}</div>
                </div>
            </div>

            <div class="info-box">
                <h4>Interpreting the Metrics</h4>
                <ul>
                    <li><strong>Expected Calibration Error (ECE)</strong>: Primary reliability metric. Values below 0.05 are excellent,
                    below 0.10 are acceptable. Shows how well probability estimates match reality.</li>
                    <li><strong>Correlation</strong>: Measures how well predicted values align with actual outcomes. Values above 0.7
                    indicate strong predictive power, while values above 0.5 are moderate.</li>
                    <li><strong>Coverage</strong>: Analysis covers {len(y1_pred):,} corners across {len(team_cti_df)} teams,
                    providing a robust sample for model evaluation.</li>
                </ul>
            </div>
        </div>

        <!-- Reliability Curves -->
        <div class="section">
            <h2 class="section-title">Calibration & Reliability Analysis</h2>

            <div class="info-box">
                <h4>Understanding Reliability Curves</h4>
                <p>
                    Reliability curves show whether the model's confidence matches reality. The dashed diagonal line
                    represents perfect calibration. If the model predicts 70% probability, that outcome should
                    occur 70% of the time. Deviations from this line indicate over-confidence (above the line) or
                    under-confidence (below the line).
                </p>
            </div>

            <div class="grid-2">
                <div class="plot-container">
                    <h3 style="margin-bottom: 15px; color: #1a1a1a;">y₁: Shot Probability</h3>
                    <div id="plot_y1"></div>
                </div>
                <div class="plot-container">
                    <h3 style="margin-bottom: 15px; color: #1a1a1a;">y₃: Counter-Attack Risk</h3>
                    <div id="plot_y3"></div>
                </div>
            </div>
        </div>

        <!-- Scatter Plots & Correlations -->
        <div class="section">
            <h2 class="section-title">Prediction vs Reality: Scatter Analysis</h2>

            <div class="info-box">
                <h4>Understanding the Scatter Plots</h4>
                <p>
                    Scatter plots show individual predictions (x-axis) against actual outcomes (y-axis).
                    Points close to the diagonal line indicate accurate predictions. The density of points reveals
                    where the model is most confident and where it struggles. These visualizations help identify
                    systematic biases and improvement opportunities.
                </p>
            </div>

            <div class="grid-2">
                <div class="plot-container">
                    <h3 style="margin-bottom: 15px; color: #1a1a1a;">y₂: xG Prediction</h3>
                    <div id="scatter_y2"></div>
                </div>
                <div class="plot-container">
                    <h3 style="margin-bottom: 15px; color: #1a1a1a;">y₅: Territorial Gain (ΔxT)</h3>
                    <div id="scatter_y5"></div>
                </div>
            </div>

            <div class="grid-2">
                <div class="plot-container">
                    <h3 style="margin-bottom: 15px; color: #1a1a1a;">CTI: Overall Index</h3>
                    <div id="scatter_cti"></div>
                </div>
                <div class="plot-container">
                    <h3 style="margin-bottom: 15px; color: #1a1a1a;">Distribution: CTI Values</h3>
                    <div id="hist_cti"></div>
                </div>
            </div>
        </div>

        <!-- Team Selector -->
        <div class="section">
            <h2 class="section-title">Interactive Team Analysis</h2>

            <div class="team-selector">
                <h3>Select a Team to Analyze Their Corner Patterns</h3>
                <select id="teamSelect" onchange="updateTeamDisplay()">
                    <option value="">-- Choose a team --</option>
                    {team_options}
                </select>

                <div id="teamDisplay" class="team-display">
                    <div class="team-header">
                        <img id="teamLogo" src="" alt="Team logo">
                        <div>
                            <h4 id="teamName"></h4>
                            <p id="teamCorners" style="color: #666; font-size: 1.1em;"></p>
                        </div>
                    </div>

                    <div class="team-animation">
                        <img id="teamAnimation" src="" alt="Corner pattern comparison">
                        <p style="margin-top: 10px; color: #666; font-style: italic;">
                            <strong>Side-by-side comparison:</strong><br>
                            <strong>Left:</strong> NMF tactical pattern showing player movement from initial zones (blue dots) to target areas (red arrows).<br>
                            <strong>Right:</strong> Model predictions visualized - green zone shows shot danger, red zone shows counter-attack risk, yellow arrow shows territorial gain.
                        </p>
                    </div>

                    <div class="team-stats">
                        <div class="team-stat">
                            <div class="team-stat-label">CTI Score</div>
                            <div class="team-stat-value" id="teamCTI"></div>
                        </div>
                        <div class="team-stat">
                            <div class="team-stat-label">Shot Prob</div>
                            <div class="team-stat-value" id="teamShotProb"></div>
                        </div>
                        <div class="team-stat">
                            <div class="team-stat-label">Counter Risk</div>
                            <div class="team-stat-value" id="teamCounterProb"></div>
                        </div>
                        <div class="team-stat">
                            <div class="team-stat-label">Δx Threat</div>
                            <div class="team-stat-value" id="teamDeltaXT"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Teams Logos -->
        <div class="section">
            <h2 class="section-title">Teams in This Analysis</h2>
            <p style="margin-bottom: 20px; color: #444;">
                This analysis covers all {len(team_logos)} Premier League teams from the 2024/25 season. Each team's
                corner patterns, tactical approaches, and defensive vulnerabilities contribute to the model's
                comprehensive understanding of set-piece dynamics.
            </p>
            <div class="logo-grid">
                {''.join([f'<img src="{logo}" title="{name}" alt="{name}">' for name, logo in sorted(team_logos.items())])}
            </div>
        </div>

        <!-- Team Top Features Table -->
        <div class="section">
            <h2 class="section-title">Team Tactical Patterns - NMF Feature Analysis</h2>

            <div class="info-box">
                <h4>Understanding NMF Features</h4>
                <p>
                    Using Non-negative Matrix Factorization (NMF), 30 distinct tactical patterns have been identified
                    in corner kick execution across all teams. Each team has a dominant pattern (top feature) that
                    characterizes their most common corner strategy. The "Top Weight" shows how strongly a team
                    relies on this particular pattern, while "xT Total" and "xT Avg" measure the territorial
                    threat generated using this approach.
                </p>
            </div>

            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width: 5%;">Rank</th>
                        <th style="width: 30%;">Team</th>
                        <th style="width: 10%;">Corners</th>
                        <th style="width: 15%;">Top Feature ID</th>
                        <th style="width: 15%;">Top Weight</th>
                        <th style="width: 12.5%;">xT Total</th>
                        <th style="width: 12.5%;">xT Avg</th>
                    </tr>
                </thead>
                <tbody>
                    {team_table_rows}
                </tbody>
            </table>

            <div class="info-box">
                <h4>How to Read This Table</h4>
                <ul>
                    <li><strong>Rank</strong>: Teams ordered by total number of corners analyzed</li>
                    <li><strong>Top Feature ID</strong>: The dominant tactical pattern (1-30) used by this team</li>
                    <li><strong>Top Weight</strong>: Strength of association with this pattern (higher = more consistent)</li>
                    <li><strong>xT Total</strong>: Cumulative Expected Threat generated from corners</li>
                    <li><strong>xT Avg</strong>: Average Expected Threat per corner (efficiency metric)</li>
                </ul>
            </div>
        </div>

        <!-- Team Corner Performance Analysis Section -->
        <div class="section">
            <h2 class="section-title">Team Corner Performance Analysis</h2>

            <div class="info-box">
                <h4>CTI Rankings & Metrics</h4>
                <p>
                    This section presents comprehensive team rankings based on Corner Threat Index (CTI) averages.
                    The table includes key performance indicators: shot probability, counter-attack risk, and
                    territorial gain. Hover over team logos to see detailed statistics. CTI badges are color-coded:
                    <strong style="color: #4ade80;">Green (Excellent ≥6.5)</strong>,
                    <strong style="color: #60a5fa;">Blue (Good ≥5.5)</strong>,
                    <strong style="color: #fbbf24;">Orange (Average ≥4.5)</strong>,
                    <strong style="color: #f87171;">Red (Poor <4.5)</strong>.
                </p>
            </div>

            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width: 5%; text-align: center;">Rank</th>
                        <th style="width: 5%; text-align: center;">Logo</th>
                        <th style="width: 25%; text-align: left;">Team</th>
                        <th style="width: 15%; text-align: center;">CTI Avg</th>
                        <th style="width: 12.5%; text-align: center;">Shot Prob</th>
                        <th style="width: 12.5%; text-align: center;">Counter Risk</th>
                        <th style="width: 12.5%; text-align: center;">ΔxT</th>
                        <th style="width: 12.5%; text-align: center;">Corners</th>
                    </tr>
                </thead>
                <tbody>
                    {team_performance_rows}
                </tbody>
            </table>

            <div class="scatter-plot-container">
                <h3 style="margin-bottom: 15px; color: #1a1a1a;">Offense vs Counter Risk Quadrants</h3>
                <img src="{offense_counter_img}" alt="Team Offense vs Counter Risk">
                <p style="margin-top: 15px; color: #555; font-size: 0.95em;">
                    This scatter plot visualizes the trade-off between offensive threat (shot probability) and
                    defensive vulnerability (counter-attack risk). Teams in the top-right quadrant combine high
                    attacking output with elevated counter risk, while bottom-left teams are more conservative.
                    Badge positions reflect each team's tactical approach to corner kicks.
                </p>
            </div>

            <div class="info-box">
                <h4>Understanding the Metrics</h4>
                <ul>
                    <li><strong>CTI Avg</strong>: Average Corner Threat Index across all corners taken by the team</li>
                    <li><strong>Shot Prob</strong>: Probability of generating a shot within 10 seconds of the corner</li>
                    <li><strong>Counter Risk</strong>: Probability of conceding a counter-attack shot opportunity</li>
                    <li><strong>ΔxT</strong>: Average territorial gain measured by Expected Threat change</li>
                    <li><strong>Corners</strong>: Total number of corners analyzed for this team</li>
                </ul>
            </div>
        </div>
    </div>

    <footer>
        <p><strong>CTI Model Reliability & Performance Report</strong></p>
        <p>Generated by: Tiago | Premier League 2024/25 Season</p>
        <p style="margin-top: 10px; font-size: 0.9em;">
            This report provides a comprehensive evaluation of model performance across all variables.
            For technical details, please refer to the model documentation.
        </p>
    </footer>

    <script>
        // Team data
        const teamData = {team_data_json};

        // Update team display
        function updateTeamDisplay() {{
            const select = document.getElementById('teamSelect');
            const display = document.getElementById('teamDisplay');
            const teamId = select.value;

            if (!teamId || !teamData[teamId]) {{
                display.classList.remove('active');
                return;
            }}

            const team = teamData[teamId];

            document.getElementById('teamLogo').src = team.logo;
            document.getElementById('teamName').textContent = team.name;
            document.getElementById('teamCorners').textContent = `${{team.corners}} corners analyzed`;
            document.getElementById('teamAnimation').src = team.animation || '';
            document.getElementById('teamCTI').textContent = team.cti.toFixed(4);
            document.getElementById('teamShotProb').textContent = (team.shot_prob * 100).toFixed(1) + '%';
            document.getElementById('teamCounterProb').textContent = (team.counter_prob * 100).toFixed(1) + '%';
            document.getElementById('teamDeltaXT').textContent = team.delta_xt.toFixed(4);

            display.classList.add('active');
        }}

        // Plotly configuration
        const plotConfig = {{responsive: true, displayModeBar: false}};
        const layoutCommon = {{
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white',
            font: {{family: 'system-ui', color: '#1a1a1a'}},
            showlegend: true,
            hovermode: 'closest'
        }};

        // Reliability curves
        const perfectLine = {{
            x: [0, 1], y: [0, 1],
            mode: 'lines',
            name: 'Perfect Calibration',
            line: {{color: '#999', width: 2, dash: 'dash'}}
        }};

        // y1 reliability
        const y1Calib = {{
            x: {y1_metrics['prob_pred']},
            y: {y1_metrics['prob_true']},
            mode: 'lines+markers',
            name: 'Model',
            line: {{color: '#1a1a1a', width: 3}},
            marker: {{size: 10, color: '#1a1a1a'}}
        }};

        Plotly.newPlot('plot_y1', [perfectLine, y1Calib], {{
            ...layoutCommon,
            xaxis: {{title: 'Predicted Probability', range: [0, 1]}},
            yaxis: {{title: 'Observed Frequency', range: [0, 1]}}
        }}, plotConfig);

        // y3 reliability
        const y3Calib = {{
            x: {y3_metrics['prob_pred']},
            y: {y3_metrics['prob_true']},
            mode: 'lines+markers',
            name: 'Model',
            line: {{color: '#1a1a1a', width: 3}},
            marker: {{size: 10, color: '#1a1a1a'}}
        }};

        Plotly.newPlot('plot_y3', [perfectLine, y3Calib], {{
            ...layoutCommon,
            xaxis: {{title: 'Predicted Probability', range: [0, 1]}},
            yaxis: {{title: 'Observed Frequency', range: [0, 1]}}
        }}, plotConfig);

        // Scatter plots
        const scatterY2 = {{
            x: {list(y2_pred)},
            y: {list(y2_true)},
            mode: 'markers',
            type: 'scatter',
            marker: {{
                size: 4,
                color: '#333',
                opacity: 0.5
            }},
            name: 'Predictions'
        }};

        const perfectLineY2 = {{
            x: [Math.min(...{list(y2_pred)}), Math.max(...{list(y2_pred)})],
            y: [Math.min(...{list(y2_pred)}), Math.max(...{list(y2_pred)})],
            mode: 'lines',
            line: {{color: '#999', width: 2, dash: 'dash'}},
            name: 'Perfect'
        }};

        Plotly.newPlot('scatter_y2', [perfectLineY2, scatterY2], {{
            ...layoutCommon,
            xaxis: {{title: 'Predicted xG'}},
            yaxis: {{title: 'Actual xG'}}
        }}, plotConfig);

        // y5 scatter
        const scatterY5 = {{
            x: {list(y5_pred)},
            y: {list(y5_true)},
            mode: 'markers',
            type: 'scatter',
            marker: {{size: 4, color: '#333', opacity: 0.5}},
            name: 'Predictions'
        }};

        const perfectLineY5 = {{
            x: [Math.min(...{list(y5_pred)}), Math.max(...{list(y5_pred)})],
            y: [Math.min(...{list(y5_pred)}), Math.max(...{list(y5_pred)})],
            mode: 'lines',
            line: {{color: '#999', width: 2, dash: 'dash'}},
            name: 'Perfect'
        }};

        Plotly.newPlot('scatter_y5', [perfectLineY5, scatterY5], {{
            ...layoutCommon,
            xaxis: {{title: 'Predicted ΔxT'}},
            yaxis: {{title: 'Actual ΔxT'}}
        }}, plotConfig);

        // CTI scatter
        const scatterCTI = {{
            x: {list(cti_pred)},
            y: {list(cti_true)},
            mode: 'markers',
            type: 'scatter',
            marker: {{size: 4, color: '#1a1a1a', opacity: 0.6}},
            name: 'Predictions'
        }};

        const perfectLineCTI = {{
            x: [Math.min(...{list(cti_pred)}), Math.max(...{list(cti_pred)})],
            y: [Math.min(...{list(cti_pred)}), Math.max(...{list(cti_pred)})],
            mode: 'lines',
            line: {{color: '#999', width: 2, dash: 'dash'}},
            name: 'Perfect'
        }};

        Plotly.newPlot('scatter_cti', [perfectLineCTI, scatterCTI], {{
            ...layoutCommon,
            xaxis: {{title: 'Predicted CTI'}},
            yaxis: {{title: 'Actual CTI'}}
        }}, plotConfig);

        // CTI histogram
        const histCTI = {{
            x: {list(cti_pred)},
            type: 'histogram',
            marker: {{color: '#1a1a1a'}},
            name: 'CTI Distribution',
            nbinsx: 50
        }};

        Plotly.newPlot('hist_cti', [histCTI], {{
            ...layoutCommon,
            xaxis: {{title: 'CTI Value'}},
            yaxis: {{title: 'Frequency'}}
        }}, plotConfig);
    </script>
</body>
</html>
"""

    # Write HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding='utf-8')
    print(f"\n[OK] Saved enhanced reliability report: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    """
    Main execution function to generate the enhanced reliability report.

    Loads predictions and models, computes metrics, and produces the HTML output.
    """
    from cti_paths import FINAL_PROJECT_DIR
    from cti_gmm_zones import load_zone_models

    # Load data
    data_dir = FINAL_PROJECT_DIR / 'cti_data'
    pred_path = data_dir / 'predictions.csv'
    if not pred_path.exists():
        print(f"Warning: predictions.csv not found at {pred_path}. Skipping reliability report generation.")
        return

    predictions_df = pl.read_csv(pred_path)
    
    detailed_path = data_dir / 'team_cti_detailed.csv'
    if not detailed_path.exists():
         print(f"Warning: team_cti_detailed.csv not found at {detailed_path}. Skipping reliability report generation.")
         return
         
    team_cti_df = pl.read_csv(detailed_path)
    
    # Use goal-weighted CTI as the primary CTI metric if available (to match summary/v2)
    if "cti_goal_weighted" in team_cti_df.columns:
        print("Using cti_goal_weighted as cti_avg for consistency.")
        team_cti_df = team_cti_df.with_columns(pl.col("cti_goal_weighted").alias("cti_avg"))

    team_top_feature_df = pl.read_csv(data_dir / 'team_top_feature.csv')

    # Load models for animations
    try:
        with open(data_dir / 'nmf_model.pkl', 'rb') as f:
            nmf_model = pickle.load(f)
        zone_models = load_zone_models(data_dir / 'gmm_zones.pkl')
    except Exception as e:
        print(f"Warning: Could not load models for animations: {e}")
        nmf_model = None
        zone_models = None

    # Generate report
    output_path = FINAL_PROJECT_DIR / 'cti_outputs' / 'reliability_report.html'
    create_enhanced_reliability_html(
        predictions_df,
        team_cti_df,
        team_top_feature_df,
        output_path,
        nmf_model,
        zone_models
    )

    print(f"\n[DONE] Open in browser: {output_path}")


if __name__ == '__main__':
    main()
