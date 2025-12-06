"""
Author: Tiago
Date: 2025-12-04
Description: Generate interactive HTML report for team CTI analysis with hoverable team logos.
"""

from pathlib import Path
import polars as pl
import base64


def generate_team_cti_html(
    team_df: pl.DataFrame,
    output_path: Path,
    table_png_path: Path,
    scatter_png_path: Path,
    assets_dir: Path
) -> None:
    """
    Generate an interactive HTML report with:
    - Interactive table with hoverable team logos showing tooltips
    - Scatter plot visualization

    Args:
        team_df: DataFrame with columns: team, cti_avg, p_shot, counter_risk, delta_xt, n_corners
        output_path: Path to save HTML file
        table_png_path: Path to the team CTI table PNG
        scatter_png_path: Path to the offense vs counter scatter plot PNG
        assets_dir: Directory containing team logo PNGs
    """

    def sanitize(name: str) -> str:
        return ''.join(ch.lower() for ch in str(name) if ch.isalnum())

    def image_to_base64(img_path: Path) -> str:
        """Convert image to base64 for embedding"""
        with open(img_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    # Sort by CTI descending
    team_df = team_df.sort("cti_avg", descending=True)

    # Generate table rows HTML
    table_rows_html = []
    for i, row in enumerate(team_df.iter_rows(named=True), start=1):
        team_name = row['team']
        cti_avg = row['cti_avg']
        p_shot = row['p_shot']
        counter_risk = row['counter_risk']
        delta_xt = row['delta_xt']
        n_corners = row.get('n_corners', 0)

        # Try to load team logo
        logo_path = assets_dir / f"{sanitize(team_name)}.png"
        if logo_path.exists():
            try:
                logo_b64 = image_to_base64(logo_path)
                logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="{team_name}" class="team-logo">'
            except:
                logo_html = f'<span class="team-name-text">{team_name}</span>'
        else:
            logo_html = f'<span class="team-name-text">{team_name}</span>'

        # Determine CTI badge color
        if cti_avg >= 0.065:
            badge_class = "cti-badge-high"
        elif cti_avg >= 0.045:
            badge_class = "cti-badge-medium"
        else:
            badge_class = "cti-badge-low"

        tooltip_html = f"""
            <div class="tooltip-content">
                <strong>{team_name}</strong><br>
                <span class="tooltip-label">Rank:</span> #{i}<br>
                <span class="tooltip-label">CTI (avg):</span> {cti_avg:.4f}<br>
                <span class="tooltip-label">P(shot):</span> {p_shot:.3f}<br>
                <span class="tooltip-label">Counter Risk:</span> {counter_risk:.4f}<br>
                <span class="tooltip-label">ΔxT:</span> {delta_xt:.4f}<br>
                <span class="tooltip-label">Corners:</span> {int(n_corners)}
            </div>
        """

        row_html = f"""
        <tr class="team-row">
            <td class="rank-cell">#{i}</td>
            <td class="team-cell">
                <div class="team-logo-container">
                    {logo_html}
                    {tooltip_html}
                </div>
                <span class="team-name">{team_name}</span>
            </td>
            <td class="metric-cell">
                <span class="cti-badge {badge_class}">{cti_avg:.4f}</span>
            </td>
            <td class="metric-cell">{p_shot:.3f}</td>
            <td class="metric-cell">{counter_risk:.4f}</td>
            <td class="metric-cell">{delta_xt:.4f}</td>
        </tr>
        """
        table_rows_html.append(row_html)

    # Embed the scatter plot
    try:
        scatter_b64 = image_to_base64(scatter_png_path)
        scatter_html = f'<img src="data:image/png;base64,{scatter_b64}" alt="Offense vs Counter Risk" class="scatter-plot">'
    except:
        scatter_html = '<p>Scatter plot not available</p>'

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Team Corner Performance - CTI Analysis</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            color: #1a1a1a;
            line-height: 1.6;
        }}

        .header {{
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            border-bottom: 4px solid #229954;
        }}

        .header h1 {{
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
            letter-spacing: -0.5px;
        }}

        .header p {{
            font-size: 1.1em;
            opacity: 0.95;
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
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}

        .section-title {{
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 3px solid #2ecc71;
            color: #1a1a1a;
        }}

        .interactive-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 20px;
        }}

        .interactive-table thead {{
            background: #2ecc71;
            color: white;
        }}

        .interactive-table th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 1.05em;
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        .team-row {{
            border-bottom: 1px solid #e0e0e0;
            transition: all 0.3s ease;
        }}

        .team-row:hover {{
            background: #f8f9fa;
            transform: scale(1.01);
            box-shadow: 0 2px 8px rgba(46, 204, 113, 0.2);
        }}

        .rank-cell {{
            padding: 15px;
            font-weight: 600;
            color: #666;
            width: 60px;
        }}

        .team-cell {{
            padding: 15px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .team-logo-container {{
            position: relative;
            display: inline-block;
        }}

        .team-logo {{
            width: 40px;
            height: 40px;
            object-fit: contain;
            cursor: pointer;
            transition: transform 0.3s ease;
        }}

        .team-logo:hover {{
            transform: scale(1.2);
        }}

        .team-name {{
            font-weight: 500;
            font-size: 1.05em;
        }}

        .team-name-text {{
            display: inline-block;
            width: 40px;
            height: 40px;
            background: #e0e0e0;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8em;
            color: #666;
        }}

        .tooltip-content {{
            visibility: hidden;
            position: absolute;
            z-index: 1000;
            background: #1a1a1a;
            color: white;
            padding: 12px;
            border-radius: 8px;
            font-size: 0.9em;
            line-height: 1.6;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            min-width: 220px;
            left: 50%;
            transform: translateX(-50%);
            bottom: 100%;
            margin-bottom: 8px;
            opacity: 0;
            transition: opacity 0.3s ease, visibility 0.3s ease;
        }}

        .tooltip-content::after {{
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 8px solid transparent;
            border-top-color: #1a1a1a;
        }}

        .team-logo-container:hover .tooltip-content {{
            visibility: visible;
            opacity: 1;
        }}

        .tooltip-label {{
            color: #2ecc71;
            font-weight: 600;
        }}

        .metric-cell {{
            padding: 15px;
            text-align: center;
            font-family: 'Courier New', monospace;
            font-size: 1em;
        }}

        .cti-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 1.05em;
        }}

        .cti-badge-high {{
            background: #2ecc71;
            color: white;
        }}

        .cti-badge-medium {{
            background: #f39c12;
            color: white;
        }}

        .cti-badge-low {{
            background: #e74c3c;
            color: white;
        }}

        .scatter-plot {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-top: 20px;
        }}

        .legend {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .legend-badge {{
            width: 40px;
            height: 24px;
            border-radius: 4px;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Team Corner Performance</h1>
        <p>Interactive CTI Analysis - Premier League 2024</p>
    </div>

    <div class="container">
        <!-- Interactive Table Section -->
        <div class="section">
            <h2 class="section-title">Team Rankings & Metrics</h2>
            <p style="margin-bottom: 20px; color: #666;">
                Hover over team logos for detailed statistics. Click on column headers to sort.
            </p>

            <table class="interactive-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Team</th>
                        <th style="text-align: center;">CTI (avg)</th>
                        <th style="text-align: center;">P(shot)</th>
                        <th style="text-align: center;">Counter Risk</th>
                        <th style="text-align: center;">ΔxT</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows_html)}
                </tbody>
            </table>

            <div class="legend">
                <div class="legend-item">
                    <span class="legend-badge" style="background: #2ecc71;"></span>
                    <span>High CTI (≥ 0.065)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-badge" style="background: #f39c12;"></span>
                    <span>Medium CTI (0.045 - 0.065)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-badge" style="background: #e74c3c;"></span>
                    <span>Low CTI (< 0.045)</span>
                </div>
            </div>
        </div>

        <!-- Scatter Plot Section -->
        <div class="section">
            <h2 class="section-title">Offensive Threat vs Counter-Attack Risk</h2>
            <p style="margin-bottom: 20px; color: #666;">
                Visual analysis of team corner performance positioning: CTI (offensive threat) vs counter-attack risk.
                The green trend line shows the correlation between offensive threat and defensive vulnerability.
            </p>
            {scatter_html}
        </div>
    </div>

    <div class="footer">
        <p>Generated with Corner Threat Index (CTI) Analysis Pipeline</p>
        <p style="margin-top: 5px; font-size: 0.85em;">
            CTI = y₁×y₂ - λ×y₃×y₄ + γ×y₅ | Premier League 2024 Data
        </p>
    </div>
</body>
</html>
"""

    output_path.write_text(html_content, encoding='utf-8')
    print(f"OK Generated interactive HTML: {output_path}")


if __name__ == "__main__":
    print("CTI Team HTML Generator")
    print("Use from cti_infer_cti.py pipeline")
