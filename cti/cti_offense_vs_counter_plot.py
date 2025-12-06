"""
Author: Tiago
Date: 2025-12-04
Description: Generate the offense vs counter scatter plot using goal-weighted CTI values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from pathlib import Path

BACKGROUND = "#F7F5F0"
AXIS_GREEN = "#2ecc71"

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "cti_data"
OUT_DIR = ROOT / "cti_outputs"
ASSETS_DIR = ROOT / "assets"


def sanitize(name: str) -> str:
    """
    Sanitize a team name for logo filenames.

    :param name: Team name string.
    :return: Lowercase alphanumeric string suitable for filenames.
    """
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def generate_plot() -> None:
    """
    Build and save the offense vs counter scatter plot from team_cti_summary.csv.
    Uses CTI Average for offensive threat and counter_risk on X.
    Output: cti_outputs/team_offense_vs_counter_presentation.png.
    """
    # Load data
    summary_path = DATA_DIR / "team_cti_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"team_cti_summary.csv not found at {summary_path}")

    team_df = pd.read_csv(summary_path)

    # Map columns
    team_df["offensive_threat"] = team_df["cti_avg"]
    team_df["counter_attack_risk"] = team_df["counter_risk"]

    # Figure & axes styling
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlabel("Counter-Attack Risk (Low)", fontsize=14, fontweight="bold", color=AXIS_GREEN)
    ax.set_ylabel("Corner Threat Index - CTI (High)", fontsize=14, fontweight="bold", color=AXIS_GREEN)
    fig.suptitle("Team Corner Performance", fontsize=18, fontweight="bold", y=0.97, color=AXIS_GREEN)
    fig.text(0.5, 0.92, "Offensive Threat vs Counter-Attack Risk",
             fontsize=14, ha="center", color="#4a4a4a", style="italic")

    # Quadrant guides
    median_x = team_df["counter_attack_risk"].median()
    median_y = team_df["offensive_threat"].median()
    ax.axhline(y=median_y, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)
    ax.axvline(x=median_x, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)

    # Quadrant labels
    ax.text(0.05, 0.95, "High Threat, Low Risk", transform=ax.transAxes,
            fontsize=12, color="gray", alpha=0.6, ha="left", va="top", fontweight="bold")
    ax.text(0.95, 0.95, "High Threat, High Risk", transform=ax.transAxes,
            fontsize=12, color="gray", alpha=0.6, ha="right", va="top", fontweight="bold")
    ax.text(0.05, 0.05, "Low Threat, Low Risk", transform=ax.transAxes,
            fontsize=12, color="gray", alpha=0.6, ha="left", va="bottom", fontweight="bold")
    ax.text(0.95, 0.05, "Low Threat, High Risk", transform=ax.transAxes,
            fontsize=12, color="gray", alpha=0.6, ha="right", va="bottom", fontweight="bold")

    # Collision avoidance
    def repel_points(x, y, labels, iterations=50, repulsion=0.02, attraction=0.01):
        """
        Simple force-directed placement to separate overlapping points.
        Works in normalized 0-1 coordinates.
        """
        n = len(x)
        # Normalize
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0
        
        # Current positions (normalized)
        cx = (x - x_min) / x_range
        cy = (y - y_min) / y_range
        
        # Original positions (normalized)
        ox = cx.copy()
        oy = cy.copy()
        
        for _ in range(iterations):
            fx = np.zeros(n)
            fy = np.zeros(n)
            
            # Repulsion between pairs
            for i in range(n):
                for j in range(i + 1, n):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist_sq = dx*dx + dy*dy
                    
                    # Minimum distance threshold
                    min_dist = 0.06  # Adjust based on badge size/plot size
                    
                    if dist_sq < min_dist*min_dist:
                        dist = np.sqrt(dist_sq)
                        if dist < 1e-6: dist = 1e-6
                        force = (min_dist - dist) / dist * repulsion
                        fx[i] += dx * force
                        fy[i] += dy * force
                        fx[j] -= dx * force
                        fy[j] -= dy * force
            
            # Attraction to original position
            fx -= (cx - ox) * attraction
            fy -= (cy - oy) * attraction
            
            # Update
            cx += fx
            cy += fy
            
            # Clamp coordinates
            cx = np.clip(cx, -0.1, 1.1)
            cy = np.clip(cy, -0.1, 1.1)
            
        # Denormalize
        return cx * x_range + x_min, cy * y_range + y_min

    # Axes limits and styling
    x_data_min = team_df["counter_attack_risk"].min()
    x_data_max = team_df["counter_attack_risk"].max()
    y_data_min = team_df["offensive_threat"].min()
    y_data_max = team_df["offensive_threat"].max()

    x_range = x_data_max - x_data_min
    y_range = y_data_max - y_data_min

    # Axis limits with small padding (ensure padding > 0 to avoid flat limits)
    pad_x = x_range * 0.05 if x_range > 0 else 0.01
    pad_y = y_range * 0.05 if y_range > 0 else 0.01
    x_lim = (x_data_min - pad_x, x_data_max + pad_x)
    y_lim = (y_data_min - pad_y, y_data_max + pad_y)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    # Balanced diagonal line anchored to the axes limits (touches both axes)
    ax.plot([x_lim[0], x_lim[1]], [y_lim[0], y_lim[1]], "--", color=AXIS_GREEN, linewidth=2, alpha=0.3, zorder=0)

    # Prepare data for repulsion
    teams = team_df["team"].values
    orig_x = team_df["counter_attack_risk"].values
    orig_y = team_df["offensive_threat"].values
    
    # Run repulsion
    new_x, new_y = repel_points(orig_x, orig_y, teams)

    # Plot badges at new positions with connecting lines
    for i, team_name in enumerate(teams):
        x0, y0 = orig_x[i], orig_y[i]
        x1, y1 = new_x[i], new_y[i]
        
        # Draw line if moved significantly
        if np.hypot(x1-x0, y1-y0) > (x_data_max - x_data_min) * 0.01:
            ax.plot([x0, x1], [y0, y1], '-', color='gray', alpha=0.3, linewidth=1, zorder=1)
            ax.plot(x0, y0, 'o', color='gray', markersize=3, alpha=0.5, zorder=1)

        logo_path = ASSETS_DIR / f"{sanitize(team_name)}.png"

        if logo_path.exists():
            try:
                img = Image.open(logo_path).convert("RGBA")
                img.thumbnail((45, 45), Image.Resampling.LANCZOS)
                imagebox = OffsetImage(np.asarray(img), zoom=1.0)
                ax.add_artist(AnnotationBbox(imagebox, (x1, y1), frameon=False, zorder=10))
            except Exception as exc:
                print(f"Warning: Could not load logo for {team_name}: {exc}")
                ax.plot(x1, y1, "o", markersize=10, alpha=0.7)
        else:
            print(f"Warning: Logo not found for {team_name}")
            ax.plot(x1, y1, "o", markersize=10, alpha=0.7)

    ax.set_xlim(x_data_min - x_range * 0.05, x_data_max + x_range * 0.05)
    ax.set_ylim(y_data_min - y_range * 0.05, y_data_max + y_range * 0.05)

    ax.grid(False)
    for spine_name, spine in ax.spines.items():
        if spine_name in ("bottom", "left"):
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_color(AXIS_GREEN)
        else:
            spine.set_visible(False)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False, colors=AXIS_GREEN)

    plt.tight_layout()
    out_path = OUT_DIR / "team_offense_vs_counter_presentation.png"
    plt.savefig(out_path, dpi=300, facecolor=BACKGROUND, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    generate_plot()
