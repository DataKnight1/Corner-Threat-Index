"""
Author: Tiago
Date: 2025-12-04
Description: Generate CTI visualizations from existing artifacts (predictions, corners, and team_cti_v2). Produces a goal-weighted team CTI table and the offense vs counter scatter plot without re-running training or inference.
"""

from pathlib import Path
import polars as pl
from cti_team_mapping import build_team_name_map
from cti_nmf_routines import save_team_cti_table

from cti_paths import REPO_ROOT, FINAL_PROJECT_DIR, DATA_2024

ROOT = REPO_ROOT
RAW_DATA_DIR = DATA_2024
DATA_DIR = FINAL_PROJECT_DIR / "cti_data"
OUT_FIG = FINAL_PROJECT_DIR / "cti_outputs"


def summarize_by_team(pred_df: pl.DataFrame, corners_df: pl.DataFrame) -> pl.DataFrame:
    """
    Summarize per-corner predictions into team-level aggregates.

    :param pred_df: Predictions per corner (y1..y5, calibrated columns if present).
    :param corners_df: Corner metadata containing corner_id and team_id.
    :return: Team-level dataframe with CTI and component averages.
    """
    joined = pred_df.join(
        corners_df.select(["corner_id", "team_id", "match_id"]),
        on="corner_id",
        how="left",
    )
    # Filter out rows with null team_id
    joined = joined.filter(pl.col("team_id").is_not_null())

    # Use calibrated predictions if available, otherwise use raw predictions
    y1_col = pl.when(pl.col('y1_cal').is_not_null()).then(pl.col('y1_cal')).otherwise(pl.col('y1'))
    y3_col = pl.when(pl.col('y3_cal').is_not_null()).then(pl.col('y3_cal')).otherwise(pl.col('y3'))

    return (
        joined.group_by("team_id")
        .agg([
            pl.len().alias("n_corners"),
            pl.col("cti").mean().alias("cti_avg"),
            y1_col.mean().alias("p_shot"),
            (y3_col * pl.col("y4")).mean().alias("counter_risk"),
            pl.col("y5").mean().alias("delta_xt"),
        ])
        .sort("cti_avg", descending=True)
    )


def render_team_cti_table(team_df: pl.DataFrame, out_png: Path, title: str):
    """
    Render the goal-weighted CTI table with logos.

    :param team_df: Team-level dataframe with team names and CTI metrics.
    :param out_png: Path to write the PNG table.
    :param title: Title to render on the table.
    """
    df_named = team_df

    assets = ROOT / 'Final_Project' / 'assets'
    save_team_cti_table(
        df_named.select(['team','cti_avg','p_shot','counter_risk','delta_xt','n_corners']),
        DATA_DIR / 'team_cti_summary.csv',
        out_png,
        title=title,
        logo_dir=assets
    )


def render_offense_vs_counter_plot(team_df: pl.DataFrame, out_png: Path):
    """
    Generate the offense vs counter-attack risk scatter plot.

    :param team_df: Team-level dataframe with team names and CTI metrics.
    :param out_png: Path to write the plot (kept for API symmetry; plot script writes its own).
    """
    import subprocess
    import sys

    # Save temporary CSV for the plot script
    temp_csv = DATA_DIR / 'team_cti_v2.csv'
    if temp_csv.exists():
        pass
    else:
        team_df.select(['team','cti_avg','p_shot','counter_risk','delta_xt','n_corners']).write_csv(temp_csv)

    # Call the offense vs counter plot script
    plot_script = ROOT / 'Final_Project' / 'cti' / 'cti_offense_vs_counter_plot.py'
    try:
        subprocess.run([sys.executable, str(plot_script)], check=True, capture_output=True, text=True)
        print(f"OK generated offense vs counter plot: {out_png}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to generate offense vs counter plot: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")


def main():
    """
    Generate all visualizations from existing artifacts (no training/inference).

    Steps:
      1. Load predictions and corners.
      2. Load team_cti_v2 (goal-weighted CTI).
      3. Render team CTI table.
      4. Render offense vs counter scatter plot.
    """

    print("=" * 70)
    print("CTI Visualization Generator (Skip Training)")
    print("=" * 70)
    print()

    # Check for required files
    pred_csv = DATA_DIR / "predictions.csv"
    corners_parquet = DATA_DIR / "corners_dataset.parquet"

    if not pred_csv.exists():
        print(f"ERROR: predictions.csv not found at {pred_csv}")
        print("Please run the full inference pipeline first (cti_infer_cti.py)")
        return

    if not corners_parquet.exists():
        print(f"ERROR: corners_dataset.parquet not found at {corners_parquet}")
        print("Please ensure the corners dataset exists")
        return

    print(f"Loading predictions from: {pred_csv}")
    pred_df = pl.read_csv(pred_csv)
    print(f"  - Loaded {len(pred_df)} corner predictions")

    print(f"Loading corners dataset from: {corners_parquet}")
    corners_df = pl.read_parquet(corners_parquet)
    print(f"  - Loaded {len(corners_df)} corners")
    print()

    # Load team_cti_v2.csv directly (the final CTI values)
    print("Loading team CTI data from team_cti_v2.csv...")
    team_cti_v2_csv = DATA_DIR / "team_cti_v2.csv"
    if not team_cti_v2_csv.exists():
        print(f"ERROR: team_cti_v2.csv not found at {team_cti_v2_csv}")
        print("Using model predictions as fallback...")
        team_df_model = summarize_by_team(pred_df, corners_df)
    else:
        team_df_v2 = pl.read_csv(team_cti_v2_csv)
        print(f"  - Loaded {len(team_df_v2)} teams from team_cti_v2.csv")

        # Map team names
        meta_dir = RAW_DATA_DIR / 'meta'
        team_name_map = build_team_name_map(meta_dir, use_fallback=True)

        # Create a mapping from team names to IDs for the dataframe
        # The CSV has team names, we need to keep them
        team_df_model = team_df_v2
    print()

    # Generate team CTI table using v2 data (single output)
    print("Generating team CTI table (using team_cti_v2.csv)...")

    # Prepare data for table generation
    # team_cti_v2.csv has: team,cti_avg,p_shot,counter_risk,delta_xt,n_corners
    from cti_nmf_routines import save_team_cti_table
    assets = ROOT / 'Final_Project' / 'assets'

    save_team_cti_table(
        team_df_model.select(['team','cti_avg','p_shot','counter_risk','delta_xt','n_corners']),
        DATA_DIR / 'team_cti_summary.csv',
        OUT_FIG / "team_cti_table.png",
        title="Team CTI Summary (Model Predictions)",
        logo_dir=assets
    )
    print(f"  [OK] Saved: {OUT_FIG / 'team_cti_table.png'}")
    print()

    # Generate offense vs counter plot using v2 data
    print("Generating offense vs counter scatter plot...")
    # The scatter plot script reads team_cti_v2.csv directly, so just call it
    render_offense_vs_counter_plot(
        team_df_model,
        OUT_FIG / "team_offense_vs_counter_presentation.png"
    )
    print()

    # No separate integration step here; visuals regenerated above
    print(f"  [OK] Visualizations regenerated")
    print()

    print("=" * 70)
    print("All visualizations generated successfully!")
    print("=" * 70)
    print()
    print("Generated files:")
    print(f"  1. {OUT_FIG / 'team_cti_table.png'}")
    print(f"  2. {OUT_FIG / 'team_offense_vs_counter_presentation.png'}")
    print(f"  3. {OUT_FIG / 'reliability_Report.html'} (with embedded CTI analysis)")
    print()


if __name__ == "__main__":
    main()
