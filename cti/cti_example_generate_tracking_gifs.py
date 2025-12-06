"""
Author: Tiago
Date: 2025-12-04
Description: Example script showing how to generate animated GIF tracking visualizations for corners.
"""

from pathlib import Path
import polars as pl
import pickle
from cti_tracking_visualization import generate_team_corner_samples
from cti_paths import FINAL_PROJECT_DIR, TRACKING_DIR


def load_tracking_data_for_matches(match_ids: list) -> dict:
    """
    Load tracking data for specified matches.

    Args:
        match_ids: List of match IDs to load

    Returns:
        Dict mapping match_id -> tracking DataFrame
    """
    tracking_dict = {}

    tracking_dir = TRACKING_DIR

    for match_id in match_ids:
        tracking_path = tracking_dir / f'match_{match_id}_tracking.parquet'

        if tracking_path.exists():
            try:
                tracking_df = pl.read_parquet(tracking_path)
                tracking_dict[match_id] = tracking_df
                print(f"  Loaded tracking data for match {match_id}")
            except Exception as e:
                print(f"  Warning: Failed to load tracking for match {match_id}: {e}")
        else:
            print(f"  Warning: Tracking file not found for match {match_id}")

    return tracking_dict


def main():
    print("Generating Tracking GIF Visualizations")
    print("=" * 70)

    # Load data
    data_dir = FINAL_PROJECT_DIR / 'cti_data'
    corners_df = pl.read_parquet(data_dir / 'corners_dataset.parquet')
    predictions_df = pl.read_csv(data_dir / 'predictions.csv')

    # Load xT surface
    with open(data_dir / 'xt_surface.pkl', 'rb') as f:
        xt_surface = pickle.load(f)

    print(f"\nLoaded {len(corners_df)} corners")
    print(f"Loaded {len(predictions_df)} predictions")

    # Example: Generate GIFs for Liverpool (team_id=2)
    team_id = 2
    team_name = "Liverpool Football Club"

    print(f"\nGenerating tracking GIF samples for {team_name}...")

    # Get matches for this team
    team_corners = corners_df.filter(pl.col('team_id') == team_id)
    team_match_ids = team_corners['match_id'].unique().to_list()[:3]  # First 3 matches

    print(f"Loading tracking data for {len(team_match_ids)} matches...")
    tracking_dict = load_tracking_data_for_matches(team_match_ids)

    if not tracking_dict:
        print("ERROR: No tracking data found!")
        print("Please update the tracking_dir path in load_tracking_data_for_matches()")
        return

    # Generate GIF visualizations for 4-5 sample corners
    print(f"\nGenerating GIF visualizations...")
    corner_gifs = generate_team_corner_samples(
        team_id=team_id,
        team_name=team_name,
        corners_df=corners_df,
        tracking_dict=tracking_dict,
        predictions_df=predictions_df,
        xt_surface=xt_surface,
        max_corners=5,
        use_gif=True  # Set to True for animated GIFs
    )

    print(f"\n[DONE] Generated {len(corner_gifs)} tracking GIFs for {team_name}")

    # Save GIFs to files (optional)
    output_dir = FINAL_PROJECT_DIR / 'cti_outputs' / 'tracking_gifs'
    output_dir.mkdir(exist_ok=True)

    import base64
    for corner_id, gif_data in corner_gifs.items():
        # Extract base64 data (remove "data:image/gif;base64," prefix)
        base64_str = gif_data.split(',')[1]
        gif_bytes = base64.b64decode(base64_str)

        output_path = output_dir / f'{team_name.replace(" ", "_")}_corner_{corner_id}.gif'
        with open(output_path, 'wb') as f:
            f.write(gif_bytes)

        print(f"  Saved: {output_path}")

    print(f"\nGIFs saved to: {output_dir}")


if __name__ == "__main__":
    main()
