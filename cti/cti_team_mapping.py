"""
Author: Tiago
Date: 2025-12-04
Description: Team ID to name mapping utilities for the CTI pipeline. Reads SkillCorner meta files and merges with a predefined Premier League mapping to ensure coverage when metadata is incomplete.
"""

import json
from pathlib import Path
from typing import Dict


def build_team_name_map(meta_dir: Path, use_fallback: bool = True) -> Dict[int, str]:
    """
    Build a team_id -> team_name mapping from SkillCorner meta files.

    :param meta_dir: Path to directory containing match metadata JSON files.
    :param use_fallback: If True, merge with PREMIER_LEAGUE_2024_TEAMS to fill gaps.
    :return: Dictionary mapping team_id (int) to team_name (str).
    """
    # Start with predefined mapping if fallback is enabled
    team_map = PREMIER_LEAGUE_2024_TEAMS.copy() if use_fallback else {}

    if not meta_dir.exists():
        print(f"Warning: Meta directory not found: {meta_dir}")
        if use_fallback:
            print(f"Using predefined team mapping: {len(team_map)} teams")
        return team_map

    # Load from meta files and override predefined names if present
    meta_count = 0
    for meta_file in meta_dir.glob('*.json'):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            home = data.get('home_team', {})
            away = data.get('away_team', {})

            if home.get('id') and home.get('name'):
                team_map[int(home['id'])] = str(home['name'])
                meta_count += 1
            if away.get('id') and away.get('name'):
                team_map[int(away['id'])] = str(away['name'])
                meta_count += 1
        except Exception as e:
            # Skip files that can't be parsed
            continue

    print(f"OK Built team mapping: {len(team_map)} teams (from meta: {meta_count}, predefined: {len(PREMIER_LEAGUE_2024_TEAMS)})")
    return team_map


# Predefined mapping for Premier League 2024 teams
# This serves as a fallback if meta files are not available
# Note: Some teams may not appear in meta files but appear in the event data
PREMIER_LEAGUE_2024_TEAMS = {
    2: "Liverpool Football Club",
    3: "Arsenal Football Club",
    31: "Manchester United",
    32: "Newcastle United",
    37: "West Ham United",  # Not in meta files, but appears in event data
    39: "Aston Villa",
    40: "Manchester City",
    41: "Everton",
    44: "Tottenham Hotspur",
    48: "Fulham",
    49: "Chelsea",
    52: "Wolverhampton Wanderers",
    58: "Southampton",
    60: "Crystal Palace",
    62: "Leicester City",
    63: "Bournemouth",
    308: "Brighton and Hove Albion",
    747: "Nottingham Forest",
    752: "Ipswich Town",
    754: "Brentford FC",
}


if __name__ == "__main__":
    # Test the function
    from pathlib import Path

    meta_dir = Path("C:/Users/Tiago/Solutions/twelve-deep-learning/PremierLeague_data/2024/meta")
    team_map = build_team_name_map(meta_dir)

    print("\nTeam ID -> Name mapping:")
    for tid in sorted(team_map.keys()):
        print(f"  {tid}: {team_map[tid]}")
