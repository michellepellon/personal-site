#!/Users/mpellon/dev/personal-site/portfolio/venv/bin/python3
"""
NFL Game Results Updater

Fetches live NFL scores and updates the predictions page with results.
Runs on game days to check for completed games hourly and auto-commits changes.
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

from recalculate_elo import recalculate_all

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
PORTFOLIO_DIR = Path(__file__).parent
DATA_DIR = PORTFOLIO_DIR / "data"
WEBPAGE_DATA_FILE = DATA_DIR / "webpage_data.json"
ESPN_API_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"


def fetch_nfl_scores(week: Optional[int] = None) -> dict[str, Any]:
    """
    Fetch current NFL scores from ESPN API.

    Args:
        week: Optional week number to fetch. If None, fetches current week.

    Returns:
        Dictionary containing game data from ESPN API.
    """
    try:
        params = {}
        if week:
            params["week"] = week

        response = requests.get(ESPN_API_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error fetching NFL scores: {e}")
        return {}


def normalize_team_name(espn_name: str) -> str:
    """
    Normalize ESPN team names to match prediction data format.

    ESPN uses different formats than our prediction data.
    This function maps ESPN names to our canonical team names.
    """
    # Mapping of ESPN team names to our format
    team_mapping = {
        "Cardinals": "Arizona Cardinals",
        "Falcons": "Atlanta Falcons",
        "Ravens": "Baltimore Ravens",
        "Bills": "Buffalo Bills",
        "Panthers": "Carolina Panthers",
        "Bears": "Chicago Bears",
        "Bengals": "Cincinnati Bengals",
        "Browns": "Cleveland Browns",
        "Cowboys": "Dallas Cowboys",
        "Broncos": "Denver Broncos",
        "Lions": "Detroit Lions",
        "Packers": "Green Bay Packers",
        "Texans": "Houston Texans",
        "Colts": "Indianapolis Colts",
        "Jaguars": "Jacksonville Jaguars",
        "Chiefs": "Kansas City Chiefs",
        "Raiders": "Las Vegas Raiders",
        "Chargers": "Los Angeles Chargers",
        "Rams": "Los Angeles Rams",
        "Dolphins": "Miami Dolphins",
        "Vikings": "Minnesota Vikings",
        "Patriots": "New England Patriots",
        "Saints": "New Orleans Saints",
        "Giants": "New York Giants",
        "Jets": "New York Jets",
        "Eagles": "Philadelphia Eagles",
        "Steelers": "Pittsburgh Steelers",
        "49ers": "San Francisco 49ers",
        "Seahawks": "Seattle Seahawks",
        "Buccaneers": "Tampa Bay Buccaneers",
        "Titans": "Tennessee Titans",
        "Commanders": "Washington Commanders",
    }

    # Try to find the team name in the mapping
    for short_name, full_name in team_mapping.items():
        if short_name.lower() in espn_name.lower():
            return full_name

    # If not found, return the original name
    logger.warning(f"Could not normalize team name: {espn_name}")
    return espn_name


def update_predictions_with_results(
    predictions: list[dict[str, Any]],
    espn_data: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    """
    Update prediction data with actual game results.

    Args:
        predictions: List of game predictions
        espn_data: ESPN API response data

    Returns:
        Tuple of (updated predictions, number of games updated)
    """
    updated_count = 0

    if not espn_data or "events" not in espn_data:
        logger.warning("No events found in ESPN data")
        return predictions, 0

    for event in espn_data["events"]:
        # Skip games that aren't completed
        if event["status"]["type"]["state"] != "post":
            continue

        # Extract team names and scores
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        competition = competitions[0]
        competitors = competition.get("competitors", [])

        if len(competitors) != 2:
            continue

        # ESPN API: competitors[0] is usually home, competitors[1] is away
        home_team_data = next(
            (c for c in competitors if c.get("homeAway") == "home"), None
        )
        away_team_data = next(
            (c for c in competitors if c.get("homeAway") == "away"), None
        )

        if not home_team_data or not away_team_data:
            continue

        home_team = normalize_team_name(home_team_data["team"]["displayName"])
        away_team = normalize_team_name(away_team_data["team"]["displayName"])
        home_score = int(home_team_data["score"])
        away_score = int(away_team_data["score"])

        # Find matching prediction
        for prediction in predictions:
            if (
                prediction["home_team"] == home_team
                and prediction["visiting_team"] == away_team
            ):
                # Update with actual scores if not already present
                if prediction.get("actual_home_score") is None:
                    prediction["actual_home_score"] = home_score
                    prediction["actual_away_score"] = away_score
                    updated_count += 1
                    logger.info(
                        f"Updated: {away_team} @ {home_team}: "
                        f"{away_score}-{home_score}"
                    )
                break

    return predictions, updated_count


def commit_and_push_changes(message: str) -> bool:
    """
    Commit and push changes to GitHub.

    Args:
        message: Commit message

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PORTFOLIO_DIR.parent,
            capture_output=True,
            text=True,
            check=True,
        )

        if not result.stdout.strip():
            logger.info("No changes to commit")
            return False

        # Add changes
        subprocess.run(
            ["git", "add", str(WEBPAGE_DATA_FILE.relative_to(PORTFOLIO_DIR.parent))],
            cwd=PORTFOLIO_DIR.parent,
            check=True,
        )

        # Commit with message
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=PORTFOLIO_DIR.parent,
            check=True,
        )

        # Push to remote
        subprocess.run(
            ["git", "push"],
            cwd=PORTFOLIO_DIR.parent,
            check=True,
        )

        logger.info("Successfully committed and pushed changes")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        return False


def main() -> None:
    """Main execution function."""
    logger.info("Starting NFL results update")

    # Load current predictions
    if not WEBPAGE_DATA_FILE.exists():
        logger.error(f"Predictions file not found: {WEBPAGE_DATA_FILE}")
        return

    with open(WEBPAGE_DATA_FILE) as f:
        data = json.load(f)

    current_week = data.get("current_week", 10)
    predictions = data.get("predictions", [])

    if not predictions:
        logger.warning("No predictions found to update")
        return

    # Fetch current scores
    logger.info(f"Fetching scores for week {current_week}")
    espn_data = fetch_nfl_scores(week=current_week)

    # Update predictions with results
    updated_predictions, updated_count = update_predictions_with_results(
        predictions, espn_data
    )

    if updated_count == 0:
        logger.info("No new completed games found")
        return

    # Update the data structure
    data["predictions"] = updated_predictions
    data["generated_at"] = datetime.now().isoformat()

    # Recalculate ELO ratings and playoff probabilities
    logger.info("Recalculating ELO ratings and playoff probabilities...")
    data = recalculate_all(data)

    # Write updated data back to file
    with open(WEBPAGE_DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Updated {updated_count} game(s) with results")

    # Commit and push changes
    commit_message = f"chore: update NFL Week {current_week} results ({updated_count} games)"
    if commit_and_push_changes(commit_message):
        logger.info("Changes committed and pushed to GitHub")
    else:
        logger.warning("Failed to commit/push changes")


if __name__ == "__main__":
    main()
