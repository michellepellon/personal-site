#!/Users/mpellon/dev/personal-site/portfolio/venv/bin/python3
"""
NFL ELO and Playoff Probability Recalculator

Recalculates ELO ratings based on completed game results and runs
Monte Carlo simulations for playoff probabilities.

Called automatically after game scores are updated.
"""

import json
import logging
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ESPN API endpoints
ESPN_STANDINGS_URL = "https://site.api.espn.com/apis/v2/sports/football/nfl/standings"
ESPN_SCHEDULE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

# Constants
PORTFOLIO_DIR = Path(__file__).parent
DATA_DIR = PORTFOLIO_DIR / "data"
WEBPAGE_DATA_FILE = DATA_DIR / "webpage_data.json"
CALIBRATED_METRICS_FILE = DATA_DIR / "calibrated_metrics.json"

# ELO Configuration
K_FACTOR = 20
HOME_ADVANTAGE = 52
INITIAL_ELO = 1500

# NFL Division/Conference Structure
NFL_DIVISIONS = {
    "AFC East": ["Buffalo Bills", "Miami Dolphins", "New England Patriots", "New York Jets"],
    "AFC North": ["Baltimore Ravens", "Cincinnati Bengals", "Cleveland Browns", "Pittsburgh Steelers"],
    "AFC South": ["Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Tennessee Titans"],
    "AFC West": ["Denver Broncos", "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers"],
    "NFC East": ["Dallas Cowboys", "New York Giants", "Philadelphia Eagles", "Washington Commanders"],
    "NFC North": ["Chicago Bears", "Detroit Lions", "Green Bay Packers", "Minnesota Vikings"],
    "NFC South": ["Atlanta Falcons", "Carolina Panthers", "New Orleans Saints", "Tampa Bay Buccaneers"],
    "NFC West": ["Arizona Cardinals", "Los Angeles Rams", "San Francisco 49ers", "Seattle Seahawks"],
}

TEAM_INFO = {}
for division, teams in NFL_DIVISIONS.items():
    conf = "AFC" if division.startswith("AFC") else "NFC"
    for team in teams:
        TEAM_INFO[team] = {"conf": conf, "division": division}

# ESPN team name mapping
ESPN_TEAM_MAPPING = {
    "Arizona Cardinals": "Arizona Cardinals",
    "Atlanta Falcons": "Atlanta Falcons",
    "Baltimore Ravens": "Baltimore Ravens",
    "Buffalo Bills": "Buffalo Bills",
    "Carolina Panthers": "Carolina Panthers",
    "Chicago Bears": "Chicago Bears",
    "Cincinnati Bengals": "Cincinnati Bengals",
    "Cleveland Browns": "Cleveland Browns",
    "Dallas Cowboys": "Dallas Cowboys",
    "Denver Broncos": "Denver Broncos",
    "Detroit Lions": "Detroit Lions",
    "Green Bay Packers": "Green Bay Packers",
    "Houston Texans": "Houston Texans",
    "Indianapolis Colts": "Indianapolis Colts",
    "Jacksonville Jaguars": "Jacksonville Jaguars",
    "Kansas City Chiefs": "Kansas City Chiefs",
    "Las Vegas Raiders": "Las Vegas Raiders",
    "Los Angeles Chargers": "Los Angeles Chargers",
    "Los Angeles Rams": "Los Angeles Rams",
    "Miami Dolphins": "Miami Dolphins",
    "Minnesota Vikings": "Minnesota Vikings",
    "New England Patriots": "New England Patriots",
    "New Orleans Saints": "New Orleans Saints",
    "New York Giants": "New York Giants",
    "New York Jets": "New York Jets",
    "Philadelphia Eagles": "Philadelphia Eagles",
    "Pittsburgh Steelers": "Pittsburgh Steelers",
    "San Francisco 49ers": "San Francisco 49ers",
    "Seattle Seahawks": "Seattle Seahawks",
    "Tampa Bay Buccaneers": "Tampa Bay Buccaneers",
    "Tennessee Titans": "Tennessee Titans",
    "Washington Commanders": "Washington Commanders",
}


def fetch_nfl_standings() -> dict[str, dict]:
    """
    Fetch current NFL standings from ESPN API.

    Returns:
        Dictionary mapping team names to their records {wins, losses, ties}
    """
    try:
        response = requests.get(ESPN_STANDINGS_URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        standings = {}

        # ESPN structure: children are conferences (AFC/NFC)
        # Each conference has standings.entries with teams
        for conference in data.get("children", []):
            for team_standing in conference.get("standings", {}).get("entries", []):
                team_info = team_standing.get("team", {})
                team_name = team_info.get("displayName", "")

                # Get record stats - stats is a list of dicts with name/value
                stats = {}
                for stat in team_standing.get("stats", []):
                    stat_name = stat.get("name")
                    stat_value = stat.get("value", 0)
                    if stat_name:
                        stats[stat_name] = stat_value

                wins = int(stats.get("wins", 0))
                losses = int(stats.get("losses", 0))
                ties = int(stats.get("ties", 0))

                if team_name in ESPN_TEAM_MAPPING:
                    canonical_name = ESPN_TEAM_MAPPING[team_name]
                    standings[canonical_name] = {
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                    }
                    logger.debug(f"{canonical_name}: {wins}-{losses}-{ties}")

        logger.info(f"Fetched standings for {len(standings)} teams from ESPN")
        return standings

    except requests.RequestException as e:
        logger.error(f"Error fetching NFL standings: {e}")
        return {}


def fetch_remaining_schedule(current_week: int) -> list[dict]:
    """
    Fetch remaining games for the season from ESPN API.

    Args:
        current_week: Current week number

    Returns:
        List of remaining games with home/away teams
    """
    remaining_games = []

    # Fetch weeks from current week to week 18
    for week in range(current_week, 19):
        try:
            params = {"week": week}
            response = requests.get(ESPN_SCHEDULE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            for event in data.get("events", []):
                # Skip completed games
                status = event.get("status", {}).get("type", {}).get("state", "")
                if status == "post":
                    continue

                competitions = event.get("competitions", [])
                if not competitions:
                    continue

                competition = competitions[0]
                competitors = competition.get("competitors", [])

                if len(competitors) != 2:
                    continue

                home_team_data = next(
                    (c for c in competitors if c.get("homeAway") == "home"), None
                )
                away_team_data = next(
                    (c for c in competitors if c.get("homeAway") == "away"), None
                )

                if not home_team_data or not away_team_data:
                    continue

                home_name = home_team_data["team"].get("displayName", "")
                away_name = away_team_data["team"].get("displayName", "")

                if home_name in ESPN_TEAM_MAPPING and away_name in ESPN_TEAM_MAPPING:
                    remaining_games.append({
                        "week": week,
                        "home_team": ESPN_TEAM_MAPPING[home_name],
                        "visiting_team": ESPN_TEAM_MAPPING[away_name],
                    })

        except requests.RequestException as e:
            logger.error(f"Error fetching week {week} schedule: {e}")
            continue

    logger.info(f"Fetched {len(remaining_games)} remaining games from ESPN")
    return remaining_games


def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score using ELO formula."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def margin_of_victory_multiplier(winner_elo: float, loser_elo: float, margin: int) -> float:
    """
    Calculate margin of victory multiplier following FiveThirtyEight methodology.
    Larger margins = bigger ELO changes, but diminishing returns.
    """
    elo_diff = winner_elo - loser_elo
    return math.log(max(abs(margin), 1) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))


def update_elo(
    winner_elo: float,
    loser_elo: float,
    margin: int,
    home_winner: bool,
) -> tuple[float, float]:
    """
    Update ELO ratings after a game.

    Returns:
        Tuple of (new_winner_elo, new_loser_elo)
    """
    # Adjust for home field advantage
    if home_winner:
        adjusted_winner = winner_elo + HOME_ADVANTAGE
        adjusted_loser = loser_elo
    else:
        adjusted_winner = winner_elo
        adjusted_loser = loser_elo + HOME_ADVANTAGE

    # Expected scores
    expected_winner = expected_score(adjusted_winner, adjusted_loser)
    expected_loser = 1 - expected_winner

    # Margin of victory multiplier
    mov_mult = margin_of_victory_multiplier(winner_elo, loser_elo, margin)

    # ELO changes
    winner_change = K_FACTOR * mov_mult * (1 - expected_winner)
    loser_change = K_FACTOR * mov_mult * (0 - expected_loser)

    return winner_elo + winner_change, loser_elo + loser_change


def recalculate_elo_from_results(
    ratings: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Update ELO ratings based on newly completed game results.

    Uses the pre-game ELO ratings stored in predictions to calculate
    post-game adjustments, preserving the rating history.

    Args:
        ratings: Current team ratings (pre-game ratings for current week)
        predictions: List of predictions with actual scores

    Returns:
        Updated ratings list
    """
    # Build team ELO lookup from current ratings
    team_elo = {r["team"]: r["elo_rating"] for r in ratings}

    # Process completed games that have results
    completed_games = [
        p for p in predictions
        if p.get("actual_home_score") is not None and p.get("actual_away_score") is not None
    ]

    # Sort by game_id to process in order
    completed_games.sort(key=lambda x: x.get("game_id", 0))

    games_processed = 0
    for game in completed_games:
        home_team = game["home_team"]
        away_team = game["visiting_team"]
        home_score = game["actual_home_score"]
        away_score = game["actual_away_score"]

        if home_team not in team_elo or away_team not in team_elo:
            logger.warning(f"Team not found in ratings: {home_team} or {away_team}")
            continue

        # Use the pre-game ELO from the prediction if available
        pre_home_elo = game.get("home_team_elo_rating", team_elo[home_team])
        pre_away_elo = game.get("visiting_team_elo_rating", team_elo[away_team])

        margin = abs(home_score - away_score)
        home_won = home_score > away_score

        if home_won:
            new_home, new_away = update_elo(
                pre_home_elo, pre_away_elo, margin, home_winner=True
            )
        else:
            new_away, new_home = update_elo(
                pre_away_elo, pre_home_elo, margin, home_winner=False
            )

        team_elo[home_team] = new_home
        team_elo[away_team] = new_away
        games_processed += 1

    logger.info(f"Processed {games_processed} completed games for ELO updates")

    # Update ratings list
    for rating in ratings:
        if rating["team"] in team_elo:
            rating["elo_rating"] = team_elo[rating["team"]]

    # Sort by ELO rating descending
    ratings.sort(key=lambda x: x["elo_rating"], reverse=True)

    return ratings


def simulate_game(home_elo: float, away_elo: float) -> bool:
    """
    Simulate a single game.

    Returns:
        True if home team wins, False otherwise
    """
    home_adjusted = home_elo + HOME_ADVANTAGE
    home_win_prob = expected_score(home_adjusted, away_elo)
    return random.random() < home_win_prob


def get_team_record(team: str, predictions: list[dict]) -> tuple[int, int]:
    """Get current win-loss record for a team from completed games."""
    wins = 0
    losses = 0

    for game in predictions:
        if game.get("actual_home_score") is None:
            continue

        home_score = game["actual_home_score"]
        away_score = game["actual_away_score"]

        if game["home_team"] == team:
            if home_score > away_score:
                wins += 1
            else:
                losses += 1
        elif game["visiting_team"] == team:
            if away_score > home_score:
                wins += 1
            else:
                losses += 1

    return wins, losses


def get_all_team_records(predictions: list[dict], all_teams: list[str]) -> dict[str, dict]:
    """Get current win-loss records for all teams."""
    records = {team: {"wins": 0, "losses": 0} for team in all_teams}

    for game in predictions:
        if game.get("actual_home_score") is None:
            continue

        home_team = game["home_team"]
        away_team = game["visiting_team"]
        home_score = game["actual_home_score"]
        away_score = game["actual_away_score"]

        if home_score > away_score:
            if home_team in records:
                records[home_team]["wins"] += 1
            if away_team in records:
                records[away_team]["losses"] += 1
        else:
            if away_team in records:
                records[away_team]["wins"] += 1
            if home_team in records:
                records[home_team]["losses"] += 1

    return records


def run_playoff_simulation(
    ratings: list[dict[str, Any]],
    current_week: int,
    n_simulations: int = 10000,
) -> list[dict[str, Any]]:
    """
    Run Monte Carlo playoff probability simulations.

    Fetches current standings and remaining schedule from ESPN API
    for accurate playoff projections.

    Args:
        ratings: Current team ratings
        current_week: Current NFL week
        n_simulations: Number of simulations to run

    Returns:
        List of playoff probability data for each team
    """
    team_elo = {r["team"]: r["elo_rating"] for r in ratings}

    # Fetch current standings from ESPN
    espn_standings = fetch_nfl_standings()
    if not espn_standings:
        logger.warning("Could not fetch ESPN standings, using empty records")
        espn_standings = {team: {"wins": 0, "losses": 0, "ties": 0} for team in team_elo}

    # Fetch remaining schedule from ESPN
    remaining_games = fetch_remaining_schedule(current_week)
    if not remaining_games:
        logger.warning("Could not fetch remaining schedule")

    logger.info(f"Simulating {len(remaining_games)} remaining games with {n_simulations} scenarios")

    # Track simulation results
    playoff_counts = defaultdict(int)
    bye_counts = defaultdict(int)
    total_wins = defaultdict(float)

    for sim in range(n_simulations):
        # Copy current standings
        sim_records = {
            team: {"wins": espn_standings.get(team, {}).get("wins", 0),
                   "losses": espn_standings.get(team, {}).get("losses", 0)}
            for team in team_elo
        }

        # Simulate remaining games
        for game in remaining_games:
            home_team = game["home_team"]
            away_team = game["visiting_team"]

            if home_team not in team_elo or away_team not in team_elo:
                continue

            home_wins = simulate_game(team_elo[home_team], team_elo[away_team])

            if home_wins:
                sim_records[home_team]["wins"] += 1
                sim_records[away_team]["losses"] += 1
            else:
                sim_records[away_team]["wins"] += 1
                sim_records[home_team]["losses"] += 1

        # Determine playoff teams for each conference
        for conf in ["AFC", "NFC"]:
            conf_teams = [t for t in team_elo if TEAM_INFO.get(t, {}).get("conf") == conf]

            # Sort by wins (tiebreaker: ELO rating)
            conf_teams.sort(
                key=lambda t: (sim_records[t]["wins"], team_elo[t]),
                reverse=True
            )

            # Top 7 make playoffs
            playoff_teams = conf_teams[:7]
            for team in playoff_teams:
                playoff_counts[team] += 1

            # #1 seed gets bye
            if playoff_teams:
                bye_counts[playoff_teams[0]] += 1

        # Track total wins
        for team, record in sim_records.items():
            total_wins[team] += record["wins"]

    # Calculate probabilities
    playoff_data = []
    for team in team_elo:
        info = TEAM_INFO.get(team, {"conf": "Unknown", "division": "Unknown"})
        current_wins = espn_standings.get(team, {}).get("wins", 0)
        current_losses = espn_standings.get(team, {}).get("losses", 0)
        playoff_data.append({
            "team": team,
            "conf": info["conf"],
            "elo_rating": team_elo[team],
            "playoff_prob_pct": round(100 * playoff_counts[team] / n_simulations, 1),
            "bye_prob_pct": round(100 * bye_counts[team] / n_simulations, 1),
            "avg_wins": round(total_wins[team] / n_simulations, 1),
            "current_wins": current_wins,
            "current_losses": current_losses,
            "n_scenarios": n_simulations,
        })

    # Sort by playoff probability descending
    playoff_data.sort(key=lambda x: (x["playoff_prob_pct"], x["avg_wins"]), reverse=True)

    return playoff_data


def calculate_week_performance(predictions: list[dict], week: int) -> dict[str, Any]:
    """Calculate performance metrics for a specific week."""
    week_games = [
        p for p in predictions
        if p.get("week_number") == week
        and p.get("actual_home_score") is not None
    ]

    if not week_games:
        return None

    correct = 0
    brier_sum = 0
    log_loss_sum = 0

    for game in week_games:
        home_score = game["actual_home_score"]
        away_score = game["actual_away_score"]
        home_won = 1 if home_score > away_score else 0
        home_prob = game["home_win_probability"]

        # Check if prediction was correct
        predicted_home = home_prob > 0.5
        if predicted_home == bool(home_won):
            correct += 1

        # Brier score component
        brier_sum += (home_prob - home_won) ** 2

        # Log loss component (with small epsilon to avoid log(0))
        eps = 1e-10
        prob = max(min(home_prob, 1 - eps), eps)
        if home_won:
            log_loss_sum -= math.log(prob)
        else:
            log_loss_sum -= math.log(1 - prob)

    n_games = len(week_games)
    accuracy = correct / n_games
    brier = brier_sum / n_games
    log_loss = log_loss_sum / n_games

    # Determine rating
    if brier < 0.20:
        rating = "Excellent"
    elif brier < 0.25:
        rating = "Good"
    elif brier < 0.30:
        rating = "Fair"
    else:
        rating = "Needs improvement"

    return {
        "week_number": week,
        "brier_score": round(brier, 4),
        "log_loss": round(log_loss, 4),
        "accuracy": accuracy,
        "performance_rating": rating,
    }


def recalculate_all(data: dict[str, Any]) -> dict[str, Any]:
    """
    Recalculate all ELO ratings, playoff probabilities, and performance metrics.

    Uses pre-game ELO ratings stored in predictions to update post-game ratings.

    Args:
        data: The webpage data dictionary

    Returns:
        Updated data dictionary
    """
    predictions = data.get("predictions", [])
    ratings = data.get("ratings", [])
    current_week = data.get("current_week", 14)

    # Update ELO ratings based on completed game results
    # Uses pre-game ELO from predictions, not resetting from scratch
    logger.info("Updating ELO ratings from game results...")
    ratings = recalculate_elo_from_results(ratings, predictions)
    data["ratings"] = ratings

    logger.info("Running playoff probability simulations...")
    playoffs = run_playoff_simulation(ratings, current_week, n_simulations=10000)
    data["playoffs"] = playoffs

    # Update performance for current week if we have results
    logger.info(f"Calculating Week {current_week} performance metrics...")
    week_perf = calculate_week_performance(predictions, current_week)
    if week_perf:
        # Update or add week performance
        performance = data.get("performance", [])
        existing_idx = next(
            (i for i, p in enumerate(performance) if p["week_number"] == current_week),
            None
        )
        if existing_idx is not None:
            performance[existing_idx] = week_perf
        else:
            performance.append(week_perf)
            performance.sort(key=lambda x: x["week_number"])
        data["performance"] = performance
        logger.info(f"Week {current_week}: {week_perf['accuracy']*100:.1f}% accuracy, Brier: {week_perf['brier_score']:.3f}")

    data["generated_at"] = datetime.now().isoformat()

    return data


def main() -> None:
    """Main execution function."""
    logger.info("Starting ELO and playoff probability recalculation")

    if not WEBPAGE_DATA_FILE.exists():
        logger.error(f"Data file not found: {WEBPAGE_DATA_FILE}")
        return

    with open(WEBPAGE_DATA_FILE) as f:
        data = json.load(f)

    # Recalculate everything
    data = recalculate_all(data)

    # Write updated data
    with open(WEBPAGE_DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Recalculation complete. Data saved.")

    # Also update calibrated metrics timestamp
    if CALIBRATED_METRICS_FILE.exists():
        with open(CALIBRATED_METRICS_FILE) as f:
            calibrated = json.load(f)
        calibrated["generated_at"] = datetime.now().isoformat()
        with open(CALIBRATED_METRICS_FILE, "w") as f:
            json.dump(calibrated, f, indent=2)


if __name__ == "__main__":
    main()
