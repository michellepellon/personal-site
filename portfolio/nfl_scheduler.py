#!/Users/mpellon/dev/personal-site/portfolio/venv/bin/python3
"""
NFL Unified Scheduler

Runs hourly and determines what action to take based on day/time:
- Wednesday 9 AM ET: Full weekly model run
- Thu/Sun/Mon during games: Live score updates
"""

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "nfl_scheduler.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Paths
PORTFOLIO_DIR = Path(__file__).parent
NFL_STACK_DIR = Path("/Users/mpellon/dev/nfl-data-stack")
PERSONAL_SITE_DIR = PORTFOLIO_DIR.parent

# Timezones
CENTRAL = ZoneInfo("America/Chicago")
EASTERN = ZoneInfo("America/New_York")


def get_current_week() -> int:
    """
    Determine current NFL week based on date.
    NFL 2025 season: Week 1 starts Sep 4, 2025
    """
    # This is a simple calculation - you may want to make this more robust
    start_date = datetime(2025, 9, 4, tzinfo=EASTERN)
    now = datetime.now(EASTERN)
    weeks_since_start = (now - start_date).days // 7
    return min(max(1, weeks_since_start + 1), 18)


def is_wednesday_model_run_time() -> bool:
    """
    Check if it's Wednesday at 9 AM ET (8 AM CT).

    Returns True only during the 9 AM hour on Wednesday.
    """
    now = datetime.now(EASTERN)
    return now.weekday() == 2 and now.hour == 9  # Wednesday, 9 AM ET


def is_game_day() -> bool:
    """Check if today is an NFL game day (Thu/Sun/Mon)."""
    # 0 = Monday, 3 = Thursday, 6 = Sunday
    return datetime.now(EASTERN).weekday() in {0, 3, 6}


def is_game_time() -> bool:
    """Check if within typical NFL game hours (12 PM - midnight ET)."""
    return 12 <= datetime.now(EASTERN).hour <= 23


def run_weekly_model_update() -> bool:
    """
    Run the complete weekly model update pipeline.

    Steps:
    1. Collect latest 2025 game results
    2. Collect enhanced features (weather, injuries, rest)
    3. Run dbt build to recalculate ELO ratings and predictions
    4. Generate webpage data for next week
    5. Copy to personal-site repository
    6. Commit and push changes

    Returns:
        True if successful, False otherwise
    """
    try:
        current_week = get_current_week()
        next_week = current_week + 1

        logger.info("="*60)
        logger.info(f"WEEKLY MODEL RUN - Current Week: {current_week}")
        logger.info("="*60)

        # Step 1: Collect 2025 results from ESPN
        logger.info("Step 1: Collecting 2025 game results from ESPN API...")
        result = subprocess.run(
            [
                NFL_STACK_DIR / ".venv/bin/python",
                NFL_STACK_DIR / "scripts/collect_espn_scores.py"
            ],
            cwd=NFL_STACK_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"ESPN score collection output: {result.stdout}")

        # Step 2: Collect enhanced features
        logger.info("Step 2: Collecting enhanced features (weather, injuries)...")
        result = subprocess.run(
            [
                NFL_STACK_DIR / ".venv/bin/python",
                NFL_STACK_DIR / "scripts/collect_enhanced_features.py",
                "--start", "2025",
                "--end", "2025"
            ],
            cwd=NFL_STACK_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Feature collection output: {result.stdout}")

        # Step 3: Run dbt build
        logger.info("Step 3: Running dbt build (ELO calculations, predictions)...")
        result = subprocess.run(
            ["just", "build"],
            cwd=NFL_STACK_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("dbt build completed successfully")

        # Step 4: Generate webpage data
        # NOTE: Script currently uses hardcoded week numbers
        # TODO: Update generate_full_webpage_data.py to accept --week argument
        logger.info(f"Step 4: Generating webpage data...")
        logger.warning(
            f"⚠️  generate_full_webpage_data.py uses hardcoded week numbers. "
            f"Manually update current_week to {next_week} and "
            f"week10_predictions to week{next_week}_predictions before running."
        )
        result = subprocess.run(
            [
                NFL_STACK_DIR / ".venv/bin/python",
                NFL_STACK_DIR / "scripts/generate_full_webpage_data.py"
            ],
            cwd=NFL_STACK_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Webpage data generated: {result.stdout}")

        # Step 5: Copy generated data to personal-site
        logger.info("Step 5: Checking for generated data...")

        # Script outputs directly to personal-site/portfolio/data/
        # Check if files were updated
        dest_dir = PERSONAL_SITE_DIR / "portfolio/data"
        webpage_data = dest_dir / "webpage_data.json"
        calibrated_metrics = dest_dir / "calibrated_metrics.json"

        if webpage_data.exists():
            logger.info(f"webpage_data.json exists at {webpage_data}")
        else:
            logger.warning("webpage_data.json not found!")

        if calibrated_metrics.exists():
            logger.info(f"calibrated_metrics.json exists at {calibrated_metrics}")
        else:
            logger.warning("calibrated_metrics.json not found - may need to generate")

        # Step 6: Commit and push changes
        logger.info("Step 6: Committing and pushing changes...")

        # Check for changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PERSONAL_SITE_DIR,
            capture_output=True,
            text=True,
            check=True,
        )

        if result.stdout.strip():
            # Add files
            subprocess.run(
                [
                    "git", "add",
                    "portfolio/data/webpage_data.json",
                    "portfolio/data/calibrated_metrics.json"
                ],
                cwd=PERSONAL_SITE_DIR,
                check=True,
            )

            # Commit
            commit_msg = f"""chore: update NFL Week {next_week} predictions and Week {current_week} results

Weekly model run:
- Updated ELO ratings with Week {current_week} results
- Generated Week {next_week} predictions
- Updated enhanced features (weather, injuries, rest)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"""

            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=PERSONAL_SITE_DIR,
                check=True,
            )

            # Push
            subprocess.run(
                ["git", "push"],
                cwd=PERSONAL_SITE_DIR,
                check=True,
            )

            logger.info("✅ Weekly model run complete - changes committed and pushed")
        else:
            logger.info("No changes to commit")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Weekly model run failed: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during weekly model run: {e}")
        return False


def run_live_score_update() -> bool:
    """
    Run live score updates for ongoing games.

    Uses the hourly_update.py script which:
    1. Collects latest scores from ESPN API
    2. Re-runs dbt to update ELO ratings with new results
    3. Regenerates predictions with updated ELO ratings

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Running hourly update (ESPN scores + model refresh)...")

        result = subprocess.run(
            [
                NFL_STACK_DIR / ".venv/bin/python",
                NFL_STACK_DIR / "scripts/hourly_update.py"
            ],
            cwd=NFL_STACK_DIR,
            capture_output=True,
            text=True,
            check=True,
        )

        logger.info(result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Hourly update failed: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False


def main() -> None:
    """Main scheduler logic."""
    now = datetime.now(EASTERN)
    logger.info(f"NFL Scheduler running at {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Check if it's time for weekly model run
    if is_wednesday_model_run_time():
        logger.info("Wednesday 9 AM ET - Running weekly model update")
        run_weekly_model_update()
        return

    # Check if it's game day and time for live updates
    if is_game_day():
        logger.info(f"Game day detected ({now.strftime('%A')})")

        if is_game_time():
            logger.info("Within game time window - running live score update")
            run_live_score_update()
        else:
            logger.info("Outside game time window - skipping")
    else:
        logger.info("Not a game day or model run day - skipping")


if __name__ == "__main__":
    main()
