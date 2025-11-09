#!/Users/mpellon/dev/personal-site/portfolio/venv/bin/python3
"""
NFL Game Day Monitor

Runs on NFL game days and checks for completed games every hour.
Automatically updates the predictions page and pushes changes to GitHub.

Usage:
    python nfl_game_day_monitor.py

To run as a background service:
    nohup python nfl_game_day_monitor.py >> nfl_monitor.log 2>&1 &

To schedule with cron (runs hourly on Thursdays, Sundays, and Mondays):
    0 * * * 0,1,4 cd /path/to/portfolio && python nfl_game_day_monitor.py
"""

import logging
import time
from datetime import datetime
from pathlib import Path

from update_nfl_results import main as update_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "nfl_monitor.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def is_game_day() -> bool:
    """
    Check if today is an NFL game day.

    NFL games are typically played on:
    - Thursday (Thursday Night Football)
    - Sunday (most games)
    - Monday (Monday Night Football)

    Returns:
        True if today is a game day, False otherwise
    """
    # 0 = Monday, 3 = Thursday, 6 = Sunday
    game_days = {0, 3, 6}
    return datetime.now().weekday() in game_days


def is_game_time() -> bool:
    """
    Check if it's during typical NFL game hours.

    Games typically run from:
    - Thursday: 8:00 PM - 11:30 PM ET
    - Sunday: 1:00 PM - 11:30 PM ET
    - Monday: 8:00 PM - 11:30 PM ET

    For simplicity, we check between 12 PM and 12 AM ET (noon to midnight).

    Returns:
        True if within game hours, False otherwise
    """
    current_hour = datetime.now().hour
    # Check if between 12 PM and 11:59 PM (12-23 in 24hr format)
    return 12 <= current_hour <= 23


def run_continuous_monitoring(interval_minutes: int = 60) -> None:
    """
    Run continuous monitoring loop.

    Args:
        interval_minutes: Minutes between checks (default 60)
    """
    logger.info("Starting NFL game day monitoring")
    logger.info(f"Check interval: {interval_minutes} minutes")

    while True:
        try:
            current_time = datetime.now()
            logger.info(f"Running check at {current_time}")

            if is_game_day():
                logger.info("Today is a game day")

                if is_game_time():
                    logger.info("Within game time window - checking for updates")
                    update_results()
                else:
                    logger.info("Outside game time window - skipping")
            else:
                logger.info("Not a game day - skipping")

            # Wait for next interval
            logger.info(f"Waiting {interval_minutes} minutes until next check")
            time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}", exc_info=True)
            logger.info("Continuing after error...")
            time.sleep(60)  # Wait 1 minute before retrying


def run_single_check() -> None:
    """
    Run a single check immediately.

    Useful for testing or cron-based scheduling.
    """
    logger.info("Running single check")

    if is_game_day() and is_game_time():
        logger.info("Game day and game time - checking for updates")
        update_results()
    else:
        logger.info("Not game day/time - skipping")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Run once (for cron)
        run_single_check()
    else:
        # Run continuous monitoring
        run_continuous_monitoring()
