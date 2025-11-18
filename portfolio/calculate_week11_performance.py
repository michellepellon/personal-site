#!/Users/mpellon/dev/personal-site/portfolio/venv/bin/python3
"""
Calculate Week 11 Performance Metrics

Computes Brier score, log loss, and accuracy for Week 11 predictions
and adds correctness indicators to each game.
"""

import json
import math
from pathlib import Path
from typing import Any


# Constants
PORTFOLIO_DIR = Path(__file__).parent
DATA_DIR = PORTFOLIO_DIR / "data"
WEBPAGE_DATA_FILE = DATA_DIR / "webpage_data.json"


def calculate_brier_score(predictions: list[dict[str, Any]]) -> float:
    """
    Calculate Brier score for predictions.

    Brier score = (1/n) * Σ(probability - outcome)²
    where outcome is 1 if prediction correct, 0 otherwise
    """
    total = 0.0
    count = 0

    for pred in predictions:
        if pred.get("actual_home_score") is None:
            continue

        # Determine actual winner
        home_score = pred["actual_home_score"]
        away_score = pred["actual_away_score"]

        if home_score > away_score:
            actual_winner = pred["home_team"]
        else:
            actual_winner = pred["visiting_team"]

        # Get predicted winner and probability
        predicted_winner = pred["predicted_winner"]

        # Get the probability for the predicted winner
        if predicted_winner == pred["home_team"]:
            prob = pred["home_win_probability"]
        else:
            prob = 1 - pred["home_win_probability"]

        # Outcome is 1 if correct, 0 if wrong
        outcome = 1.0 if predicted_winner == actual_winner else 0.0

        # Add squared error
        total += (prob - outcome) ** 2
        count += 1

    return total / count if count > 0 else 0.0


def calculate_log_loss(predictions: list[dict[str, Any]]) -> float:
    """
    Calculate log loss (cross-entropy loss) for predictions.

    Log loss = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]
    where y is actual outcome (1 if correct, 0 if wrong), p is predicted probability
    """
    total = 0.0
    count = 0
    epsilon = 1e-15  # Small value to avoid log(0)

    for pred in predictions:
        if pred.get("actual_home_score") is None:
            continue

        # Determine actual winner
        home_score = pred["actual_home_score"]
        away_score = pred["actual_away_score"]

        if home_score > away_score:
            actual_winner = pred["home_team"]
        else:
            actual_winner = pred["visiting_team"]

        # Get predicted winner and probability
        predicted_winner = pred["predicted_winner"]

        # Get the probability for the predicted winner
        if predicted_winner == pred["home_team"]:
            prob = pred["home_win_probability"]
        else:
            prob = 1 - pred["home_win_probability"]

        # Clip probability to avoid log(0)
        prob = max(epsilon, min(1 - epsilon, prob))

        # Outcome is 1 if correct, 0 if wrong
        outcome = 1.0 if predicted_winner == actual_winner else 0.0

        # Add log loss
        total -= (outcome * math.log(prob) + (1 - outcome) * math.log(1 - prob))
        count += 1

    return total / count if count > 0 else 0.0


def calculate_accuracy(predictions: list[dict[str, Any]]) -> float:
    """
    Calculate simple accuracy (% of correct predictions).
    """
    correct = 0
    total = 0

    for pred in predictions:
        if pred.get("actual_home_score") is None:
            continue

        # Determine actual winner
        home_score = pred["actual_home_score"]
        away_score = pred["actual_away_score"]

        if home_score > away_score:
            actual_winner = pred["home_team"]
        else:
            actual_winner = pred["visiting_team"]

        # Check if prediction was correct
        if pred["predicted_winner"] == actual_winner:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0.0


def get_performance_rating(brier_score: float, log_loss: float, accuracy: float) -> str:
    """
    Assign a performance rating based on metrics.

    Based on observed patterns from weeks 1-10:
    - Excellent: accuracy >= 0.70, brier < 0.20, log_loss < 0.60
    - Good: accuracy >= 0.60, brier < 0.25, log_loss < 0.70
    - Fair: accuracy >= 0.40
    - Needs improvement: otherwise
    """
    if accuracy >= 0.70 and brier_score < 0.20 and log_loss < 0.60:
        return "Excellent"
    elif accuracy >= 0.60 and brier_score < 0.25 and log_loss < 0.70:
        return "Good"
    elif accuracy >= 0.40:
        return "Fair"
    else:
        return "Needs improvement"


def add_correctness_to_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add 'is_correct' field to each prediction.
    """
    for pred in predictions:
        if pred.get("actual_home_score") is None:
            pred["is_correct"] = None
            continue

        # Determine actual winner
        home_score = pred["actual_home_score"]
        away_score = pred["actual_away_score"]

        if home_score > away_score:
            actual_winner = pred["home_team"]
        else:
            actual_winner = pred["visiting_team"]

        # Mark if prediction was correct
        pred["is_correct"] = pred["predicted_winner"] == actual_winner

    return predictions


def main() -> None:
    """Main execution function."""
    print("Calculating Week 11 performance metrics...")

    # Load current data
    if not WEBPAGE_DATA_FILE.exists():
        print(f"Error: {WEBPAGE_DATA_FILE} not found")
        return

    with open(WEBPAGE_DATA_FILE) as f:
        data = json.load(f)

    # Get Week 11 predictions
    week11_predictions = [
        p for p in data["predictions"]
        if p["week_number"] == 11
    ]

    if not week11_predictions:
        print("No Week 11 predictions found")
        return

    # Calculate metrics
    brier = calculate_brier_score(week11_predictions)
    log_loss = calculate_log_loss(week11_predictions)
    accuracy = calculate_accuracy(week11_predictions)
    rating = get_performance_rating(brier, log_loss, accuracy)

    print(f"\nWeek 11 Performance:")
    print(f"  Brier Score: {brier:.6f}")
    print(f"  Log Loss: {log_loss:.6f}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Rating: {rating}")

    # Add Week 11 performance to data
    week11_performance = {
        "week_number": 11,
        "brier_score": brier,
        "log_loss": log_loss,
        "accuracy": accuracy,
        "performance_rating": rating
    }

    # Add or update in performance list
    performance_list = data.get("performance", [])

    # Remove existing Week 11 entry if present
    performance_list = [p for p in performance_list if p["week_number"] != 11]

    # Add new Week 11 entry
    performance_list.append(week11_performance)
    performance_list.sort(key=lambda x: x["week_number"])

    data["performance"] = performance_list

    # Add correctness to predictions
    data["predictions"] = add_correctness_to_predictions(data["predictions"])

    # Save updated data
    with open(WEBPAGE_DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Updated {WEBPAGE_DATA_FILE}")
    print(f"   Added Week 11 performance metrics")
    print(f"   Added 'is_correct' field to all predictions")


if __name__ == "__main__":
    main()
