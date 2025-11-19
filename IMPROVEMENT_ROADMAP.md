# NFL Prediction Model Improvement Roadmap

**Created:** 2025-11-18
**Current Status:** Week 11 completed (28.6% accuracy), Feature Integration implemented but minimal impact

## Overview

After Week 11's disaster (28.6% accuracy vs 55.2% season average), we implemented Improvement #1 (Feature Integration) and discovered it had **zero impact on accuracy** (both baseline and adjusted got 4/14 correct).

This document re-evaluates all 7 proposed improvements based on Week 11's actual failure modes.

---

## Completed: ✅ Improvement #1 - Feature Integration

**Implementation:** Week 11 (Nov 18, 2025)

**What we did:**
- Collected weather and rest data for 2025 season
- Calculated ELO adjustments (rest ±20, temp -10 to 0, wind -15 to 0)
- Updated predictions to use adjusted probabilities

**Impact:**
- Accuracy: 0 games improved (still 28.6%)
- Brier Score: -1.8% improvement (0.3332 → 0.3273)
- Log Loss: -1.8% improvement (0.8959 → 0.8799)

**Lessons Learned:**
1. Adjustments were too small (avg 9.3 ELO vs 50-100 team differences)
2. Adjustments went wrong direction 57% of the time
3. Weather assumptions (symmetric penalties) may be incorrect
4. Missing injury data (not available for 2025) is critical

**Verdict:** ⚠️ Implemented but ineffective. Need to:
- Recalibrate adjustment weights (current too conservative)
- Test asymmetric weather effects (home team advantage in bad weather)
- Add injury data when available

---

## Analysis: What Would Have Fixed Week 11?

Looking at the 5 biggest prediction errors:

| Game | Predicted | Actual | Error | Root Cause |
|------|-----------|--------|-------|------------|
| HOU @ TEN | Titans 89% | Texans | 0.886 | **Stale ELO** (Titans 0-3 last 3 games) |
| BAL @ CLE | Browns 72% | Ravens | 0.719 | **Stale ELO** (Browns 0-3, Ravens 3-0 last 3) |
| WAS @ MIA | Commanders 70% | Dolphins | 0.701 | **Recent Form** (WAS 0-3 last 3) |
| CIN @ PIT | Bengals 62% | Steelers | 0.622 | **Recent Form** (CIN 0-3 last 3) |
| DET @ PHI | Lions 60% | Eagles | 0.602 | **Recent Form** (PHI 3-0 last 3) |

**Pattern:** 5 of 5 biggest errors would have been fixed by incorporating **recent form/momentum**.

Teams on 3-game win streaks: PHI, BAL, DEN, LAR, CHI, NE (all won in Week 11)
Teams on 3-game losing streaks: TEN, CLE, WAS, CIN, ATL, NYG (all lost in Week 11)

---

## Remaining Improvements: Priority Re-Ranking

### 🔥 HIGHEST PRIORITY

#### #2 - Temporal Decay / Recent Form (MOVED UP)

**Original Priority:** 2/7
**New Priority:** 1/7 (most critical)

**Why it would have helped Week 11:**
- Tennessee: Rated 1669 ELO but lost 3 straight → Model predicted 89% win, they lost
- Cleveland: Rated 1734 ELO but lost 3 straight → Model predicted 72% win, they lost
- Teams on 3-game win streaks went 6-0 in Week 11
- Teams on 3-game losing streaks went 0-5 in Week 11

**Estimated Impact:**
- Would have flipped 3-4 predictions (Tennessee, Cleveland, possibly Washington, Cincinnati)
- Could improve accuracy by 15-20 percentage points

**Implementation:**

```python
# Exponential time decay on ELO updates
k_factor = base_k * exp(-weeks_ago * decay_rate)

# Recent form adjustment (last 3 games)
recent_games = get_last_n_games(team, n=3)
performance_vs_expectation = actual_points - expected_points
momentum_adj = sum(performance_vs_expectation) / 3 * momentum_weight

# Sliding window ELO (only last 8 games count)
current_elo = calculate_elo(games[-8:])
```

**Complexity:** Medium (requires historical game tracking)
**Data Required:** ✅ Already have game results
**Recommended Next:** Yes - Implement ASAP

---

#### #4 - Ensemble with Vegas Lines (MOVED UP)

**Original Priority:** 4/7
**New Priority:** 2/7

**Why it would have helped Week 11:**
- Vegas oddsmakers incorporate information we don't have (insider injury reports, public betting patterns, sharp money)
- Vegas was likely right on Tennessee (probably had them as underdogs or small favorites)
- Ensemble reduces model variance

**Estimated Impact:**
- Vegas lines typically 65-70% accurate
- Ensemble (50% ELO + 50% Vegas) could hit 60-65% accuracy
- Improvement: +5-10 percentage points

**Implementation:**

```python
# Collect Vegas spreads and implied probabilities
vegas_spread = get_vegas_line(game)
vegas_prob = spread_to_probability(vegas_spread)

# Ensemble prediction
ensemble_prob = (
    0.40 * elo_prob +           # Base ELO
    0.30 * vegas_prob +          # Market wisdom
    0.20 * recent_form_prob +    # Momentum
    0.10 * features_prob         # Weather/rest
)
```

**Complexity:** Low (just API integration + weighting)
**Data Required:** ⚠️ Need vegas lines API (available)
**Recommended Next:** Yes - High ROI, low effort

---

### 🎯 MEDIUM PRIORITY

#### #3 - Calibration Layer (Isotonic Regression)

**Original Priority:** 3/7
**New Priority:** 3/7 (keep as-is)

**Why it would have helped Week 11:**
- Wouldn't change predictions, but would improve probability estimates
- Brier/Log Loss would improve more than the 1.8% we got
- Better for long-term confidence

**Estimated Impact:**
- Accuracy: 0 games (doesn't change predictions)
- Brier Score: -5 to -10% improvement
- Log Loss: -5 to -10% improvement

**Implementation:**

```python
from sklearn.isotonic import IsotonicRegression

# Train on historical data (weeks 1-10)
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(raw_probabilities, actual_outcomes)

# Apply to new predictions
calibrated_prob = calibrator.transform(raw_elo_probability)
```

**Complexity:** Low (sklearn has it built-in)
**Data Required:** ✅ Already have historical predictions
**Recommended Next:** Maybe - After #2 and #4

---

#### #5 - Feature Engineering (Matchup-Specific)

**Original Priority:** 5/7
**New Priority:** 4/7 (slight upgrade)

**Why it would have helped Week 11:**
- Pass-heavy teams vs weak secondaries
- Run-heavy teams vs weak run defense
- Divisional rivalry factors

**Estimated Impact:**
- Accuracy: +2-5 percentage points
- Better than current features (which did nothing)

**Implementation:**

```python
features = {
    # Offense/Defense splits
    'home_pass_offense_rating': calculate_pass_offense_elo(home),
    'away_pass_defense_rating': calculate_pass_defense_elo(away),
    'pass_matchup_advantage': home_pass_off - away_pass_def,

    # Matchup-specific
    'divisional_game': is_division_rival(home, away),
    'revenge_game': lost_to_recently(home, away),
    'primetime_factor': is_national_tv(game),

    # Situational
    'playoff_implications': playoff_race_intensity(home, away),
    'travel_distance': calculate_distance(home_city, away_city),
}
```

**Complexity:** High (requires position-level data)
**Data Required:** ⚠️ Need play-by-play data
**Recommended Next:** Later - After simpler improvements

---

### 🔍 LOWER PRIORITY

#### #6 - Model Validation (Time-Series Cross-Validation)

**Original Priority:** 6/7
**New Priority:** 5/7 (keep)

**Why this matters:**
- Prevents overfitting
- Proper evaluation of improvements
- Doesn't improve predictions directly

**Estimated Impact:**
- Accuracy: 0 (doesn't change model)
- Better understanding of true performance
- Prevents false confidence

**Implementation:**

```python
for week in range(5, 18):
    train_data = all_games[all_games['week_number'] < week]
    test_data = all_games[all_games['week_number'] == week]

    model.fit(train_data)
    predictions = model.predict(test_data)

    metrics[week] = evaluate(predictions, actuals)
```

**Complexity:** Medium (requires refactoring)
**Data Required:** ✅ Already have
**Recommended Next:** Yes - But doesn't improve Week 11 type failures

---

#### #7 - Uncertainty Quantification (Confidence Intervals)

**Original Priority:** 7/7
**New Priority:** 6/7 (keep)

**Why this matters:**
- Helps with bet sizing
- Transparency about model confidence
- Useful for high-stakes decisions

**Estimated Impact:**
- Accuracy: 0 (doesn't change predictions)
- Better decision-making under uncertainty

**Implementation:**

```python
from scipy.stats import beta

# Conformal prediction using historical errors
prediction_interval = conformal_predict(
    point_estimate=0.54,
    calibration_set=historical_errors,
    confidence_level=0.90
)
# Output: "Raiders 54% ± 12% win probability (90% CI)"
```

**Complexity:** Low (standard statistical methods)
**Data Required:** ✅ Already have
**Recommended Next:** Later - Nice to have, not critical

---

### ⚠️ RECONSIDERING

#### #1b - Feature Recalibration (NEW)

**Priority:** 1b (do with #2)

**What we learned:**
- Current feature weights are too conservative
- Weather effects might be asymmetric (home team advantage in bad weather)
- Injury weights (when available) might need to be higher

**Proposed Changes:**

```python
# CURRENT (too small)
rest_adj = rest_diff * 5.0  # ±20 cap
temp_adj = -10 if temp < 32 else 0  # symmetric
wind_adj = -15 if wind >= 20 else 0  # symmetric

# PROPOSED (more aggressive)
rest_adj = rest_diff * 10.0  # ±40 cap (doubled)
temp_adj = {
    'outdoor_home_advantage': +15 if home_used_to_cold and temp < 32
    'outdoor_away_penalty': -20 if away_from_dome and temp < 32
}
wind_adj = {
    'home_advantage': +10 if home_outdoor_stadium
    'away_penalty': -25 if away_pass_heavy and wind >= 20
}
```

**Estimated Impact:**
- Accuracy: +3-5 percentage points
- Larger adjustments would flip more predictions

**Complexity:** Low (just parameter tuning)
**Data Required:** ✅ Already have
**Recommended Next:** Yes - Test alongside #2

---

## Revised Implementation Plan

### Phase 1: Quick Wins (Next 2 Weeks)
1. **Recent Form / Momentum** (#2) - Highest impact
2. **Vegas Line Ensemble** (#4) - High ROI, low effort
3. **Feature Recalibration** (#1b) - Fix what we just built

**Expected combined impact:** +15-25 percentage points accuracy

### Phase 2: Refinement (Weeks 3-4)
4. **Calibration Layer** (#3) - Better probabilities
5. **Model Validation** (#6) - Proper testing framework

### Phase 3: Advanced (Month 2+)
6. **Feature Engineering** (#5) - Matchup-specific factors
7. **Uncertainty Quantification** (#7) - Confidence intervals

---

## Key Learnings

1. **Infrastructure ≠ Impact:** Building sophisticated feature pipelines doesn't guarantee improvement
2. **Recent Form Dominates:** Teams on 3-game streaks had perfect records in Week 11
3. **Test Assumptions:** Our weather adjustments went the wrong direction 57% of the time
4. **Start Simple:** Vegas ensemble would have been easier and more effective than feature engineering
5. **Validate Everything:** Need proper backtesting before deploying improvements

---

## Success Metrics

For each improvement, we'll measure:

- **Accuracy:** % of games predicted correctly
- **Brier Score:** Probability calibration quality
- **Log Loss:** Overall prediction quality
- **ROC-AUC:** Discrimination ability
- **Prediction Flips:** How many wrong predictions became right

Target for Week 12+:
- Accuracy: 60%+ (up from 28.6%)
- Brier Score: <0.22 (down from 0.327)
- Improvements should help, not hurt

---

**Next Action:** Implement Recent Form / Momentum adjustment for Week 12
