# CTI Results & Reports

This document consolidates the results summary, reliability reports, and validation findings for the CTI project.

---

# Part 1: Results Summary (Sample Data)

## 1.1 Overview
This section contains **actual CTI values** generated from an inference run on 3 matches (36 corners total).

## 1.2 Team-Level CTI Results (Sample)

| Rank | Team | Avg CTI | P(shot) | Counter Risk | ΔxT | N Corners |
|------|------|---------|---------|--------------|-----|-----------|
| 1 | Liverpool | **0.040** | 0.524 | 0.000 | 0.040 | 4 |
| 2 | Man Utd | **0.034** | 0.520 | 0.000 | 0.034 | 7 |
| 3 | Fulham | **0.027** | 0.517 | 0.000 | 0.027 | 11 |
| 4 | Ipswich | **0.003** | 0.520 | 0.000 | 0.003 | 11 |
| 5 | Aston Villa | **-0.004** | 0.520 | 0.000 | -0.004 | 3 |

**Key Insights:**
*   **Liverpool (0.040)**: Highest threat, excellent spatial value creation.
*   **Aston Villa (-0.004)**: Negative CTI, meaning corners lose positional threat.

## 1.3 CTI Score Interpretation
*   **Elite (0.070+)**: Exceptional threat.
*   **Excellent (0.040 - 0.070)**: High-quality routines.
*   **Good (0.020 - 0.040)**: Solid threat.
*   **Mediocre (0.000 - 0.020)**: Minimal threat.
*   **Poor (< 0.000)**: Counter-productive.

---

# Part 2: Reliability & Calibration Reports

## 2.1 Overview
The CTI system generates reliability reports to assess model calibration quality (how well predicted probabilities match observed outcomes).

**File**: `cti_outputs/reliability_report.html` (Interactive)

## 2.2 Metrics Explained
*   **ECE (Expected Calibration Error)**: Average difference between predicted and observed.
    *   Good < 0.05, Fair < 0.10, Poor > 0.10.
*   **Brier Score**: Mean squared error. Lower is better.

## 2.3 Outputs Analyzed
*   **y₁: P(Shot)**: Probability of a shot within 10s.
*   **y₃: P(Counter)**: Probability of a counter-attack.

## 2.4 Improving Calibration
If ECE is high (> 0.10), the system uses **Isotonic Regression** for post-hoc calibration.

---

# Part 3: Validation Report

## 3.1 Static Checks
*   **Canonicalization**: PASS
*   **Corner Windows**: PASS
*   **GMM Components**: PASS
*   **NMF Model**: PASS
*   **xT Surface**: PASS
*   **Quality Gates**: PASS

## 3.2 Smoke Runs
*   **Small Pipeline**: Verified with `--max-matches 2`.
*   **Inference**: Verified with `--matches 3`.
*   **GIF Generation**: Verified.

## 3.3 Artifact Inventory
*   **Data**: `corners_dataset.parquet`, `gmm_zones.pkl`, `nmf_model.pkl`, `xt_surface.pkl`, `predictions.csv`.
*   **Figures**: `gmm_zones.png`, `nmf_features_grid.png`, `xt_surface.png`, `team_cti_table.png`.
