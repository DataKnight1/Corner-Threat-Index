# Corner Threat Index (CTI) - User Guide

This guide provides instructions on how to install, run, and interpret the outputs of the CTI solution.

## 1. Overview

The CTI project combines deep learning with tactical analysis to quantify the threat of corner kicks. It integrates:
*   **Shaw & Gopaladesikan (Sloan 2018)**: GMM zones + NMF routine discovery.
*   **Deep Learning**: Multi-task GNN for outcome prediction.
*   **xT Surface**: Expected Threat model for territorial gain.
*   **CTI Metric**: Net expected goal impact from corners.

### Key Modules
*   **Pipeline**: `Final_Project/cti_pipeline.py`
*   **Source Code**: `Final_Project/cti/`
*   **Outputs**: `Final_Project/cti_outputs/`
*   **Data Artifacts**: `Final_Project/cti_data/`

---

## 2. Installation

1.  **Prerequisites**: Python 3.11+
2.  **Install Dependencies**:
    ```bash
    pip install -r Final_Project/requirements.txt
    ```

---

## 3. Quick Start

To run the full pipeline (training, inference, and visualization) on a subset of matches:

```bash
python Final_Project/cti_pipeline.py --mode train --max-matches 20
```

To run on the full dataset:

```bash
python Final_Project/cti_pipeline.py --mode train
```

---

## 4. Detailed Usage

### Pipeline Arguments
*   `--mode train`: Runs the full training pipeline.
*   `--max-matches N`: Limits the dataset to `N` matches (useful for testing).
*   `--skip-infer`: Skips the inference step.
*   `--skip-gif`: Skips generating the animated GIF.

### Standalone Runners
You can run specific components individually:

*   **Inference & Team Table**:
    ```bash
    python Final_Project/cti/cti_infer_cti.py --matches 3 --checkpoint best
    ```
*   **Corner Animation (GIF)**:
    ```bash
    python Final_Project/cti/cti_create_corner_animation.py --count 3 --freeze 6 --fps 10
    ```
*   **SHAP & Calibration Report**:
    ```bash
    python Final_Project/cti/cti_shap_report.py --matches 5 --checkpoint best
    ```

---

## 5. Understanding Outputs

### Team CTI Table (`cti_outputs/team_cti_table.png`)
The system automatically generates a team ranking table based on the average CTI.

**Metrics:**
*   **CTI**: Corner Threat Index (Net Expected Goal Impact).
    *   Formula: `CTI = y₁·y₂ - 0.5·y₃·y₄ + y₅`
*   **y₁ (Shot Prob)**: Probability of a shot within 10s.
*   **y₂ (Shot Quality)**: Max xG of the shot.
*   **y₃ (Counter Prob)**: Probability of a counter-attack.
*   **y₄ (Counter Danger)**: Max xG of the counter-attack.
*   **y₅ (Territory)**: Territorial gain (ΔxT).
*   **Counter Risk**: `0.5 * y₃ * y₄`.

### Data Files (`cti_data/`)
*   `team_cti_detailed.csv`: Detailed statistics for every team.
    *   **cti_goal_weighted**: CTI boosted by actual goal rate (Recommended metric).
*   `predictions.csv`: Model predictions for every corner.
*   `corners_dataset.parquet`: The processed dataset used for training.

### Visualizations (`cti_outputs/`)
*   `gmm_zones.png`: Visualization of the spatial zones.
*   `nmf_features_grid.png`: The discovered tactical routines.
*   `xt_surface.png`: The Expected Threat heatmap.
*   `reliability_report.html`: Interactive calibration report.

---

## 6. Troubleshooting

### "Only 5 teams in table"
*   **Cause**: You are running with `--max-matches`.
*   **Fix**: Run on the full dataset without the limit.

### "Counter risk is 0.000"
*   **Cause**: Model trained before the Phase 2b fix.
*   **Fix**: Retrain the model using `python cti_pipeline.py --mode train`.

### "Team names show as IDs"
*   **Cause**: Missing team mapping file.
*   **Fix**: Ensure `data/meta/teams.parquet` exists.
