# Corner Threat Index: From Corners to Cinema

- Author: You  
- Repo root: `Final_Project/`
- Figures: `Final_Project/cti_outputs/`  
- Data artifacts: `Final_Project/cti_data/`

## Abstract
- End-to-end, reproducible pipeline to quantify and communicate threat from corner kicks.
- Methodology combines:
  - GMM zone modeling of initial/target positions to encode corner “runs” (42‑d).
  - NMF to discover recurring run-combinations (features/routines).
  - Half‑pitch xT estimation tailored for corners.
  - A graph-based, multi‑task deep model to predict five outcomes and derive CTI.
- Outputs include publication-ready figures (with team crests), a feature grid, team tables, clean CSV/PKL artifacts, and a GIF that fuses tracking + models for multiple corners.

## Why Corners Matter
- Corners drive meaningful shot volume and pressure. Clubs need to identify repeatable routines, quantify risk vs reward (including counters), and explain results to coaches.
- CTI (Corner Threat Index) balances:
  - Offense: chance of shot (y1), expected xG if a shot occurs (y2), added ΔxT (y5).
  - Defense: counter-shot risk and magnitude (y3·y4).
- CTI = y1·y2 − λ·y3·y4 + γ·y5.

## Data
- Inputs (Premier League 2024):
  - Events: `PremierLeague_data/2024/dynamic/{match_id}.parquet`
  - Tracking: `PremierLeague_data/2024/tracking/{match_id}.json` (SkillCorner centered meters)
- Outputs split for clarity:
  - Figures (PNGs/GIF): `Final_Project/cti_outputs/`
  - Data artifacts (CSV/Parquet/PKL/NPY/TXT): `Final_Project/cti_data/`

## Pipeline Overview
- Orchestrator: `Final_Project/cti_pipeline.py`
- Phases:
  1) Corner extraction + windowing: `cti_corner_extraction.py`
  2) Feature engineering + models (GMM, 42‑d runs, NMF, xT): `cti_gmm_zones.py`, `cti_nmf_routines.py`, `cti_xt_surface_half_pitch.py`
  3) Deep model training (GraphSAGE, multi‑task): `cti_integration.py`
  4) Evaluation placeholder + post-processing (inference + GIF)
- Post-run defaults: inference + team CTI table + multi‑corner GIF (can be skipped by flags).

## Modeling Method

### GMM Zones (Initial + Target)
- 6‑component GMM for initial positions (≈ −2s).
- 15‑component GMM for target positions with 7 active PA components.
- Canonicalization flips ensure the attacking team always attacks the top‑right.
- Artifacts: `cti_data/gmm_zones.pkl`, figure `cti_outputs/gmm_zones.png`.

### 42‑Dimensional Run Vectors
- Probability mass across 6×7 initial/target zone combinations.
- Artifact: `cti_data/run_vectors.npy`.

### NMF Routines
- Discovers recurring run combinations (features).
- Produces feature grid and “top corners for feature”.
- Artifacts: `cti_data/nmf_model.pkl`, figures `cti_outputs/nmf_features_grid.png`, `cti_outputs/feature_12_top_corners.png`.

### xT (Corner Phase, Half‑Pitch)
- Corner-zone xT via value iteration over actions.
- Visualization focused on attacking right half (x ∈ [52.5, 105]).
- Zero xT visualized in black; positive values in RdYlGn.
- Artifacts: `cti_data/xt_surface.pkl`, figure `cti_outputs/xt_surface.png`.

### Multi‑Task Deep Model (GraphSAGE)
- Predicts y1–y5 (shot, xG, counter-shot, counter-xG, ΔxT).
- CTI computed differentiably from predictions.
- Lightning checkpoints: `cti_outputs/checkpoints/`.

## Engineering Details That Matter
- Canonicalization in SkillCorner (SC): ball starts at corner; players in PA; visuals consistent across games.
- Crest handling: RGBA logos preserved and placed properly; missing crest names listed for quick fixing.
- Dark‑theme legibility: white-text legends with transparent frames.
- Clean layout: artifacts in `cti_data/`, figures in `cti_outputs/`.

## Artifacts Produced
- Data
  - `cti_data/corners_dataset.parquet`
  - `cti_data/gmm_zones.pkl`, `cti_data/run_vectors.npy`, `cti_data/nmf_model.pkl`, `cti_data/xt_surface.pkl`
  - `cti_data/team_top_feature.csv` (team→top NMF feature + metrics)
  - `cti_data/predictions.csv` (y1–y5 + CTI per corner)
- Figures
  - `cti_outputs/gmm_zones.png`, `cti_outputs/nmf_features_grid.png`, `cti_outputs/feature_12_top_corners.png`
  - `cti_outputs/xt_surface.png`, `cti_outputs/team_top_feature.png`
  - `cti_outputs/team_cti_table.png`
  - `cti_outputs/corners_showcase.gif` (multi‑corner animation)

## Corner Animation (Cinema Mode)
- Script: `Final_Project/cti/cti_create_corner_animation.py`
- Shows for each selected corner:
  - xT half-pitch heatmap (zeros black; positive color).
  - Animated tracking (attack/defense + ball).
  - GMM target zones in light grey (legend on left), emphasized at kick freeze.
  - CTI and components (y1–y5); Top NMF feature for attacking team.
  - Logos “Home VS Away” centered below the pitch (compact, non‑overlapping).
- Selection:
  - `--corner-id` for a specific one, or `--count N` to concatenate N distinct corners (default 3).

## Reproducibility
- Install deps: `pip install -r Final_Project/requirements.txt`
- Full pipeline (trains, then runs inference + GIF by default):
  - `python Final_Project/cti_pipeline.py --mode train --max-matches 20`
  - Skip post steps: `--skip-infer`, `--skip-gif`
- Standalone inference: `py -3 Final_Project/cti/cti_infer_cti.py --matches 3 --checkpoint best`
- Standalone GIF:
  - Random corners: `py -3 Final_Project/cti/cti_create_corner_animation.py --count 3 --freeze 6 --fps 10`
  - Specific: `py -3 Final_Project/cti/cti_create_corner_animation.py --corner-id 12345 --freeze 6 --fps 10`
- Regenerate team table with crests: `py -3 Final_Project/regenerate_team_table.py`

## Code Highlights
- Post-processing automation: `cti_pipeline.py` (`_run_subprocess_py`, post steps)
- Crest-preserving table: `cti_nmf_routines.py` (RGBA logos, row alignment)
- GIF engine: selection + canonicalization + overlays
  - `_add_team_ids`, `_get_team_ids_from_map`, `_sc_to_std`, `draw_xt_base`, `draw_gmm_zones`
  - Multi-corner concatenation with separators

## Results (Qualitative)
- NMF reveals routine archetypes (e.g., near‑post overloads, second‑ball ramps).
- Team “Top Feature” table with crests accelerates communication.
- xT surface clarifies where threat accumulates inside the corner-phase zone.
- CTI unifies offense, counter-risk, and ΔxT on a single dial.
- GIF turns the models into an easily digestible story per corner.

## Limitations & Future Work
- Defensive roles (zonal vs man‑marking) left as future work due to labeling demands.
- λ, γ currently defaults; add robust match-level learning once evaluation data are finalized.
- Scale-out: batch inference for a full season and dashboards.

## Selected File Map
- Pipeline: `cti_pipeline.py`
- Corners: `cti_corner_extraction.py`
- GMM & runs: `cti_gmm_zones.py`
- NMF & visuals: `cti_nmf_routines.py`
- xT half-pitch: `cti_xt_surface_half_pitch.py`
- Deep model: `cti_integration.py`
- Inference: `cti/cti_infer_cti.py`
- GIF: `cti/cti_create_corner_animation.py`
- Table regeneration: `regenerate_team_table.py`

## Closing
This project turns corner analysis into an automated production: interpretable features, measurable threat, and compelling visuals. The codebase is structured to run end‑to‑end, export clean artifacts, and generate media that coaches and analysts can act on quickly.

## Update: Sanity Baseline + Label Bug + Class-Imbalance Fix

What I found
- P(shot) and Counter risk were ≈ 0.000 in the team table. Inspecting labels showed the shot detector was too strict: it only checked `event_type == "shot"`. In this dataset, shots are also encoded via `lead_to_shot == True`, and sometimes appear as `event_subtype == 'shot'` or `end_type == 'shot'`.
- A batching quirk concatenated per-graph global features with pooled node embeddings using mismatched shapes, which could suppress head activations.

Fixes implemented
- Robust shot/xG target extraction (cti_integration.py: extract_targets):
  - Shot detector now accepts any of: `event_type=='shot'`, `event_subtype=='shot'`, `lead_to_shot==True`, or `end_type=='shot'`.
  - y2/y4 use max `xthreat` as xG proxy within their windows for the attacking or opposing team respectively.
- Global features batching fix (cti_integration.py → CTIMultiTaskModel.forward):
  - Ensure `global_feats` collates to `[B, 3]` and concatenate safely with `[B, H]`.
- Empirical baseline in inference (cti/cti_infer_cti.py):
  - For each corner, compute window-based y1..y5 directly from events (`*_e`) and `CTI_e`.
  - Emit both model and empirical team tables and a short sanity report with means and correlations: `Final_Project/cti_outputs/sanity_report.txt`.
- Class imbalance for y1/y3 (cti_integration.py + cti_pipeline.py):
  - Switch to `BCEWithLogitsLoss` for y1/y3 and train on logits (still returning probabilities for CTI/inference).
  - Estimate `pos_weight = #neg/#pos` on the train split and pass into Lightning; clamps to [1, 100] to avoid extremes. Printed during training.

Calibration diagnostics
- Added reliability curves for P(shot) and P(counter) using the empirical labels as a quick reference. Files:
  - `Final_Project/cti_outputs/reliability_y1.png`
  - `Final_Project/cti_outputs/reliability_y3.png`
- The sanity report includes aggregate comparisons; the reliability plots visualize calibration across 10 probability bins (bin counts overlaid).
- The Lightning module can optionally update `pos_weight` dynamically at the end of each training epoch using observed prevalence, further stabilizing training on imbalanced data.
- In addition to plots, the sanity report now logs ECE (expected calibration error) and Brier scores for P(shot) and P(counter), enabling quick quantitative checks alongside the visual curves.

How to validate quickly
- Run training on a small subset and check the printed label preview and pos_weights:
  - `python Final_Project/cti_pipeline.py --mode train --max-matches 3 --skip-infer --skip-gif`
  - Look for: `Label preview ...` and `Class balance (train est.) ...` in logs.
- Re-run inference to get both model and empirical summaries:
  - `py -3 Final_Project/cti/cti_infer_cti.py --matches 3 --checkpoint best`
  - Compare `team_cti_table.png` vs `team_cti_table_empirical.png` and read `sanity_report.txt` for alignment (means + correlation of P(shot), ΔxT, CTI).

Why this matters
- The broader detector correctly captures shots across schema variants, so y1/y3 are no longer all-zero. The empirical baseline provides a fast, model‑free yardstick to spot data or training issues. Handling class imbalance with `pos_weight` improves calibration and prevents the model from collapsing to a trivial “no‑shot” solution.

