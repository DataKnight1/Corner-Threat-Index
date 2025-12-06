Title: Corner Threat Index (CTI): A Reproducible Pipeline for Threat Estimation From Corner Kicks

Authors: Tiago et al.

Affiliations: Internal Analytics Engineering

Corresponding: n/a

Keywords: football analytics, corner kicks, GMM, NMF, expected threat, graph neural networks, PyTorch Geometric, Polars

Abstract
- We present a code-grounded, end‑to‑end pipeline that quantifies the net expected goal impact of corner kicks. The system combines: (i) GMM zoning of initial/target locations to encode 42‑dimensional run vectors; (ii) NMF to discover recurring routines; (iii) a half‑pitch Expected Threat surface; and (iv) a multi‑task GNN to predict five outcomes (y1–y5) and compute CTI = y1·y2 − λ·y3·y4 + γ·y5. We release artifacts and figures directly from this repository.

1. Introduction
- Corners contribute materially to chance creation. CTI balances offensive payoff, counter‑risk, and threat accumulation to provide a single continuous measure that is useful for scouting, match preparation, and communication.

2. Repository & Artifacts
- Root: `Final_Project/`
- Data artifacts: `Final_Project/cti_data/`
  - corners_dataset.parquet; run_vectors.npy; gmm_zones.pkl; nmf_model.pkl; xt_surface.pkl; predictions.csv; team_top_feature.csv
- Figures: `Final_Project/cti_outputs/`
  - gmm_zones.png; nmf_features_grid.png; feature_12_top_corners.png; xt_surface.png; team_top_feature.png; team_cti_table.png; corners_showcase.gif

3. Data Sources & Ingestion
- Events: Parquet under `PremierLeague_data/2024/dynamic/{match_id}.parquet`
- Tracking: JSON under `PremierLeague_data/2024/tracking/{match_id}.json` (SkillCorner centered meters)
- Windows: setup [-5,0)s, flight [0,2]s, attack [0,10]s, counter (10,25]s (see cti_corner_extraction.py)

4. Data Engineering Architecture
- We separate figures from data artifacts for cleanliness. Canonicalization (cti_gmm_zones.py) reflects trajectories so the attacking team always attacks the top‑right corner.

5. Quality Control
- Gates (cti_corner_extraction.py): ball detected in [-0.5,+2]s; ≤2 missing defenders in [-2,+2]s. Failures are annotated in the corners parquet.

6. Domain‑Specific Filtering
- Corner flag: events with `start_type_id ∈ {11,12}`. Short corners handled by the same windows; policy can be adapted.

7. Data Treatments
- SkillCorner → standard [0,105]×[0,68] m conversions; canonicalization flips; optional Gaussian smoothing for xT visualization.

8. Feature Engineering
- GMM Zones: 6 initial, 15 target (7 active) → 42‑d run vectors.
- NMF: topic discovery over run vectors (rank bounded by sample size; see cti_nmf_routines.fit_nmf_routines).
- xT: half‑pitch surface via value iteration (cti_xt_surface_half_pitch.py) saved to `cti_data/xt_surface.pkl` and visualized in `cti_outputs/xt_surface.png`.

9. CTI Target, Model, and Training
- Heads y1–y5 (cti_integration.py): shot, xG, counter‑shot, opponent xG, ΔxT. CTI = y1·y2 − λ·y3·y4 + γ·y5. λ, γ may be learned over aggregates (module `learn_lambda_gamma`). Model: GraphSAGE encoder, five MLP heads.

10. Validation Protocol
- Group‑wise train/test with held‑out matches; reliability targets for P(shot), P(counter‑shot). Utility measured as uplift in shot rate for top‑decile CTI. For this report we executed a small run to confirm artifact production (see Reproducibility).

11. Negative Results
- Defensive roles classifier was de‑scoped here (future work) and disabled to avoid creating unused artifacts. Early attempts to embed images without RGBA conversion produced tinted crests; fixed in `cti_nmf_routines.py` by converting to RGBA before embedding. An initial animation version anchored to annotated frames could miss the true kick; we canonicalized and tightened tracking selection by frame windows.

12. Reproducibility & Execution
- Minimal commands (Windows):
  - `py -3 Final_Project/cti_pipeline.py --mode train --max-matches 2`
- `py -3 Final_Project/cti/cti_infer_cti.py --matches 3 --checkpoint best`
- `py -3 Final_Project/cti/cti_create_corner_animation.py --count 3 --freeze 6 --fps 10`
- Produced figures (relative):
  - `Final_Project/cti_outputs/gmm_zones.png`
  - `Final_Project/cti_outputs/nmf_features_grid.png`
  - `Final_Project/cti_outputs/xt_surface.png`
  - `Final_Project/cti_outputs/team_top_feature.png`
  - `Final_Project/cti_outputs/team_cti_table.png`
  - `Final_Project/cti_outputs/corners_showcase.gif`

13. Ethical Considerations
- Player anonymization is preserved by not exposing identities beyond team‑level reporting. The system is designed for team performance analysis, not player profiling.

14. Limitations & Future Work
- Full calibration and λ/γ learning are left for a larger run. Defensive roles classifier and SHAP explanations are documented as future work. Expanded ablations (event‑only, tracking‑only) are proposed.

15. Conclusion
- CTI unifies threat generation, counter‑risk, and threat accumulation into a coherent pipeline that produces actionable figures and a multi‑corner GIF directly from this repository.

References
- Spearman W. Quantifying Pitch Control.
- Singh K. Introducing Expected Threat (xT).
- Stats Perform. Making Offensive Play Predictable (GCN whitepaper).
- Sloan SSAC: Routine Inspection: A Playbook for Corner Kicks.
