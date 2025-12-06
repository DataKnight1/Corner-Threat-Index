# CTI Methodology & Insights Guide

This document provides a comprehensive guide to the CTI methodology, from technical label definitions to coach-facing interpretations and presentation strategies.

---

# Part 1: Technical Definitions (Labels & Formula)

## 1.1 Coordinate & Time Conventions
- **Tracking**: SkillCorner meters, origin at midfield (x∈[-52.5,52.5], y∈[-34,34]).
- **Events**: Wyscout-like (0–105, 0–68). Delivery zones are binned in this space.
- **FPS**: 25 frames/sec. Windows are defined in seconds and converted to frames.
- **Corner timestamp**: Default `frame_start / 25`; if an event with `start_type_id ∈ {11,12}` matches same period & frame_start, its `time_start` is used.

## 1.2 Label Definitions

### y1 — P(shot in 0–10s)
Indicator of any attacking-team shot in `[t0, t0+10s]`.
- Event cues (any true → shot): `event_type|event_subtype|end_type == "shot"`, `lead_to_shot == True`, `is_shot == True`.
- Window: frames `[frame_start, frame_start + 10*fps]`, same period, same team_id.

### y2 — Corner danger (xThreat by delivery zone)
Expected threat from the **first touch** of the corner delivery, using a historical model by zone.
- Delivery window: `[t0, t0+3s]` in events (attacking period).
- First touch coordinates `(x,y)` mapped to 4×3 bins.
- Label: `y2 = xthreat_corner(zone)` (fallback 0.05 if unseen).

### y3 — P(counter-attack shot, 0–7s, tracking-enhanced)
Binary counter-attack detection for the defending team.
1) Defending event in `(0,7s]` after corner.
2) Attacking **does not** regain possession within 3s.
3) Ball movement check: Ball crosses midfield **or** advances ≥15m in defending direction.

### y4 — Counter xThreat (10–25s, opponent)
Maximum opponent xThreat in the counter window.
- Window: frames `[frame_start + 10*fps, frame_start + 25*fps]`.
- If event column `xthreat` exists: `y4 = max(xthreat)` for opponent in window, else 0.

### y5 — ΔxT (territorial gain, 0–10s)
Change in expected threat from ball movement in the attacking window using the half-pitch xT grid.
- Label: `y5 = xT(final) - xT(start)` aggregated over the window.

## 1.3 CTI Formula
```
CTI = y1 * y2  -  λ * y3 * y4  +  γ * y5
```
- Default weights: `λ = 0.5`, `γ = 1.0`.
- **Interpretation**:
  - `y1*y2` = offensive success: likelihood of shot × quality of delivery.
  - `λ*y3*y4` = counter-risk: likelihood opponents counter + their xThreat.
  - `γ*y5` = territorial gain bonus from ball progression.

---

# Part 2: Coach-Facing Translation

## 2.1 Interpreting Metrics for Coaches
- **CTI**: “Net corner value” → higher = better overall execution and safety.
- **P(shot)**: “How often we turn corners into shots.”
- **Delivery quality (y2)**: “How dangerous our delivery zones are historically.”
- **Counter risk (y3×y4)**: “How often and how dangerous the opponent’s counters are after our corners.”
- **ΔxT**: “How much territory/pressure we gain even without shooting.”

## 2.2 Actionable Levers
- **Improve y1**: Rehearse short/quick routines, blockers/screens to free shooters.
- **Improve y2**: Target higher-value zones; adjust delivery height/trajectory.
- **Reduce y3/y4**: Lock rest-defense positions (half-spaces, weak side fullback), slower loading of box.
- **Improve y5**: Scripted exits after first/second ball to keep territorial pressure.

---

# Part 3: NMF Routine Analysis (Coach's Guide)

## 3.1 What is NMF?
**Non-negative Matrix Factorization (NMF)** automatically discovers **recurring patterns** (routines) in corner kick data. It identifies the 30 most common combinations of player runs.

## 3.2 Interpreting NMF Outputs
### Feature Grid
- **Blue dots**: Initial position zones (where attackers start).
- **Arrows**: The most important runs in this feature.
- **Application**: Identify your team's style and scout opponents.

### Team Top Feature Table
- Shows each team's most-used corner routine.
- **Usage**:
    - **Opposition Analysis**: "Arsenal uses Feature 7 (second-ball ramp) 40% of the time."
    - **Self-Analysis**: "Are we too predictable?"

## 3.3 Practical Workflow
1.  **Pre-Season**: Select 5-8 features as your playbook.
2.  **Weekly Prep**: Check opponent's top feature and design defensive setup.
3.  **Post-Match**: Analyze if you executed the intended routines.

---

# Part 4: Presenting Insights (Slide Guide)

## 4.1 Slide Deck Structure
1.  **The Discovery**: Show the 30 Features Grid. "We found 30 recurring patterns."
2.  **Real Examples**: Show "Top 10 Corners" for a specific feature (e.g., Feature 12).
3.  **Team Analysis**: Show the Team Feature Table. "Who does what?"
4.  **Success Stories**: Highlight Tottenham (Feature 4, high xT) vs Leicester (Feature 3, low xT).
5.  **Scouting Application**: "Facing Arsenal? Watch out for Feature 12."

## 4.2 Key Insights to Present
- **Execution > Routine**: Feature 3 is used by 8 teams, but Man Utd executes it 3.7x better than Leicester.
- **Diversity**: Top teams use 3-5 routines to avoid predictability.
- **Physics**: Dynamic runs (Feature 4) often outperform static positioning (Feature 3).

## 4.3 Q&A Prep
- **"Can we create new routines?"**: Yes, NMF finds existing patterns. You can innovate and see if new patterns emerge.
- **"How do we choose?"**: Combine NMF (pattern) with CTI (outcome). Choose high-xT routines that fit your players.
