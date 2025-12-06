"""
Author: Tiago
Date: 2025-12-04
Description: Refactored xT surface builder for corner kick analysis. Implements proper xT formulation with corner-specific phase filtering, direction standardization, and spatial smoothing.
"""

import polars as pl
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter
from cti_corner_extraction import load_events_basic
from cti_paths import DATA_2024, FINAL_PROJECT_DIR

# Configuration
DATA_DIR = DATA_2024
OUTPUT_DIR = FINAL_PROJECT_DIR / "cti_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Pitch dimensions
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# SkillCorner centered coordinates: x∈[-52.5, 52.5], y∈[-34, 34]
SC_X_MIN, SC_X_MAX = -52.5, 52.5
SC_Y_MIN, SC_Y_MAX = -34.0, 34.0

# Expanded state space: last 30m (not just PA) to capture build-up
SC_ATTACK_ZONE_START = 22.5  # 52.5 - 30 = 22.5 (30m from goal)
SC_ATTACK_ZONE_LENGTH = 30.0
GRID_SHAPE = (40, 40)  # default grid; can be overridden via args

# Corner phase window
CORNER_WINDOW_SEC = 15.0

# Smoothing parameters
SMOOTH_SIGMA = 0.9  # Spatial smoothing for counts (meters≈cells proxy)
# Display-only smoothing to make heatmap less blocky without affecting values
VISUAL_SMOOTH_SIGMA = 0.6
PCTL_LOW, PCTL_HIGH = 15.0, 99.5  # visualization dynamic range percentiles


def standardize_direction(df: pl.DataFrame) -> pl.DataFrame:
    """
    Flip coordinates so attacking team always goes left→right (goal at x=52.5 in SC coords).

    Requires 'attacking_right' boolean column indicating if team attacks towards positive x.
    If not present, infers from team/period (assumes teams swap at halftime).
    """
    if "attacking_right" not in df.columns:
        # Infer attacking direction: assume home attacks right in period 1
        # This is a heuristic - adjust based on your data provider
        if "team_id" in df.columns and "period" in df.columns:
            # Get home team (first unique team_id per match)
            df = df.with_columns([
                ((pl.col("period") % 2 == 1)).alias("attacking_right")
            ])
        else:
            print("Warning: Cannot infer attacking direction, assuming all attack right")
            return df.with_columns(pl.lit(True).alias("attacking_right"))

    # Flip coordinates for teams attacking left
    return df.with_columns([
        pl.when(~pl.col("attacking_right"))
          .then(-pl.col("x_start"))
          .otherwise(pl.col("x_start")).alias("x_start"),
        pl.when(~pl.col("attacking_right"))
          .then(-pl.col("y_start"))
          .otherwise(pl.col("y_start")).alias("y_start"),
        pl.when(~pl.col("attacking_right"))
          .then(-pl.col("x_end"))
          .otherwise(pl.col("x_end")).alias("x_end"),
        pl.when(~pl.col("attacking_right"))
          .then(-pl.col("y_end"))
          .otherwise(pl.col("y_end")).alias("y_end"),
    ])


def parse_time_to_seconds(time_str: str) -> float:
    """
    Convert time string 'MM:SS.T' to float seconds.
    Example: '00:04.2' -> 4.2
    """
    if not isinstance(time_str, str) or ':' not in time_str:
        return 0.0
    try:
        parts = time_str.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    except:
        return 0.0


def filter_corner_phases(events_df: pl.DataFrame, corners_df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter events to corner phases only.

    Keep events from: corner taken → +15 seconds, same team only.
    This captures corner delivery + immediate second balls.
    """
    # Check required columns (use time_start instead of event_time)
    if "time_start" not in events_df.columns:
        print("Warning: events_df missing time_start column, skipping phase filtering")
        return events_df

    # Convert time_start to seconds for events
    events_with_time = events_df.with_columns([
        pl.col("time_start").map_elements(parse_time_to_seconds, return_dtype=pl.Float64).alias("time_sec")
    ])

    # Create corner windows from frame_start (use frames since time_start_s may not exist)
    # Assume 25 FPS
    if "frame_start" in corners_df.columns:
        corner_windows = (
            corners_df
            .select(["match_id", "team_id", "period", "frame_start"])
            .unique()
            .with_columns([
                (pl.col("frame_start") / 25.0).alias("t_start"),
                ((pl.col("frame_start") / 25.0) + CORNER_WINDOW_SEC).alias("t_end")
            ])
        )
    else:
        print("Warning: corners_df missing required columns, skipping phase filtering")
        return events_df

    # Join events with corner windows
    events_corner = (
        events_with_time
        .join(
            corner_windows,
            on=["match_id", "period"],
            how="inner"
        )
        .filter(
            (pl.col("time_sec") >= pl.col("t_start")) &
            (pl.col("time_sec") <= pl.col("t_end")) &
            (pl.col("team_id") == pl.col("team_id_right"))  # Same team as corner
        )
        .drop(["t_start", "t_end", "time_sec", "team_id_right", "frame_start_right"])
    )

    return events_corner


def simple_xg_from_xy(x_std: float, y_std: float) -> float:
    """
    xG model using distance and angle to goal.

    Goal center at (105, 34) in standard coordinates.
    Uses logistic regression: xG = 1 / (1 + exp(-z))
    where z = intercept - dist_coef*distance + angle_coef*angle

    Coefficients are illustrative; ideally calibrate to your shot data.
    """
    # Goal center
    goal_x, goal_y = 105.0, 34.0

    dx = goal_x - x_std
    dy = goal_y - y_std
    dist = np.hypot(dx, dy)

    # Angle: measure of "opening" to goal (simplified)
    # goal_width = 7.32m
    angle = np.arctan2(7.32 / 2.0, dist)  # radians

    # Logistic model (coefficients tuned for illustrative purposes)
    z = -3.0 - 0.12 * dist + 4.0 * angle
    xg = 1.0 / (1.0 + np.exp(-z))

    return float(np.clip(xg, 0.01, 0.99))


def classify_event_action(end_type: str, pass_outcome: str = None) -> str:
    """
    Classify event into move/shot/loss for xT.

    For SkillCorner data:
    - end_type indicates what happened (pass, shot, tackle, etc.)
    - pass_outcome indicates success/failure

    move: Successful passes/carries that continue possession
    shot: Shots (terminate chain, receive xG reward)
    loss: Failed passes, tackles, interceptions (terminate chain, no reward)
    """
    end_type_lower = str(end_type).lower() if end_type else ""
    outcome_lower = str(pass_outcome).lower() if pass_outcome else ""

    # Shot events
    if "shot" in end_type_lower:
        return "shot"

    # Loss events (terminate possession)
    loss_keywords = ["tackle", "interception", "clearance", "foul", "aerial", "challenge"]
    if any(kw in end_type_lower for kw in loss_keywords):
        return "loss"

    # Pass outcome handling
    if "pass" in end_type_lower:
        if outcome_lower and "successful" in outcome_lower:
            return "move"
        elif outcome_lower and any(term in outcome_lower for term in ["incomplete", "offside", "out"]):
            return "loss"
        elif not outcome_lower or outcome_lower == "none":
            # No outcome → assume successful move
            return "move"
        else:
            return "loss"

    # Carry/dribble
    if any(term in end_type_lower for term in ["carry", "dribble"]):
        return "move"

    # Unknown events treated as loss to be conservative
    if "unknown" in end_type_lower or not end_type:
        return "loss"

    # Default: treat as move
    return "move"


def extract_action_sequences(events_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract event sequences with action classifications.

    Returns DataFrame with: [x_start, y_start, x_end, y_end, action_type, xg]
    """
    # Check required columns
    required_cols = ["x_start", "y_start", "x_end", "y_end"]
    if not all(col in events_df.columns for col in required_cols):
        print("Warning: Events missing required coordinate columns")
        return pl.DataFrame()

    # Filter valid coordinates
    events_valid = events_df.filter(
        (pl.col("x_start").is_not_null()) &
        (pl.col("y_start").is_not_null()) &
        (pl.col("x_end").is_not_null()) &
        (pl.col("y_end").is_not_null())
    )

    # Keep attacking third (last 30m in SC coords: x >= 22.5)
    events_attacking = events_valid.filter(pl.col("x_start") >= SC_ATTACK_ZONE_START)

    # Classify actions
    action_types = []
    xg_values = []

    for row in events_attacking.iter_rows(named=True):
        # Use end_type and pass_outcome from SkillCorner data
        end_type = row.get("end_type", "")
        pass_outcome = row.get("pass_outcome", "")
        action = classify_event_action(end_type, pass_outcome)
        action_types.append(action)

        # Compute xG for shots
        if action == "shot":
            # Try xshot columns (SkillCorner provides xshot_player_possession_*)
            xg = row.get("xshot_player_possession_start", None) or row.get("xshot_player_possession_max", None)
            if xg is None or xg == 0:
                # Use model fallback
                x_sc, y_sc = row["x_start"], row["y_start"]
                x_std = x_sc + 52.5
                y_std = y_sc + 34.0
                xg = simple_xg_from_xy(x_std, y_std)
            xg_values.append(xg)
        else:
            xg_values.append(0.0)

    return events_attacking.select([
        "x_start", "y_start", "x_end", "y_end"
    ]).with_columns([
        pl.Series("action_type", action_types),
        pl.Series("xg", xg_values)
    ])


def discretize_position(x: float, y: float) -> tuple[int, int]:
    """
    Convert position to grid indices (attacking zone).

    Input: SkillCorner coordinates (x: 22.5-52.5, y: -34..34)
    Output: Grid indices (0-39, 0-39) for 40x40 grid
    """
    # Normalize to attack zone range
    x_norm = max(0.0, min(SC_ATTACK_ZONE_LENGTH, x - SC_ATTACK_ZONE_START))
    y_norm = max(0.0, min(SC_Y_MAX - SC_Y_MIN, y - SC_Y_MIN))

    n_x, n_y = GRID_SHAPE

    # Map to grid indices
    grid_x = int((x_norm / SC_ATTACK_ZONE_LENGTH) * n_x)
    grid_y = int((y_norm / (SC_Y_MAX - SC_Y_MIN)) * n_y)

    # Clamp to valid range
    grid_x = min(grid_x, n_x - 1)
    grid_y = min(grid_y, n_y - 1)

    return grid_x, grid_y


def build_action_counts(actions: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build cell-level action frequency counts.

    Returns:
        counts_shot[i,j]: Number of shots from cell (i,j)
        xg_sum[i,j]: Sum of xG values for shots from cell (i,j)
        counts_loss[i,j]: Number of loss events from cell (i,j)
        counts_move[i,j,x,y]: Number of moves from cell (i,j) to cell (x,y)
    """
    n_x, n_y = GRID_SHAPE

    counts_shot = np.zeros((n_x, n_y))
    xg_sum = np.zeros((n_x, n_y))
    counts_loss = np.zeros((n_x, n_y))
    counts_move = np.zeros((n_x, n_y, n_x, n_y))

    for row in actions.iter_rows(named=True):
        x_start = row["x_start"]
        y_start = row["y_start"]
        x_end = row["x_end"]
        y_end = row["y_end"]
        action_type = row["action_type"]
        xg = row["xg"]

        # Only count actions starting in attack zone
        if x_start < SC_ATTACK_ZONE_START:
            continue

        gx, gy = discretize_position(x_start, y_start)

        if action_type == "shot":
            counts_shot[gx, gy] += 1
            xg_sum[gx, gy] += xg
        elif action_type == "loss":
            counts_loss[gx, gy] += 1
        elif action_type == "move":
            # Allow moves to any cell
            if x_end >= SC_ATTACK_ZONE_START:
                gx_end, gy_end = discretize_position(x_end, y_end)
                counts_move[gx, gy, gx_end, gy_end] += 1
            else:
                # Move exits zone - treat as loss
                counts_loss[gx, gy] += 1

    return counts_shot, xg_sum, counts_loss, counts_move


def smooth_counts(
    counts_shot: np.ndarray,
    xg_sum: np.ndarray,
    counts_loss: np.ndarray,
    counts_move: np.ndarray,
    sigma: float = SMOOTH_SIGMA
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply spatial Gaussian smoothing to borrow strength from neighbors.

    This is more principled than uniform Dirichlet priors over 1600 cells.
    """
    cs = gaussian_filter(counts_shot, sigma=sigma, mode="nearest")
    xs = gaussian_filter(xg_sum, sigma=sigma, mode="nearest")
    cl = gaussian_filter(counts_loss, sigma=sigma, mode="nearest")

    # For 4D move matrix, smooth each origin cell's destination distribution
    cm = np.zeros_like(counts_move)
    for i in range(counts_move.shape[0]):
        for j in range(counts_move.shape[1]):
            cm[i, j] = gaussian_filter(counts_move[i, j], sigma=sigma, mode="nearest")

    return cs, xs, cl, cm


def build_xT_matrices(
    counts_shot: np.ndarray,
    xg_sum: np.ndarray,
    counts_loss: np.ndarray,
    counts_move: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build xT probability and reward matrices.

    No artificial priors for empty cells - use spatial smoothing instead.

    Returns:
        P_shot[i,j]: Probability of shot from cell (i,j)
        R_shot[i,j]: Expected xG reward for shot from cell (i,j)
        P_loss[i,j]: Probability of loss from cell (i,j)
        P_move[i,j,x,y]: Conditional probability of moving to (x,y) given move from (i,j)
    """
    n_x, n_y = GRID_SHAPE

    total_moves = counts_move.sum(axis=(2, 3))
    counts_total = counts_shot + counts_loss + total_moves

    # Outcome probabilities (no mass for unseen states)
    P_shot = np.divide(
        counts_shot,
        counts_total,
        out=np.zeros_like(counts_shot),
        where=counts_total > 0
    )

    P_loss = np.divide(
        counts_loss,
        counts_total,
        out=np.zeros_like(counts_loss),
        where=counts_total > 0
    )

    # Conditional move distribution (only where moves > 0)
    P_move = np.divide(
        counts_move,
        total_moves[..., None, None],
        out=np.zeros_like(counts_move),
        where=total_moves[..., None, None] > 0
    )

    # Shot rewards: mean xG per cell
    R_shot = np.divide(
        xg_sum,
        counts_shot,
        out=np.zeros_like(xg_sum),
        where=counts_shot > 0
    )

    # For cells with no shots, use model-based fallback
    for i in range(n_x):
        for j in range(n_y):
            if counts_shot[i, j] == 0:
                # Cell center in SC coords
                x_cell_sc = SC_ATTACK_ZONE_START + (i + 0.5) * (SC_ATTACK_ZONE_LENGTH / n_x)
                y_cell_sc = SC_Y_MIN + (j + 0.5) * ((SC_Y_MAX - SC_Y_MIN) / n_y)

                # Convert to standard coords
                x_cell_std = x_cell_sc + 52.5
                y_cell_std = y_cell_sc + 34.0

                R_shot[i, j] = simple_xg_from_xy(x_cell_std, y_cell_std)

    return P_shot, R_shot, P_loss, P_move


def validate_probabilities(
    P_shot: np.ndarray,
    P_loss: np.ndarray,
    P_move: np.ndarray,
    tol: float = 1e-8
) -> bool:
    """
    Validate probability mass conservation and non-negativity.

    Returns True if all checks pass.
    """
    issues = []

    # Check move mass non-negativity
    P_move_total = 1.0 - P_shot - P_loss
    if np.any(P_move_total < -tol):
        n_negative = np.sum(P_move_total < -tol)
        min_val = P_move_total.min()
        issues.append(f"Negative move mass in {n_negative} cells (min={min_val:.6f})")

    # Check P_move row normalization (where move mass > 0)
    row_sums = np.sum(P_move, axis=(2, 3))
    active_cells = P_move_total > tol
    if np.any(active_cells):
        max_error = np.max(np.abs(row_sums[active_cells] - 1.0))
        if max_error > tol:
            issues.append(f"P_move rows not normalized (max error={max_error:.6f})")

    # Check bounds
    if np.any(P_shot < 0) or np.any(P_shot > 1):
        issues.append(f"P_shot out of bounds: [{P_shot.min():.6f}, {P_shot.max():.6f}]")
    if np.any(P_loss < 0) or np.any(P_loss > 1):
        issues.append(f"P_loss out of bounds: [{P_loss.min():.6f}, {P_loss.max():.6f}]")

    if issues:
        print("WARNING: Probability validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("OK Probability validation passed")
    return True


def value_iteration_vectorized(
    P_shot: np.ndarray,
    R_shot: np.ndarray,
    P_loss: np.ndarray,
    P_move: np.ndarray,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    max_iterations: int = 2000
) -> tuple[np.ndarray, int, bool]:
    """
    Vectorized value iteration with einsum for efficiency.

    Bellman equation:
        V(s) = P_shot(s) * R_shot(s) + P_move_total(s) * Σ P(s'|s,move) * V(s')

    Returns:
        V: Value function (xT grid)
        n_iterations: Number of iterations run
        converged: Whether convergence was achieved
    """
    n_x, n_y = P_shot.shape
    V = np.zeros((n_x, n_y))

    converged = False
    for iteration in range(max_iterations):
        # Expected future value from moves: Σ_s' P(s'|s,move) * V(s')
        # Using einsum for efficient tensor contraction
        future_value = np.einsum('ijxy,xy->ij', P_move, V)

        # Bellman update
        P_move_total = np.clip(1.0 - P_shot - P_loss, 0.0, 1.0)
        V_new = P_shot * R_shot + P_move_total * gamma * future_value

        # Check convergence
        delta = np.max(np.abs(V_new - V))
        V = V_new

        if delta < epsilon:
            converged = True
            print(f"OK Value iteration converged after {iteration + 1} iterations (delta={delta:.6f})")
            break

    if not converged:
        print(f"Warning: Value iteration did not converge after {max_iterations} iterations (delta={delta:.6f})")

    # Validate boundedness (xT should be ≤ 1 since it's an expected goal probability)
    if V.max() > 1.0 + epsilon:
        print(f"WARNING: xT values exceed 1.0 (max={V.max():.4f})")

    return V, iteration + 1, converged


def visualize_xt_surface(
    xt_grid: np.ndarray,
    converged: bool,
    n_iterations: int,
    output_path: Path,
    n_actions: int,
    draw_flow: bool = True,
):
    """
    Visualize xT surface on half pitch with improvements:
    - Quantile stretch instead of hard threshold
    - Contour lines for xT isolines
    - Grid cell edges for clarity
    """
    pitch = Pitch(
        pitch_type='custom',
        pitch_length=PITCH_LENGTH,
        pitch_width=PITCH_WIDTH,
        half=True,
        line_zorder=2
    )
    fig, ax = pitch.draw(figsize=(10, 8))
    # Dark outer background
    fig.patch.set_facecolor('#313332')

    n_x, n_y = GRID_SHAPE
    max_xt = xt_grid.max()
    min_xt = xt_grid.min()
    mean_xt = xt_grid.mean()

    # Extent in standard coords: attack zone is [75, 105] × [0, 68]
    x_start_std = SC_ATTACK_ZONE_START + 52.5  # 75.0
    x_end_std = SC_X_MAX + 52.5  # 105.0

    if max_xt > 0:
        # Display smoothing (does not affect saved grid)
        xt_disp = gaussian_filter(xt_grid, sigma=VISUAL_SMOOTH_SIGMA, mode='nearest')

        # Dynamic range from percentiles to avoid outliers dominating
        finite_vals = xt_disp[xt_disp > 0]
        vmin = float(np.percentile(finite_vals, PCTL_LOW)) if finite_vals.size else min_xt
        vmax = float(np.percentile(finite_vals, PCTL_HIGH)) if finite_vals.size else max_xt
        vmin = max(vmin, 0.0)
        vmax = max(vmax, vmin + 1e-6)

        # Do NOT mask low values; keep full field colored (clip to vmin)
        data_to_plot = xt_disp

        cmap = plt.cm.viridis
        cmap.set_bad(color='lightgray', alpha=0.1)

        # Heatmap
        im = ax.imshow(
            data_to_plot.T,
            extent=[x_start_std, x_end_std, 0, 68],
            origin='lower',
            cmap=cmap,
            alpha=0.85,
            aspect='auto',
            interpolation='bicubic',
            vmin=vmin,
            vmax=vmax,
        )

        # Contour lines by percentile levels
        if vmax > 0.01:
            X = np.linspace(x_start_std, x_end_std, n_x)
            Y = np.linspace(0, 68, n_y)
            pct_levels = [70, 80, 90, 95, 98]
            levels = [np.percentile(finite_vals, p) for p in pct_levels if finite_vals.size]

            if levels:
                CS = ax.contour(
                    X, Y, xt_disp.T,
                    levels=levels,
                    colors='white',
                    linewidths=1.0,
                    alpha=0.6,
                    linestyles='dashed'
                )
                ax.clabel(CS, inline=True, fontsize=8, fmt='%.3f')

        # Optional: flow field showing gradient of xT (expected progress)
        if draw_flow and vmax > 0:
            gx, gy = np.gradient(xt_disp.T)
            step = max(1, n_x // 20)
            Xq = np.linspace(x_start_std, x_end_std, n_x)[::step]
            Yq = np.linspace(0, 68, n_y)[::step]
            U = gx[::step, ::step]
            V = gy[::step, ::step]
            # Normalize arrows for visibility
            mag = np.hypot(U, V)
            nz = mag > 0
            U[nz] /= mag[nz]
            V[nz] /= mag[nz]
            ax.quiver(Xq, Yq, U, V, color='k', alpha=0.25, scale=25)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Expected Threat (xT)', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10, colors='w')
        cbar.outline.set_edgecolor('w')
        cbar.set_label('Expected Threat (xT)', color='w')
    else:
        ax.text(90, 34, 'No xT data', ha='center', va='center', fontsize=16, color='gray')

    # Title with diagnostics
    converged_str = "Converged" if converged else "Not converged"
    ax.set_title(
        f"Expected Threat Surface - Corner Analysis ({n_x}×{n_y} grid, {SC_ATTACK_ZONE_LENGTH:.0f}m zone)\n"
        f"{converged_str} ({n_iterations} iterations) | {n_actions:,} actions | "
        f"xT ∈ [{min_xt:.4f}, {max_xt:.4f}] | μ = {mean_xt:.4f}",
        fontsize=11, fontweight='bold', pad=15, color='w'
    )

    # Legend for flow quiver (if drawn)
    if draw_flow:
        from matplotlib.lines import Line2D
        flow = Line2D([0],[0], color='k', lw=1.2, label='Flow (xT gradient)')
        leg = ax.legend(handles=[flow], loc='lower left', frameon=True)
        for txt in leg.get_texts():
            txt.set_color('w')
        # Transparent legend box for dark background
        leg.get_frame().set_alpha(0.0)
        leg.get_frame().set_facecolor('none')
        leg.get_frame().set_edgecolor('none')

    plt.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches='tight')
    print(f"OK Saved xT visualization to {output_path}")
    plt.close()


def xt_interp(xt_grid: np.ndarray, gx: float, gy: float) -> float:
    """
    Bilinear interpolation for smooth xT lookup at fractional grid coordinates.

    Args:
        xt_grid: xT values [n_x, n_y]
        gx, gy: Fractional grid indices

    Returns:
        Interpolated xT value
    """
    n_x, n_y = xt_grid.shape

    # Floor to get integer indices
    i = int(np.floor(gx))
    j = int(np.floor(gy))

    # Fractional parts
    di = gx - i
    dj = gy - j

    # Clamp indices
    i = np.clip(i, 0, n_x - 1)
    j = np.clip(j, 0, n_y - 1)
    i1 = np.clip(i + 1, 0, n_x - 1)
    j1 = np.clip(j + 1, 0, n_y - 1)

    # Bilinear interpolation
    v = (
        (1 - di) * (1 - dj) * xt_grid[i, j] +
        di * (1 - dj) * xt_grid[i1, j] +
        (1 - di) * dj * xt_grid[i, j1] +
        di * dj * xt_grid[i1, j1]
    )

    return float(v)


def compute_delta_xt_interp(ball_positions: pl.DataFrame, xt_grid: np.ndarray) -> float:
    """
    Compute cumulative ΔxT along ball trajectory with bilinear interpolation.

    Args:
        ball_positions: DataFrame with ['x', 'y'] in SkillCorner coords
        xt_grid: xT values [n_x, n_y]

    Returns:
        Cumulative ΔxT = sum(xt_next - xt_curr)
    """
    if not isinstance(ball_positions, pl.DataFrame) or "x" not in ball_positions.columns:
        return 0.0

    if ball_positions.height < 2:
        return 0.0

    n_x, n_y = GRID_SHAPE
    xt_values = []

    for row in ball_positions.iter_rows(named=True):
        x_sc = row.get("x")
        y_sc = row.get("y")

        if x_sc is None or y_sc is None:
            xt_values.append(0.0)
            continue

        # Convert to fractional grid coordinates
        x_norm = max(0.0, min(SC_ATTACK_ZONE_LENGTH, x_sc - SC_ATTACK_ZONE_START))
        y_norm = max(0.0, min(SC_Y_MAX - SC_Y_MIN, y_sc - SC_Y_MIN))

        gx = (x_norm / SC_ATTACK_ZONE_LENGTH) * n_x
        gy = (y_norm / (SC_Y_MAX - SC_Y_MIN)) * n_y

        # Clamp to valid range
        gx = np.clip(gx, 0, n_x - 1)
        gy = np.clip(gy, 0, n_y - 1)

        xt = xt_interp(xt_grid, gx, gy)
        xt_values.append(xt)

    # Cumulative difference
    delta = sum(b - a for a, b in zip(xt_values[:-1], xt_values[1:]))
    return float(delta)


def compute_delta_xt(ball_positions: pl.DataFrame, xt_grid: np.ndarray) -> float:
    """Compatibility wrapper expected by cti_integration.

    Delegates to compute_delta_xt_interp (bilinear interpolated ΔxT).
    """
    return compute_delta_xt_interp(ball_positions, xt_grid)


def main():
    print("=" * 70)
    print("Building xT Surface - Refactored Implementation")
    print("Features:")
    print("  - Corner phase filtering (15s windows)")
    print("  - Direction standardization")
    print("  - Spatial Gaussian smoothing")
    print("  - Angle+distance xG model")
    print("  - Vectorized value iteration")
    print("  - Expanded state space (30m attacking zone)")
    print("=" * 70)

    # Load corners dataset
    corners_path = OUTPUT_DIR / "corners_dataset.parquet"
    if not corners_path.exists():
        print(f"Error: {corners_path} not found")
        print("Run corner extraction first!")
        return

    corners_df = pl.read_parquet(corners_path)
    print(f"\nOK Loaded {corners_df.height} corners from dataset")

    match_ids = corners_df["match_id"].unique().to_list()
    print(f"OK Found {len(match_ids)} unique matches")

    # Load events
    print(f"\nLoading events from {len(match_ids)} matches...")
    all_events = []
    for i, match_id in enumerate(match_ids, 1):
        try:
            events = load_events_basic(match_id)
            if events.height > 0:
                if "match_id" not in events.columns:
                    events = events.with_columns(pl.lit(match_id).alias("match_id"))
                all_events.append(events)
                if i % 10 == 0:
                    print(f"  Loaded {i}/{len(match_ids)} matches...")
        except Exception as e:
            print(f"  Warning: Failed to load match {match_id}: {e}")

    if not all_events:
        print("Error: No events loaded")
        return

    events_df = pl.concat(all_events, how="vertical_relaxed")
    print(f"OK Loaded {events_df.height:,} total events")

    # Filter to corner phases
    print(f"\nFiltering to corner phases ({CORNER_WINDOW_SEC}s windows)...")
    events_corner = filter_corner_phases(events_df, corners_df)
    print(f"OK Filtered to {events_corner.height:,} events in corner phases")

    # Standardize direction
    print("\nStandardizing attacking direction...")
    events_corner = standardize_direction(events_corner)
    print("OK Direction standardized (all teams attack towards x=52.5)")

    # Extract action sequences
    print("\nExtracting action sequences (move/shot/loss)...")
    actions = extract_action_sequences(events_corner)
    print(f"OK Extracted {actions.height:,} actions")

    if actions.height == 0:
        print("Error: No actions found in corner phases")
        return

    # Show action distribution
    action_counts = actions.group_by("action_type").len().sort("action_type")
    print("\nAction distribution:")
    for row in action_counts.iter_rows(named=True):
        pct = 100.0 * row['len'] / actions.height
        print(f"  {row['action_type']:10s}: {row['len']:6,} ({pct:5.1f}%)")

    # Build raw action counts
    print("\nBuilding action frequency counts...")
    counts_shot, xg_sum, counts_loss, counts_move = build_action_counts(actions)
    print(f"OK Built action counts")
    print(f"  Total shots:  {counts_shot.sum():6.0f}")
    print(f"  Total losses: {counts_loss.sum():6.0f}")
    print(f"  Total moves:  {counts_move.sum():6.0f}")

    # Apply spatial smoothing
    print(f"\nApplying spatial smoothing (sigma={SMOOTH_SIGMA})...")
    counts_shot_s, xg_sum_s, counts_loss_s, counts_move_s = smooth_counts(
        counts_shot, xg_sum, counts_loss, counts_move, sigma=SMOOTH_SIGMA
    )
    print("OK Smoothing applied")

    # Build xT matrices
    print("\nBuilding xT probability and reward matrices...")
    P_shot, R_shot, P_loss, P_move = build_xT_matrices(
        counts_shot_s, xg_sum_s, counts_loss_s, counts_move_s
    )
    print(f"OK Built matrices")
    print(f"  Mean P(shot):        {P_shot.mean():.4f}")
    print(f"  Mean P(loss):        {P_loss.mean():.4f}")
    print(f"  Mean R(shot|shot>0): {R_shot[R_shot > 0].mean():.4f}" if np.any(R_shot > 0) else "  Mean R(shot): N/A")

    # Validate probabilities
    print("\nValidating probability matrices...")
    validate_probabilities(P_shot, P_loss, P_move)

    # Run vectorized value iteration
    print("\nRunning vectorized value iteration...")
    xt_grid, n_iter, converged = value_iteration_vectorized(
        P_shot, R_shot, P_loss, P_move
    )
    print(f"OK xT grid computed")
    print(f"  Min xT:  {xt_grid.min():.4f}")
    print(f"  Max xT:  {xt_grid.max():.4f}")
    print(f"  Mean xT: {xt_grid.mean():.4f}")
    print(f"  P90 xT:  {np.percentile(xt_grid, 90):.4f}")

    # Visualize
    print("\nGenerating visualization...")
    output_path = OUTPUT_DIR / "xt_surface.png"
    visualize_xt_surface(xt_grid, converged, n_iter, output_path, actions.height)

    # Save grid and metadata
    np.save(OUTPUT_DIR / "xt_grid_corners.npy", xt_grid)
    print(f"OK Saved xT grid to {OUTPUT_DIR / 'xt_grid_corners.npy'}")

    # Save metadata
    metadata = {
        "grid_shape": GRID_SHAPE,
        "attack_zone_start_sc": SC_ATTACK_ZONE_START,
        "attack_zone_length": SC_ATTACK_ZONE_LENGTH,
        "corner_window_sec": CORNER_WINDOW_SEC,
        "smooth_sigma": SMOOTH_SIGMA,
        "n_corners": corners_df.height,
        "n_matches": len(match_ids),
        "n_actions": actions.height,
        "converged": converged,
        "n_iterations": n_iter,
        "xt_min": float(xt_grid.min()),
        "xt_max": float(xt_grid.max()),
        "xt_mean": float(xt_grid.mean()),
    }

    import json
    with open(OUTPUT_DIR / "xt_grid_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"OK Saved metadata to {OUTPUT_DIR / 'xt_grid_metadata.json'}")

    print("\n" + "=" * 70)
    print("COMPLETE - xT Surface Built Successfully")
    print("=" * 70)
    print("\nKey improvements implemented:")
    print("  [*] Corner-specific phase filtering")
    print("  [*] Direction standardization")
    print("  [*] Spatial Gaussian smoothing (no uniform priors)")
    print("  [*] Angle+distance xG model")
    print("  [*] Vectorized value iteration (einsum)")
    print("  [*] Probability validation checks")
    print("  [*] Improved visualization (quantiles, contours)")
    print("  [*] Bilinear interpolation for deltaXT")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-x", type=int, default=GRID_SHAPE[0])
    ap.add_argument("--grid-y", type=int, default=GRID_SHAPE[1])
    ap.add_argument("--sigma", type=float, default=SMOOTH_SIGMA)
    ap.add_argument("--vis-sigma", type=float, default=VISUAL_SMOOTH_SIGMA)
    ap.add_argument("--window", type=float, default=CORNER_WINDOW_SEC)
    args = ap.parse_args()

    # Allow quick tuning from CLI
    GRID_SHAPE = (args.grid_x, args.grid_y)
    SMOOTH_SIGMA = args.sigma
    VISUAL_SMOOTH_SIGMA = args.vis_sigma
    CORNER_WINDOW_SEC = args.window

    main()
