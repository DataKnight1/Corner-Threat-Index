"""
Author: Tiago
Date: 2025-12-04
Description: CTI NMF routine discovery module. Implements NMF-based topic modeling for corner routines, feature extraction, and routine assignment.
"""

import numpy as np
import polars as pl
from sklearn.decomposition import NMF as SklearnNMF
from typing import Tuple, Optional, Dict
import pickle
from pathlib import Path
from dataclasses import dataclass


@dataclass
@dataclass
class NMFRoutineModel:
    """
    Container for NMF model and decomposition matrices.

    :param model: The fitted sklearn NMF model object.
    :param W: Matrix W (n_corners, n_topics), representing corner weights over topics.
    :param H: Matrix H (n_topics, 42), representing topic run patterns.
    :param n_topics: Number of topics (components).
    :param reconstruction_error: Frobenius norm of the matrix difference.
    """
    model: SklearnNMF
    W: np.ndarray  # (n_corners, n_topics) - corner weights over topics
    H: np.ndarray  # (n_topics, 42) - topic run patterns
    n_topics: int
    reconstruction_error: float



def fit_nmf_routines(
    run_vectors: np.ndarray,
    n_components: int = 30,
    alpha: float = 0.0,
    l1_ratio: float = 0.0,
    max_iter: int = 2000,
    random_state: int = 42,
    row_normalize: bool = True
) -> NMFRoutineModel:
    """
    Fit NMF to discover recurring corner routines as topics.

    From Paper §3.2 & Appendix 2:
    - Decomposes run vector matrix X ≈ WH.
    - W (basis coefficients): How each corner expresses each routine.
    - H (basis vectors): The 30 recurring run patterns (routines).
    - Regularization encourages sparseness (α=1.0).

    :param run_vectors: Array of shape (n_corners, 42) with run encodings.
    :param n_components: Number of routine topics (default 30 from paper).
    :param alpha: Regularization parameter for sparseness.
    :param l1_ratio: L1 vs L2 regularization mix (0.5 = balanced).
    :param max_iter: Maximum iterations for convergence.
    :param random_state: Random seed.
    :param row_normalize: Whether to row-normalize inputs.
    :return: NMFRoutineModel with fitted model and matrices.
    """
    n_corners = run_vectors.shape[0]

    print(f"Fitting NMF for routine discovery...")
    print(f"  Corners: {n_corners:,}")
    print(f"  Run dimensions: {run_vectors.shape[1]}")
    print(f"  Topics: {n_components}")

    # Optional row normalization (stabilizes scale across corners)
    X = run_vectors.copy()
    if row_normalize:
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0.0] = 1.0
        X = X / rs

    # Initialize NMF (less regularization to avoid collapsing to zeros)
    nmf = SklearnNMF(
        n_components=n_components,
        init='nndsvda',  # Non-negative double SVD init (stable)
        solver='cd',  # Coordinate descent (fast)
        alpha_W=alpha,  # W regularization
        alpha_H=alpha,  # H regularization
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0
    )

    # Fit and transform
    W = nmf.fit_transform(X)  # (n_corners, n_topics)
    H = nmf.components_  # (n_topics, 42)

    # Reconstruction error (Frobenius norm)
    reconstruction = W @ H
    denom = np.linalg.norm(X, ord='fro')
    denom = denom if denom > 0 else 1.0
    error = np.linalg.norm(X - reconstruction, ord='fro')

    print(f"OK NMF fitted successfully")
    print(f"  Reconstruction error: {error:.2f}")
    print(f"  Explained variance: {1 - (error / denom):.1%}")

    # Normalize W rows so topic weights sum to 1 per corner (interpretability)
    W_norm = W.copy()
    ws = W_norm.sum(axis=1, keepdims=True)
    ws[ws == 0.0] = 1.0
    W_norm = W_norm / ws

    return NMFRoutineModel(
        model=nmf,
        W=W_norm,
        H=H,
        n_topics=n_components,
        reconstruction_error=error
    )


# === NMF Feature Browser Utilities (consolidated from cti_nmf_feature_browser.py) ===
def load_run_vectors(path: Path) -> np.ndarray:
    """
    Load (N, 42) run vectors saved by the pipeline.

    :param path: Path to the .npy file.
    :return: Numpy array of shape (N, 42).
    :raises ValueError: If the array shape is invalid.
    """
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 42:
        raise ValueError(f"Expected (N, 42) run vectors, got {arr.shape}")
    return arr


def run_nmf_feature_browser(X_42xN: np.ndarray, k: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Alternate NMF routine for feature-browser style, accepting (42, N) input.

    :param X_42xN: Input matrix of shape (42, N).
    :param k: Number of components.
    :param seed: Random seed.
    :return: Tuple containing W (42, K), H (K, N), and reconstruction error.
    """
    from sklearn.decomposition import NMF as _NMF
    nmf = _NMF(
        n_components=k,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        l1_ratio=0.2,
        alpha_W=0.0,
        alpha_H=0.0,
        max_iter=2000,
        random_state=seed,
    )
    W = nmf.fit_transform(X_42xN)  # (42, K)
    H = nmf.components_            # (K, N)
    err = float(nmf.reconstruction_err_)
    return W, H, err


def write_top_corners_per_feature(H_KxN: np.ndarray, corners_df: pl.DataFrame, out_csv: Path, top: int = 10) -> None:
    """
    Emit CSV of top-N corners per feature (topic).

    :param H_KxN: Topic weight matrix (K, N).
    :param corners_df: Polars DataFrame with corner metadata.
    :param out_csv: Output CSV path.
    :param top: Number of top corners to include per feature.
    """
    rows = []
    K, N = H_KxN.shape
    for k in range(K):
        scores = H_KxN[k]
        order = np.argsort(scores)[::-1][:top]
        for rank, idx in enumerate(order, start=1):
            score = float(scores[idx])
            meta = corners_df.row(int(idx), named=True) if idx < len(corners_df) else {}
            rows.append({
                "feature_id": k,
                "rank": rank,
                "score": score,
                "corner_index": int(idx),
                "corner_id": meta.get("corner_id"),
                "match_id": meta.get("match_id"),
            })
    pl.from_dicts(rows).write_csv(out_csv)


def plot_feature_runs(W_42xK: np.ndarray, outdir: Path, top_m: int = 8) -> None:
    """
    Optional small bars showing top contributing runs per feature.

    :param W_42xK: Feature matrix (42, K).
    :param outdir: Output directory path.
    :param top_m: Number of top runs to show.
    """
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    K = W_42xK.shape[1]
    for k in range(K):
        w = W_42xK[:, k]
        idx = np.argsort(w)[::-1][:top_m]
        vals = w[idx]
        plt.figure(figsize=(8, 3.2))
        plt.bar(range(top_m), vals, color="#2C7FB8")
        plt.xticks(range(top_m), [f"r{int(i)+1}" for i in idx], rotation=0)
        plt.ylabel("weight")
        plt.title(f"NMF feature {k}: top {top_m} run types")
        plt.tight_layout()
        plt.savefig(outdir / f"nmf_feature_{k}_top_runs.png", dpi=150)
        plt.close()


def build_corner_samples_for_visuals(
    corners_df: pl.DataFrame,
    events_dict: Dict[int, pl.DataFrame],
    tracking_dict: Dict[int, pl.DataFrame]
) -> list[Dict]:

    """
    Build initial/target position samples per corner for visualization.

    :param corners_df: DataFrame with corners.
    :param events_dict: Dictionary mapping match_id to events DataFrame.
    :param tracking_dict: Dictionary mapping match_id to tracking DataFrame.
    :return: List of dictionaries containing 'corner_idx', 'corner_id', 'match_id', 'initial', 'target'.
    """
    from cti_gmm_zones import (
        extract_initial_positions,
        extract_target_positions,
        build_player_team_map,
    )

    samples: list[Dict] = []
    team_maps: Dict[int, Dict[int, int]] = {}

    for idx, corner in enumerate(corners_df.iter_rows(named=True)):
        mid = corner.get("match_id")
        if mid not in events_dict or mid not in tracking_dict:
            continue
        events_df = events_dict[mid]
        tracking_df = tracking_dict[mid]
        team_map = team_maps.setdefault(mid, build_player_team_map(events_df))

        payload = dict(corner)
        payload["player_team_map"] = team_map
        payload["attacking_team_id"] = corner.get("team_id")
        payload["taker_player_id"] = corner.get("player_in_possession_id")

        try:
            init_df = extract_initial_positions(payload, tracking_df, events_df)
            tgt_df = extract_target_positions(payload, tracking_df, events_df)
        except Exception:
            continue

        if init_df.height == 0 or tgt_df.height == 0:
            continue

        samples.append({
            "corner_idx": idx,
            "corner_id": corner.get("corner_id"),
            "match_id": mid,
            "initial": init_df,
            "target": tgt_df,
        })

    return samples



def assign_routines(
    corner_run_vector: np.ndarray,
    nmf_model: NMFRoutineModel
) -> np.ndarray:
    """
    Assign routine topic weights to a single corner.

    :param corner_run_vector: 42-d run vector for one corner.
    :param nmf_model: Fitted NMFRoutineModel.
    :return: Array of shape (n_topics,) with routine weights.
    """
    # Transform single sample
    weights = nmf_model.model.transform(corner_run_vector.reshape(1, -1))
    return weights.flatten()



def get_top_routines(
    corner_weights: np.ndarray,
    top_k: int = 3
) -> list[Tuple[int, float]]:
    """
    Get top-k routine topics for a corner.

    :param corner_weights: Array of shape (n_topics,) from assign_routines().
    :param top_k: Number of top routines to return (default 3).
    :return: List of (routine_id, weight) tuples, sorted by weight descending.
    """
    top_indices = np.argsort(corner_weights)[-top_k:][::-1]
    return [(int(idx), float(corner_weights[idx])) for idx in top_indices]



def identify_routine_features(
    nmf_model: NMFRoutineModel,
    zone_labels: Optional[list] = None
) -> list[Dict]:
    """
    Interpret each NMF topic as a routine feature.

    From Paper §3.2: "Features are frequently co-occurring runs".

    :param nmf_model: Fitted NMFRoutineModel.
    :param zone_labels: Optional list of 42 zone pair labels e.g., ["1a", "1b", ...].
    :return: List of dicts with routine metadata (routine_id, top_runs, run_labels).
    """
    H = nmf_model.H  # (n_topics, 42)
    n_topics = nmf_model.n_topics

    if zone_labels is None:
        # Generate default labels: 1a, 1b, ..., 6g (6 initial × 7 target)
        zone_labels = []
        for init in range(1, 7):
            for tgt in 'abcdefg':
                zone_labels.append(f"{init}{tgt}")

    routines = []

    for topic_idx in range(n_topics):
        topic_vector = H[topic_idx]  # (42,)

        # Find top contributing runs
        top_run_indices = np.argsort(topic_vector)[-5:][::-1]  # Top 5 runs
        top_runs = [(int(idx), float(topic_vector[idx])) for idx in top_run_indices
                    if topic_vector[idx] > 0.01]  # Threshold for significance

        routine = {
            'routine_id': topic_idx,
            'top_runs': top_runs,
            'run_labels': [zone_labels[idx] for idx, _ in top_runs]
        }

        routines.append(routine)

    return routines


def cluster_corners_by_routines(
    W: np.ndarray,
    n_clusters: int = 10,
    method: str = 'hierarchical'
) -> np.ndarray:

    """
    Cluster corners by their routine mixtures.

    From Paper §3.2: "Distinct corner routines can be identified by
    grouping corners that exhibit similar feature expressions using
    agglomerative hierarchical clustering."

    :param W: Corner weights matrix (n_corners, n_topics).
    :param n_clusters: Number of routine clusters.
    :param method: 'hierarchical' or 'kmeans'.
    :return: Array of cluster labels (n_corners,).
    """
    if method == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    labels = clustering.fit_predict(W)

    print(f"OK Clustered {W.shape[0]} corners into {n_clusters} routine groups")

    return labels


def find_similar_corners(
    target_corner_idx: int,
    W: np.ndarray,
    top_k: int = 10
) -> list[Tuple[int, float]]:

    """
    Find corners with similar routine profiles using cosine similarity.

    :param target_corner_idx: Index of the target corner.
    :param W: Corner weights matrix (n_corners, n_topics).
    :param top_k: Number of similar corners to return.
    :return: List of (corner_idx, similarity_score) tuples, sorted by score.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    target_vector = W[target_corner_idx].reshape(1, -1)
    similarities = cosine_similarity(target_vector, W).flatten()

    # Exclude self
    similarities[target_corner_idx] = -1

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def visualize_routine(
    routine_id: int,
    nmf_model: NMFRoutineModel,
    zone_models=None
) -> 'matplotlib.figure.Figure':

    """
    Visualize a single routine as a run pattern heatmap or arrow plot on the pitch.

    :param routine_id: Topic/routine index to visualize.
    :param nmf_model: Fitted NMFRoutineModel.
    :param zone_models: Optional ZoneModels for precise zone locations.
    :return: Matplotlib Figure showing the routine pattern.
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch

    fig, ax = plt.subplots(figsize=(12, 8))

    pitch = Pitch(
        pitch_type='custom',
        pitch_length=105,
        pitch_width=68,
        pitch_color='#22312b',
        line_color='white'
    )
    pitch.draw(ax=ax)

    # Get routine vector
    routine_vector = nmf_model.H[routine_id]  # (42,)

    # Reshape to (6 initial, 7 target) matrix
    run_matrix = routine_vector.reshape(6, 7)

    # If zone_models provided, plot actual zone locations
    # Otherwise, use generic visualization
    if zone_models is not None:
        init_means = zone_models.gmm_init.means_
        target_means = zone_models.gmm_tgt.means_[zone_models.active_tgt_ids]

        # Draw arrows for significant runs
        for init_idx in range(6):
            for target_idx in range(7):
                weight = run_matrix[init_idx, target_idx]
                if weight > 0.1:  # Threshold for visualization
                    x_start, y_start = init_means[init_idx]
                    x_end, y_end = target_means[target_idx]

                    ax.annotate('',
                                xy=(x_end, y_end),
                                xytext=(x_start, y_start),
                                arrowprops=dict(
                                    arrowstyle='->',
                                    lw=weight*3,  # Linewidth proportional to weight
                                    color='red',
                                    alpha=0.7
                                ))
    else:
        # Generic heatmap visualization
        im = ax.imshow(run_matrix, cmap='Reds', aspect='auto', alpha=0.7)
        plt.colorbar(im, ax=ax, label='Run Weight')

    ax.set_title(f"Routine {routine_id}", fontsize=16)

    return fig




    """
    Visualize all NMF features (topics) as in Figure 3.

    Draws a 5x6 grid of half-pitch subplots; for each feature, plots arrows
    from initial-zone means to active target-zone means for run types with
    weight above the threshold. Initial zone means are plotted as blue dots.

    :param nmf_model: Fitted NMF model.
    :param zone_models: GMM zone models with mean coordinates.
    :param weight_threshold: Threshold to visualize a connection arrow.
    :param max_runs_per_feature: Maximum arrows to draw per subplot.
    :param figsize: Figure size (width, height).
    :param feature_order: Optional array to reorder the subplots.
    :param feature_importance: Optional array of feature importance scores.
    :return: Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch

    H = nmf_model.H  # (n_topics, 42)
    n_topics = nmf_model.n_topics

    rows, cols = 5, 6
    assert rows * cols >= n_topics, "Grid too small for number of topics"

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    # Dark background styling similar to examples
    fig.patch.set_facecolor('#313332')
    axes = axes.flatten()

    pitch = Pitch(
        pitch_type='custom', pitch_length=105, pitch_width=68, half=True,
        pitch_color='#167d3b', line_color='white'
    )

    init_means = zone_models.gmm_init.means_  # SC coords
    tgt_means_all = zone_models.gmm_tgt.means_  # SC coords
    tgt_ids = zone_models.active_tgt_ids
    tgt_means = tgt_means_all[tgt_ids]

    def sc_to_std(arr):
        return np.column_stack((arr[:, 0] + 52.5, arr[:, 1] + 34.0))

    init_std = sc_to_std(init_means)
    tgt_std = sc_to_std(tgt_means)

    for t in range(n_topics):
        ax = axes[t]
        pitch.draw(ax=ax)
        ax.set_facecolor('#313332')

        topic = H[t].reshape(6, 7)

        # Draw initial zone dots
        ax.scatter(init_std[:, 0], init_std[:, 1], s=30, c='#2166ac', zorder=3)

        # Select top runs per feature to avoid visual blobs
        runs = []
        for i in range(6):
            for j in range(7):
                w = float(topic[i, j])
                if w >= weight_threshold:
                    runs.append((w, i, j))
        if runs:
            runs.sort(reverse=True)
            runs = runs[:max_runs_per_feature]
            wmax = runs[0][0] if runs[0][0] > 0 else 1.0
            for w, i, j in runs:
                rel = max(0.0, min(1.0, w / wmax))
                lw = 1.2 + 4.0 * rel
                x0, y0 = init_std[i]
                x1, y1 = tgt_std[j]
                ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle='-|>', lw=lw,
                                            color='#2166ac', alpha=0.9,
                                            shrinkA=0, shrinkB=0,
                                            connectionstyle='arc3,rad=-0.25'))

        ax.text(2, 3, f"{t+1}", color='white', fontsize=10, ha='left', va='bottom')
        ax.set_title("")

    # Hide any unused subplots
    for k in range(n_topics, rows * cols):
        axes[k].axis('off')

    # Legend below the grid
    from matplotlib.lines import Line2D
    dot = Line2D([0], [0], marker='o', color='w', label='Initial zone', markerfacecolor='#2166ac', markersize=6)
    arr = Line2D([0], [0], color='#2166ac', lw=2.5, label='Run trajectory', marker='', linestyle='-')
    leg = fig.legend(handles=[dot, arr], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    for txt in leg.get_texts():
        txt.set_color('w')
    fig.suptitle('Thirty features (frequently co-occurring runs) from NMF', fontsize=14, color='w')
    plt.subplots_adjust(bottom=0.08)
    fig.tight_layout()
    return fig




    """
    Visualize top-N corners that most strongly exhibit a given feature (topic).

    Draws a 2x5 grid of half-pitches with initial positions (red dots) and
    dashed lines from initial to target for active players.

    :param feature_id: The ID of the NMF feature to visualize.
    :param nmf_model: Fitted NMF model.
    :param corner_samples: List of corner sample data for plotting.
    :param zone_models: Zone models (for GMM probabilities).
    :param top_n: Number of corners to visualize (default 10).
    :param corner_index_map: Optional mapping from model indices to global corner indices.
    :return: Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch

    # Select top-N corners by feature weight
    weights = nmf_model.W[:, feature_id]
    order = np.argsort(weights)[::-1]
    selected = []
    for idx in order:
        # Map to global corner index if provided
        orig_idx = int(corner_index_map[idx]) if corner_index_map is not None else int(idx)
        # Find matching sample for this global corner index
        sample = next((s for s in corner_samples if s['corner_idx'] == orig_idx), None)
        if sample is None:
            continue
        selected.append(sample)
        if len(selected) >= top_n:
            break

    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(18, 7))
    fig.patch.set_facecolor('#313332')
    axes = axes.flatten()
    pitch = Pitch(
        pitch_type='custom', pitch_length=105, pitch_width=68, half=True,
        pitch_color='#167d3b', line_color='white'
    )

    for plot_num, (ax, sample) in enumerate(zip(axes, selected), start=1):
        pitch.draw(ax=ax)
        ax.set_facecolor('#313332')
        init = sample['initial']
        tgt = sample['target']

        init_np = init.select(['x_m', 'y_m']).to_numpy()
        tgt_np = tgt.select(['x_m', 'y_m']).to_numpy()

        # Determine active attackers by target responsibilities
        probs = zone_models.gmm_tgt.predict_proba(tgt_np)[:, zone_models.active_tgt_ids]
        active_mask = (probs.max(axis=1) >= 0.30)

        # Align by player_id
        init_r = init.rename({'x_m': 'x0', 'y_m': 'y0'})
        tgt_r = tgt.rename({'x_m': 'x1', 'y_m': 'y1'})
        joined = init_r.join(tgt_r, on='player_id', how='inner')

        if joined.height == 0:
            continue

        # Convert to standard coords for plotting
        x0 = joined['x0'].to_numpy() + 52.5
        y0 = joined['y0'].to_numpy() + 34.0
        x1 = joined['x1'].to_numpy() + 52.5
        y1 = joined['y1'].to_numpy() + 34.0

        # Background: initial dots for all attackers
        ax.scatter(x0, y0, s=12, c='#d7301f', zorder=3)

        # Dashed lines to targets (only active ones if mask length aligns)
        for i in range(min(len(x0), len(x1))):
            ax.plot([x0[i], x1[i]], [y0[i], y1[i]], ls='--', lw=1.2,
                    color='#d7301f', alpha=0.8)

        # Add subplot numeration
        ax.text(54, 3, f'({plot_num})', color='white', fontsize=11,
                fontweight='bold', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

    # If fewer than top_n available, hide remaining axes
    for k in range(len(selected), rows * cols):
        axes[k].axis('off')

    # Legend below the figure
    from matplotlib.lines import Line2D
    dot = Line2D([0], [0], marker='o', color='w', label='Initial position', markerfacecolor='#d7301f', markersize=6)
    dsh = Line2D([0], [0], color='#d7301f', lw=1.5, ls='--', label='Run trajectory')
    leg = fig.legend(handles=[dot, dsh], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    for txt in leg.get_texts():
        txt.set_color('w')
    fig.suptitle(f"Top {top_n} corners exhibiting feature {feature_id + 1}", fontsize=14, color='w')
    plt.subplots_adjust(bottom=0.10)
    fig.tight_layout()
    return fig




    """
    Compute each team's most used feature (topic) from NMF weights.

    Assumes rows in `corners_df` align with rows used to fit NMF (same order).
    Falls back to `team_id` if `team_name` missing.

    :param corners_df: Corners DataFrame.
    :param nmf_model: Fitted NMF model.
    :param team_name_col: Column name for team names (default 'team_name').
    :param team_name_map: Optional dict to map team IDs to names.
    :param extra_metrics: Optional dict of extra metrics per team.
    :return: DataFrame with columns [team, n_corners, top_feature_id, top_weight].
    """
    n = nmf_model.W.shape[0]
    teams = corners_df[:n]
    # Try to pick a friendly team name column
    fallback_cols = [team_name_col, 'team', 'team_short_name', 'team_long_name', 'team_id']
    col_name = next((c for c in fallback_cols if c in teams.columns), 'team_id')
    team_keys = teams[col_name].to_list()

    # Aggregate average weights per team
    from collections import defaultdict
    accum = defaultdict(lambda: np.zeros(nmf_model.n_topics, dtype=float))
    counts = defaultdict(int)
    for i, key in enumerate(team_keys):
        accum[key] += nmf_model.W[i]
        counts[key] += 1

    rows = []
    for key, vec in accum.items():
        avg = vec / max(1, counts[key])
        top_idx = int(np.argmax(avg))
        team_display = team_name_map.get(key, key) if team_name_map is not None else key
        row = {
            'team': team_display,
            'n_corners': counts[key],
            'top_feature_id': top_idx + 1,
            'top_weight': float(avg[top_idx])
        }
        # Attach extra metrics if provided (e.g., xT)
        if extra_metrics and key in extra_metrics:
            row.update(extra_metrics[key])
        rows.append(row)

    return pl.from_dicts(rows).sort(['n_corners'], descending=True)



def visualize_team_top_features_grid(
    team_table: pl.DataFrame,
    nmf_model: NMFRoutineModel,
    zone_models,
    logo_dir: Path | None = None,
    rows: int = 4,
    cols: int = 5,
    weight_threshold: float = 0.01,
    max_runs_per_feature: int = 30,
    title: str | None = 'Teams and Their Top Corner Feature',
):
    """
    Create a 4x5 grid: Each subplot shows the team's logo and
    the half‑pitch visualization of that team's top NMF feature.

    :param team_table: DataFrame with columns ['team', 'top_feature_id', ...].
    :param nmf_model: Trained NMF routine model (provides H).
    :param zone_models: ZoneModels with GMM means used for plotting.
    :param logo_dir: Directory containing `<sanitized-team-name>.png`.
    :param rows: Grid rows (default 4).
    :param cols: Grid cols (default 5).
    :param weight_threshold: Min weight for a run to be drawn.
    :param max_runs_per_feature: Cap number of arrows for legibility.
    :param title: Figure title.
    :return: Matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image
    import numpy as np

    teams = team_table.to_dicts()
    n = len(teams)
    rows = rows or 4
    cols = cols or 5
    assert rows * cols >= n, "Grid too small for number of teams"

    figsize = (cols * 3.6, rows * 3.0)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.patch.set_facecolor('#313332')
    axes = np.array(axes).reshape(rows, cols)

    # Pitch set up
    pitch = Pitch(
        pitch_type='custom', pitch_length=105, pitch_width=68, half=True,
        pitch_color='#167d3b', line_color='white'
    )

    init_means = zone_models.gmm_init.means_  # SC coords
    tgt_means_all = zone_models.gmm_tgt.means_  # SC coords
    tgt_ids = zone_models.active_tgt_ids
    tgt_means = tgt_means_all[tgt_ids]

    def sc_to_std(arr):
        return np.column_stack((arr[:, 0] + 52.5, arr[:, 1] + 34.0))

    init_std = sc_to_std(init_means)
    tgt_std = sc_to_std(tgt_means)

    def sanitize(name: str) -> str:
        return ''.join(ch.lower() for ch in str(name) if ch.isalnum())

    H = nmf_model.H

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.set_facecolor('#313332')
        ax.axis('off')

        if idx >= n:
            ax.set_visible(False)
            continue

        team = teams[idx]
        team_name = str(team.get('team'))
        feat_id_1b = int(team.get('top_feature_id', 1))
        feat_id = max(0, min(nmf_model.n_topics - 1, feat_id_1b - 1))

        # Draw pitch and feature runs
        pitch.draw(ax=ax)
        topic = H[feat_id].reshape(6, 7)

        # initial dots
        ax.scatter(init_std[:, 0], init_std[:, 1], s=20, c='#2166ac', zorder=3)

        # Draw initial zone dots (small) and arrows for this team's top feature
        ax.scatter(init_std[:, 0], init_std[:, 1], s=20, c='#2166ac', zorder=3)

        runs = []
        for i in range(6):
            for j in range(7):
                w = float(topic[i, j])
                if w >= weight_threshold:
                    runs.append((w, i, j))
        if runs:
            runs.sort(reverse=True)
            runs = runs[:max_runs_per_feature]
            wmax = runs[0][0] if runs[0][0] > 0 else 1.0
            for w, i0, j0 in runs:
                rel = max(0.0, min(1.0, w / wmax))
                lw = 1.0 + 3.5 * rel
                x0, y0 = init_std[i0]
                x1, y1 = tgt_std[j0]
                ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle='-|>', lw=lw,
                                            color='#2166ac', alpha=0.9,
                                            shrinkA=0, shrinkB=0,
                                            connectionstyle='arc3,rad=-0.25'))

        ax.text(2, 3, f"{feat_id_1b}", color='white', fontsize=10, ha='left', va='bottom')
        ax.set_title("")

        # Team name + feature id label
        ax.text(1.0, 1.02, f"{team_name}", color='w', fontsize=9,
                ha='right', va='bottom', transform=ax.transAxes)
        ax.text(0.02, 0.02, f"Feature {feat_id_1b}", color='w', fontsize=9,
                ha='left', va='bottom', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.4))

        # Add team logo in the upper-left corner of the axes
        if logo_dir is not None:
            logo_path = (logo_dir / f"{sanitize(team_name)}.png")
            try:
                if logo_path.exists():
                    img = Image.open(logo_path).convert('RGBA')
                    # Resize to a consistent pixel height
                    aspect = img.width / max(1, img.height)
                    target_h = 60
                    target_w = max(1, int(target_h * aspect))
                    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    import numpy as _np
                    oi = OffsetImage(_np.asarray(img), zoom=1.0, zorder=10)
                    ab = AnnotationBbox(oi, (0.08, 0.86), frameon=False,
                                        xycoords='axes fraction', box_alignment=(0.5, 0.5),
                                        zorder=10)
                    ax.add_artist(ab)
            except Exception:
                pass

    if title:
        fig.suptitle(title, fontsize=14, color='w')
    plt.tight_layout()
    return fig



def save_team_top_feature_table(
    table_df: pl.DataFrame,
    out_csv: Path,
    out_png: Path,
    title: str | None = 'Most-Used Corner Features by Team',
    subtitle: str | None = 'NMF topic with highest average weight across corners',
    logo_dir: Path | None = None
) -> None:
    """
    Save the team→top-feature mapping to CSV and a styled PNG table.

    If `logo_dir` is provided, attempts to place a team crest next to the team name
    by looking for `<logo_dir>/<sanitized-team-name>.png`.

    :param table_df: DataFrame with team features.
    :param out_csv: Output CSV Path.
    :param out_png: Output PNG Path.
    :param title: Table title.
    :param subtitle: Table subtitle.
    :param logo_dir: Directory containing team logos.
    """
    table_df.write_csv(out_csv)

    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image

    n = table_df.height
    fig_h = max(6.0, 0.42 * n + 2.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor('#313332')
    ax.set_facecolor('#313332')
    ax.axis('off')
    # Freeze a 0..1 coordinate space so all artists (text, lines, logos)
    # share the exact same reference frame
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Headers
    if title:
        fig.text(0.12, 0.975, title, fontweight='bold', fontsize=16, color='w')
    if subtitle:
        fig.text(0.12, 0.945, subtitle, fontweight='bold', fontsize=11, color='w')

    # Column positions (relative axes coords)
    # Expand columns dynamically if xT metrics present
    has_xt = 'xt_total' in table_df.columns or 'xt_avg' in table_df.columns
    if has_xt:
        # More spacing between Avg Weight and Corners
        x_rank, x_team, x_feat, x_xt, x_weight, x_corners = 0.04, 0.10, 0.60, 0.76, 0.87, 0.975
    else:
        # More spacing between Avg Weight and Corners
        x_rank, x_team, x_feat, x_weight, x_corners = 0.05, 0.10, 0.67, 0.82, 0.97
    # Bring table closer to title (but keep a small gap)
    y_top = 0.93
    row_gap = 0.9 / max(1, n + 2)
    # Center point for the 'Team' header over its column content
    x_team_hdr = x_team + 0.060

    # Header row lines (use data coords with fixed 0..1 limits)
    ax.plot([0.02, 0.98], [y_top + 0.02, y_top + 0.02], color='w', lw=1.0, zorder=1)
    ax.plot([0.02, 0.98], [y_top - (row_gap * (n + 0.5)), y_top - (row_gap * (n + 0.5))], color='w', lw=1.0, zorder=1)
    ax.text(x_rank, y_top, '#', ha='left', va='center', color='w', fontweight='bold', transform=ax.transAxes)
    ax.text(x_team_hdr, y_top, 'Team', ha='center', va='center', color='w', fontweight='bold', transform=ax.transAxes)
    ax.text(x_feat, y_top, 'Top Feature', ha='center', va='center', color='w', fontweight='bold', transform=ax.transAxes)
    if has_xt:
        ax.text(x_xt, y_top, 'xT (avg)', ha='center', va='center', color='w', fontweight='bold', transform=ax.transAxes)
    ax.text(x_weight, y_top, 'Avg Weight', ha='center', va='center', color='w', fontweight='bold', transform=ax.transAxes)
    ax.text(x_corners, y_top, 'Corners', ha='center', va='center', color='w', fontweight='bold', transform=ax.transAxes)

    def sanitize(name: str) -> str:
        return ''.join(ch.lower() for ch in str(name) if ch.isalnum())

    # Rows
    missing: list[str] = []
    logo_data = []  # Store logo data to add after text

    for i, r in enumerate(table_df.iter_rows(named=True), start=1):
        y = y_top - i * row_gap
        ax.text(x_rank, y, str(i), ha='left', va='center', color='w', fontsize=10, transform=ax.transAxes)

        # Team name + logo (if present)
        team_name = str(r.get('team'))
        has_logo = False

        if logo_dir is not None:
            logo_path = logo_dir / f"{sanitize(team_name)}.png"
            if logo_path.exists():
                has_logo = True
                logo_data.append((logo_path, x_team + 0.020, y, team_name))
                # Text with logo: shift further right
                ax.text(x_team + 0.060, y, team_name, ha='left', va='center', color='w', fontsize=9, transform=ax.transAxes)
            else:
                missing.append(team_name)

        if not has_logo:
            # No logo: show text without offset
            ax.text(x_team + 0.010, y, team_name, ha='left', va='center', color='w', fontsize=9, transform=ax.transAxes)

        ax.text(x_feat, y, str(int(r.get('top_feature_id', 0))), ha='center', va='center', color='w', fontsize=10, transform=ax.transAxes)
        if has_xt:
            xt_val = r.get('xt_avg', r.get('xt_total', 0.0))
            ax.text(x_xt, y, f"{float(xt_val):.3f}", ha='center', va='center', color='w', fontsize=10, transform=ax.transAxes)
        ax.text(x_weight, y, f"{float(r.get('top_weight', 0.0)):.3f}", ha='center', va='center', color='w', fontsize=10, transform=ax.transAxes)
        ax.text(x_corners, y, str(int(r.get('n_corners', 0))), ha='center', va='center', color='w', fontsize=10, transform=ax.transAxes)
        # Row separator
        ax.plot([0.02, 0.98], [y - row_gap * 0.45, y - row_gap * 0.45], color='gray', lw=0.5, zorder=1)

    # Add logos after all text to ensure proper layering
    for logo_path, x_pos, y_pos, team_name in logo_data:
        try:
            # Always convert to RGBA so matplotlib does not apply a colormap
            # (palette/greyscale images would otherwise be color-mapped e.g., purple)
            img = Image.open(logo_path).convert("RGBA")

            # Resize to a fixed height to ensure consistency across rows
            aspect = img.width / max(1, img.height)
            target_height = int(row_gap * fig_h * 72 * 0.60)  # 60% of row height in px
            target_width = max(1, int(target_height * aspect))
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

            # OffsetImage works best with numpy arrays for exact colors
            import numpy as _np
            img_arr = _np.asarray(img)
            oi = OffsetImage(img_arr, zoom=1.0, zorder=10)
            ab = AnnotationBbox(
                oi,
                (x_pos, y_pos),
                frameon=False,
                box_alignment=(0.0, 0.5),
                xycoords='axes fraction',
                zorder=10,
            )
            ax.add_artist(ab)
        except Exception as e:
            print(f"Warning: Failed to add logo for {team_name}: {e}")

    # Leave room for the figure title/subtitle to avoid overlap with table header
    plt.subplots_adjust(top=0.96, bottom=0.06, left=0.04, right=0.98)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # If any logos were missing, write a list next to the PNG for convenience
    if logo_dir is not None and missing:
        missing_file = out_png.with_name('missing_logos.txt')
        with open(missing_file, 'w', encoding='utf-8') as f:
            f.write('Expected logos not found (place PNGs in: ' + str(logo_dir) + ')\n')
            f.write('Use lowercase alphanumerics only, e.g., teamname.png\n\n')
            for name in sorted(set(missing)):
                f.write(f"- {name}  →  {sanitize(name)}.png\n")


"""
Note: Crest files must be manually named to sanitized team names (lowercase
alphanumerics only), e.g., "Manchester City" → "manchestercity.png". Missing
logos are listed in missing_logos.txt when rendering the table.
"""


def save_team_cti_table(
    table_df: pl.DataFrame,
    out_csv: Path,
    out_png: Path,
    title: str | None = 'Team CTI Summary (inference)',
    subtitle: str | None = None,
    logo_dir: Path | None = None
) -> None:

    """
    Save the team CTI summary to CSV and a styled PNG table with logos.

    Expected columns: 'team', 'cti_avg', 'p_shot', 'counter_risk', 'delta_xt'.
    (n_corners is optional but not displayed in the table).

    :param table_df: DataFrame with team CTI statistics.
    :param out_csv: Output CSV Path.
    :param out_png: Output PNG Path.
    :param title: Table title.
    :param subtitle: Table subtitle.
    :param logo_dir: Directory containing team logos.
    """
    table_df.write_csv(out_csv)

    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image

    n = table_df.height
    fig_h = max(6.0, 0.42 * n + 2.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor('#313332')
    ax.set_facecolor('#313332')
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Headers
    if title:
        fig.text(0.12, 0.975, title, fontweight='bold', fontsize=16, color='w')
    if subtitle:
        fig.text(0.12, 0.945, subtitle, fontweight='bold', fontsize=11, color='w')

    # Column positions (removed N column)
    x_rank, x_team, x_cti, x_pshot, x_crisk, x_dxt = 0.05, 0.10, 0.60, 0.75, 0.87, 0.97
    y_top = 0.93
    row_gap = 0.9 / max(1, n + 2)

    # Header labels (removed N column)
    ax.text(x_rank, y_top, '#', color='w', weight='bold', transform=ax.transAxes)
    ax.text(x_team, y_top, 'Team', color='w', weight='bold', transform=ax.transAxes)
    ax.text(x_cti, y_top, 'CTI (avg)', color='w', weight='bold', ha='center', transform=ax.transAxes)
    ax.text(x_pshot, y_top, 'P(shot)', color='w', weight='bold', ha='center', transform=ax.transAxes)
    ax.text(x_crisk, y_top, 'Counter risk', color='w', weight='bold', ha='center', transform=ax.transAxes)
    ax.text(x_dxt, y_top, 'ΔxT', color='w', weight='bold', ha='center', transform=ax.transAxes)

    def sanitize(name: str) -> str:
        return ''.join(ch.lower() for ch in str(name) if ch.isalnum())

    # Rows
    missing: list[str] = []

    for i, r in enumerate(table_df.iter_rows(named=True), start=1):
        y = y_top - i * row_gap
        ax.text(x_rank, y, str(i), ha='left', va='center', color='w', fontsize=10, transform=ax.transAxes)

        team_name = str(r.get('team'))
        has_logo = False
        if logo_dir is not None:
            logo_path = logo_dir / f"{sanitize(team_name)}.png"
            if logo_path.exists():
                has_logo = True
                try:
                    img = Image.open(logo_path).convert("RGBA")
                    aspect = img.width / max(1, img.height)
                    target_height = int(row_gap * fig_h * 72 * 0.60)
                    target_width = max(1, int(target_height * aspect))
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    import numpy as _np
                    oi = OffsetImage(_np.asarray(img), zoom=1.0, zorder=10)
                    ab = AnnotationBbox(
                        oi,
                        (x_team + 0.020, y),
                        frameon=False,
                        box_alignment=(0.0, 0.5),
                        xycoords='axes fraction',
                        zorder=10,
                    )
                    ax.add_artist(ab)
                except Exception as e:
                    print(f"Warning: Failed to add logo for {team_name}: {e}")
                # Add a bit more spacing between logo and team name
                ax.text(x_team + 0.100, y, team_name, ha='left', va='center', color='w', fontsize=9, transform=ax.transAxes)
            else:
                missing.append(team_name)
        if not has_logo:
            ax.text(x_team + 0.010, y, team_name, ha='left', va='center', color='w', fontsize=9, transform=ax.transAxes)

        ax.text(x_cti, y, f"{float(r.get('cti_avg', 0.0)):.3f}", ha='center', va='center', color='w', fontsize=10, transform=ax.transAxes)
        ax.text(x_pshot, y, f"{float(r.get('p_shot', 0.0)):.3f}", ha='center', va='center', color='w', fontsize=10, transform=ax.transAxes)
        ax.text(x_crisk, y, f"{float(r.get('counter_risk', 0.0)):.3f}", ha='center', va='center', color='w', fontsize=10, transform=ax.transAxes)
        ax.text(x_dxt, y, f"{float(r.get('delta_xt', 0.0)):.3f}", ha='center', va='center', color='w', fontsize=10, transform=ax.transAxes)
        ax.plot([0.02, 0.98], [y - row_gap * 0.45, y - row_gap * 0.45], color='gray', lw=0.5, zorder=1)

    plt.subplots_adjust(top=0.96, bottom=0.06, left=0.04, right=0.98)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)

    if logo_dir is not None and missing:
        missing_file = out_png.with_name('missing_logos.txt')
        with open(missing_file, 'w', encoding='utf-8') as f:
            f.write('Expected logos not found (place PNGs in: ' + str(logo_dir) + ')\n')
            f.write('Use lowercase alphanumerics only, e.g., teamname.png\n\n')
            for name in sorted(set(missing)):
                f.write(f"- {name}  →  {sanitize(name)}.png\n")


def save_nmf_model(nmf_model: NMFRoutineModel, output_path: Path) -> None:
    """
    Save fitted NMF model to disk using pickle.

    :param nmf_model: NMFRoutineModel object.
    :param output_path: Output file path.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(nmf_model, f)
    print(f"OK Saved NMF model to {output_path}")


def load_nmf_model(input_path: Path) -> NMFRoutineModel:
    """
    Load fitted NMF model from disk.

    :param input_path: Input file path.
    :return: Loaded NMFRoutineModel object.
    """
    with open(input_path, 'rb') as f:
        nmf_model = pickle.load(f)
    print(f"OK Loaded NMF model from {input_path}")
    return nmf_model


def generate_team_routine_report(
    team_id: int,
    corners_df: pl.DataFrame,
    nmf_model: NMFRoutineModel,
    top_n_routines: int = 5
) -> Dict:

    """
    Generate routine usage report for a specific team.

    :param team_id: Team identifier.
    :param corners_df: Corners DataFrame with 'team_id' column.
    :param nmf_model: Fitted NMFRoutineModel.
    :param top_n_routines: Number of top routines to report (default 5).
    :return: Dictionary containing team routine statistics, usage percentages, and diversity.
    """
    # Filter team corners
    team_corners = corners_df.filter(pl.col("team_id") == team_id)
    team_indices = team_corners.select("corner_id").to_series().to_list()

    if len(team_indices) == 0:
        return {'team_id': team_id, 'n_corners': 0}

    # Get routine weights for team corners
    team_W = nmf_model.W[team_indices]  # (n_team_corners, n_topics)

    # Average routine usage
    avg_routine_weights = team_W.mean(axis=0)

    # Top routines
    top_routine_ids = np.argsort(avg_routine_weights)[-top_n_routines:][::-1]

    report = {
        'team_id': team_id,
        'n_corners': len(team_indices),
        'top_routines': [
            {
                'routine_id': int(idx),
                'avg_weight': float(avg_routine_weights[idx]),
                'usage_pct': float(avg_routine_weights[idx] / avg_routine_weights.sum())
            }
            for idx in top_routine_ids
        ],
        'routine_diversity': float(np.std(avg_routine_weights))  # High std = varied routines
    }

    return report


if __name__ == "__main__":
    print("CTI NMF Routines Module loaded")
    print("Functions: fit_nmf_routines, assign_routines, identify_routine_features")
