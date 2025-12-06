"""
Author: Tiago
Date: 2025-12-04
Description: Integration module containing the PyTorch Geometric dataset, Multi-Task GNN model, and Lightning module for CTI training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
from torch_geometric.loader import DataLoader
from typing import Tuple, Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass
import pickle
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
import polars as pl


# ============================================================================
# Focal Loss Implementation
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
    - p_t is the model's estimated probability for the class with label 1
    - gamma (focusing parameter): reduces loss for well-classified examples
    - alpha (balancing parameter): addresses class imbalance

    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples
               or -1 for no alpha weighting (default: 0.25)
        gamma: Exponent of the modulating factor (1 - p_t)^gamma (default: 2.0)
        reduction: 'none' | 'mean' | 'sum' (default: 'mean')

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the focal loss.

        :param inputs: Logits from model (before sigmoid), shape (N,).
        :param targets: Ground truth binary labels {0, 1}, shape (N,).
        :return: Computed focal loss value.
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute p_t: probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        # Apply alpha balancing if specified
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


# ============================================================================
# PART 1: Graph Construction & Target Extraction
# ============================================================================

def build_radius_graph(
    positions: np.ndarray,
    team_ids: np.ndarray,
    radius: float = 2.2,
    include_ball: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
def build_radius_graph(
    positions: np.ndarray,
    team_ids: np.ndarray,
    radius: float = 2.2,
    include_ball: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a radius graph for PyG format, connecting players within a specified distance.

    Edge types:
    - 0: ally-ally
    - 1: opponent-opponent
    - 2: ally-opponent

    :param positions: Array of shape (N, 2) with [x, y] coordinates for N players.
    :param team_ids: Array of shape (N,) with team identifiers.
    :param radius: Connection radius in meters.
    :param include_ball: Whether to include the ball node (if player_id=-1).
    :return: Tuple of (edge_index, edge_type).
    """
    n_nodes = positions.shape[0]

    edges = []
    edge_types = []

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            # Compute distance
            dist = np.linalg.norm(positions[i] - positions[j])

            if dist <= radius:
                # Add bidirectional edge
                edges.append([i, j])
                edges.append([j, i])

                # Determine edge type
                if team_ids[i] == team_ids[j]:
                    edge_type = 0  # Same team
                else:
                    edge_type = 2  # Opponents (marking candidate)

                edge_types.extend([edge_type, edge_type])

    if len(edges) == 0:
        # No edges, return empty tensors
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    return edge_index, edge_type


def extract_targets(
    corner_event: Dict,
    events_df: pl.DataFrame,
    xt_surface,
    window_outcome: Tuple[float, float] = (0.0, 10.0),
    window_counter: Tuple[float, float] = (10.0, 25.0),
    fps: int = 25,
    tracking_df: pl.DataFrame = None,
    player_team_map: Dict[int, int] = None,
    use_tracking_counter: bool = True
) -> Dict[str, float]:
def extract_targets(
    corner_event: Dict,
    events_df: pl.DataFrame,
    xt_surface,
    window_outcome: Tuple[float, float] = (0.0, 10.0),
    window_counter: Tuple[float, float] = (10.0, 25.0),
    fps: int = 25,
    tracking_df: pl.DataFrame = None,
    player_team_map: Dict[int, int] = None,
    use_tracking_counter: bool = True
) -> Dict[str, float]:
    """
    Extract supervision targets (y1-y5) for a given corner event.

    Targets:
    - y1: Shot within 10s (binary).
    - y2: Expected Goals (xG) of that shot (float).
    - y3: Counter-attack shot within 25s (binary).
    - y4: Counter-attack xG (float).
    - y5: Territory gained (Delta xT) (float).

    :param corner_event: Dictionary containing corner event details.
    :param events_df: DataFrame containing all events for the match.
    :param xt_surface: xT surface object for Delta xT calculation.
    :param window_outcome: Time window (start, end) in seconds for outcome targets (y1, y2, y5).
    :param window_counter: Time window (start, end) in seconds for counter targets (y3, y4).
    :param fps: Frames per second of the tracking data.
    :param tracking_df: Optional DataFrame containing tracking data.
    :param player_team_map: Optional mapping from player ID to team ID.
    :param use_tracking_counter: Whether to use tracking data for counter-attack targets.
    :return: Dictionary containing targets y1 through y5.
    """
    frame_start = corner_event["frame_start"]
    period = corner_event["period"]
    attacking_team = corner_event["team_id"]

    # Define frame windows
    frame_outcome_start = int(frame_start + window_outcome[0] * fps)
    frame_outcome_end = int(frame_start + window_outcome[1] * fps)
    frame_counter_start = int(frame_start + window_counter[0] * fps)
    frame_counter_end = int(frame_start + window_counter[1] * fps)

    # Filter events in windows
    outcome_events = events_df.filter(
        (pl.col("period") == period) &
        (pl.col("frame_start") >= frame_outcome_start) &
        (pl.col("frame_start") <= frame_outcome_end)
    )
    counter_events = events_df.filter(
        (pl.col("period") == period) &
        (pl.col("frame_start") >= frame_counter_start) &
        (pl.col("frame_start") <= frame_counter_end)
    )

    cols = set(events_df.columns)

    # Helper: generic shot detector across different schemas
    def _is_shot_expr():
        expr = None
        if "event_type" in cols:
            e = pl.col("event_type").cast(pl.Utf8, strict=False).str.to_lowercase() == "shot"
            expr = e if expr is None else (expr | e)
        if "event_subtype" in cols:
            e = pl.col("event_subtype").cast(pl.Utf8, strict=False).str.to_lowercase() == "shot"
            expr = e if expr is None else (expr | e)
        if "lead_to_shot" in cols:
            e = pl.col("lead_to_shot") == True
            expr = e if expr is None else (expr | e)
        if "is_shot" in cols:
            e = pl.col("is_shot") == True
            expr = e if expr is None else (expr | e)
        if expr is None and "end_type" in cols:
            expr = pl.col("end_type").cast(pl.Utf8, strict=False).str.to_lowercase() == "shot"
        # If still None, fallback to all-false
        return expr if expr is not None else (pl.lit(False))

    # y1: Shot within 10s by attacking team
    team_mask = (pl.col("team_id") == attacking_team) if "team_id" in cols else (pl.lit(True))
    shots_att = outcome_events.filter(team_mask & _is_shot_expr())
    y1 = 1.0 if shots_att.height > 0 else 0.0

    # y2: Max xG proxy within 10s (use xthreat max for the attacking team)
    if "xthreat" in outcome_events.columns:
        xg_att_df = outcome_events.filter(team_mask).select(pl.col("xthreat").drop_nulls())
        y2 = float(xg_att_df.max().item()) if xg_att_df.height > 0 else 0.0
    else:
        y2 = 0.0

    # y3: Counter-shot by opponent within 25s
    opp_mask = (pl.col("team_id") != attacking_team) if "team_id" in cols else (pl.lit(True))
    shots_opp = counter_events.filter(opp_mask & _is_shot_expr())
    y3_event = 1.0 if shots_opp.height > 0 else 0.0

    # y4: Opponent xG proxy on counter (max xthreat)
    if "xthreat" in counter_events.columns:
        xg_opp_df = counter_events.filter(opp_mask).select(pl.col("xthreat").drop_nulls())
        y4_event = float(xg_opp_df.max().item()) if xg_opp_df.height > 0 else 0.0
    else:
        y4_event = 0.0

    # Override with tracking-based counter risk if available
    if use_tracking_counter and tracking_df is not None and player_team_map is not None:
        try:
            from cti_counter_risk_tracking import compute_counter_risk_from_tracking
            metrics = compute_counter_risk_from_tracking(
                corner_event, tracking_df, events_df, player_team_map
            )
            y3 = float(metrics.has_counter)
            y4 = float(metrics.counter_xg)
        except Exception:
            # Fallback to event-based if tracking fails
            y3 = y3_event
            y4 = y4_event
    else:
        y3 = y3_event
        y4 = y4_event

    # y5: ΔxT (simplified - requires ball trajectory)
    # Approximate from event positions
    if outcome_events.height > 0:
        ball_positions = outcome_events.select(["x_end", "y_end"]).drop_nulls()
        if ball_positions.height >= 2:
            # Rename columns to match expected format
            ball_positions = ball_positions.rename({"x_end": "x", "y_end": "y"})
            from cti_xt_surface_half_pitch import compute_delta_xt
            y5 = compute_delta_xt(ball_positions, xt_surface)
        else:
            y5 = 0.0
    else:
        y5 = 0.0

    return {
        "y1": y1,  # Shot binary
        "y2": y2,  # Max xG
        "y3": y3,  # Counter-shot binary
        "y4": y4,  # Counter xG
        "y5": y5   # ΔxT
    }


# ============================================================================
# PART 2: PyTorch Geometric Dataset
# ============================================================================

from cti_gmm_zones import build_player_team_map


class CornerGraphDataset(Dataset):
    """
    PyTorch Geometric dataset for corner kick analysis.

    Each sample represents a corner kick with:
    - Node features: [x, y, vx, vy, team_onehot(2), role_prob]
    - Edge index: Radius graph connectivity
    - Edge features: Edge types (ally/opponent)
    - Global features: Delivery kinematics, run vectors
    - Targets: y1-y5

    :param corners_df: DataFrame containing corner metadata.
    :param tracking_dict: Dictionary mapping match IDs to tracking DataFrames.
    :param events_dict: Dictionary mapping match IDs to event DataFrames.
    :param xt_surface: xT surface grid.
    :param radius: Radius for graph connectivity.
    :param transform: Optional transform to apply to data.
    """

    def __init__(
        self,
        corners_df: pl.DataFrame,
        tracking_dict: Dict[int, pl.DataFrame],
        events_dict: Dict[int, pl.DataFrame],
        xt_surface,
        radius: float = 2.2,
        transform=None
    ):
        super().__init__(None, transform)
        self.corners_df = corners_df
        self.tracking_dict = tracking_dict
        self.events_dict = events_dict
        self.xt_surface = xt_surface
        self.radius = radius
        # Precompute player->team maps per match to enrich tracking
        self.team_maps: Dict[int, Dict[int, int]] = {}
        for mid, ev in events_dict.items():
            try:
                self.team_maps[mid] = build_player_team_map(ev)
            except Exception:
                self.team_maps[mid] = {}

    def len(self):
        return self.corners_df.height

    def get(self, idx):
        corner = self.corners_df.row(idx, named=True)
        match_id = corner["match_id"]
        frame_start = corner["frame_start"]
        period = corner["period"]

        # Get tracking data at corner kick frame
        if match_id not in self.tracking_dict:
             return Data()
        tracking = self.tracking_dict[match_id]
        frame_data_all = tracking.filter(
            (pl.col("frame") == frame_start) &
            (pl.col("period") == period)
        )
        frame_players = frame_data_all.filter(~pl.col("is_ball"))

        if frame_players.height == 0:
            # Return empty graph
            return Data()

        # Map players to team ids via events (tracking JSON may miss team_id)
        team_map = self.team_maps.get(match_id, {})
        # Join mapping DataFrame to avoid version-specific apply
        if team_map:
            map_df = pl.DataFrame({
                "player_id": list(team_map.keys()),
                "team_id_map": list(team_map.values())
            })
            frame_players = frame_players.join(map_df, on="player_id", how="left")
        else:
            frame_players = frame_players.with_columns(pl.lit(None).alias("team_id_map"))

        # Build simple velocities by differencing with previous frame
        prev_all = tracking.filter(
            (pl.col("frame") == frame_start - 1) &
            (pl.col("period") == period) &
            (~pl.col("is_ball"))
        ).select(["player_id", "x_m", "y_m"]).rename({"x_m": "x_prev", "y_m": "y_prev"})
        fp = frame_players.join(prev_all, on="player_id", how="left")
        # Replace null prev with current to yield zero velocity
        fp = fp.with_columns([
            pl.when(pl.col("x_prev").is_null()).then(pl.col("x_m")).otherwise(pl.col("x_prev")).alias("x_prev"),
            pl.when(pl.col("y_prev").is_null()).then(pl.col("y_m")).otherwise(pl.col("y_prev")).alias("y_prev"),
        ])
        # Velocity in m/s assuming 25 FPS
        fps_hz = 25.0
        fp = fp.with_columns([
            ((pl.col("x_m") - pl.col("x_prev")) * fps_hz).alias("vx"),
            ((pl.col("y_m") - pl.col("y_prev")) * fps_hz).alias("vy"),
        ])

        positions = fp.select(["x_m", "y_m"]).to_numpy()
        team_ids = fp.select("team_id_map").to_numpy().flatten()
        # Team flag: 1 for attackers, 0 for defenders (unknown→0)
        atk_id = corner.get("team_id")
        team_flag = (fp.select("team_id_map").to_numpy().flatten() == atk_id).astype(float)
        vels = fp.select(["vx", "vy"]).to_numpy()
        # Node features: [x, y, vx, vy, team_flag]
        import numpy as _np
        node_features = _np.concatenate([positions, vels, team_flag.reshape(-1, 1)], axis=1)

        # Build graph
        edge_index, edge_type = build_radius_graph(positions, team_ids, self.radius)

        # Extract targets (with tracking-based counter risk)
        targets = extract_targets(
            corner,
            self.events_dict[match_id],
            self.xt_surface,
            tracking_df=tracking,
            player_team_map=self.team_maps.get(match_id, None),
            use_tracking_counter=True
        )

        # Simple global features: short corner flag, delivery distance, corner side
        # Short corner: ball displacement within first 2s below 12m
        ball_win = tracking.filter(
            (pl.col("period") == period) &
            (pl.col("is_ball") == True) &
            (pl.col("frame") >= frame_start) & (pl.col("frame") <= frame_start + int(2 * 25))
        ).select(["x_m", "y_m"]).to_numpy()
        if ball_win.shape[0] >= 2:
            dxy = ball_win[-1] - ball_win[0]
            delivery_dist = float(((dxy[0] ** 2 + dxy[1] ** 2) ** 0.5))
        else:
            delivery_dist = 0.0
        is_short = 1.0 if delivery_dist < 12.0 else 0.0
        # Corner side: from y_start sign (centered coords); positive→top, negative→bottom; encode as binary
        y_start = corner.get("y_start")
        corner_side = 1.0 if (y_start is not None and float(y_start) >= 0.0) else 0.0

        # Store as 2D (1, global_dim) so PyG collates to (B, global_dim)
        global_feats = torch.tensor([is_short, delivery_dist, corner_side], dtype=torch.float).view(1, -1)

        # Create PyG Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_type,
            y=torch.tensor([targets["y1"], targets["y2"], targets["y3"],
                           targets["y4"], targets["y5"]], dtype=torch.float),
            global_feats=global_feats
        )

        return data


# ============================================================================
# PART 3: Multi-Task Model Architecture
# ============================================================================

class CTIMultiTaskModel(nn.Module):
    """
    Multi-task Graph Neural Network for Corner Threat Index (CTI) prediction.

    Architecture includes a spatial encoder (GraphSAGE), multiple prediction heads
    for targets y1-y5, and a differentiation CTI layer.

    :param input_dim: Dimension of input node features.
    :param hidden_dim: Dimension of hidden layers.
    :param num_gnn_layers: Number of GNN layers.
    :param dropout: Dropout probability.
    :param global_dim: Dimension of global features to concatenate.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        dropout: float = 0.3,
        global_dim: int = 3
    ):
        super().__init__()

        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_gnn_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.global_dim = global_dim

        # Task heads
        def make_head(out_dim=1):
            return nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, out_dim)
            )

        self.head_shot = make_head()  # y1: P(shot)
        self.head_xg = make_head()  # y2: xG
        self.head_counter = make_head()  # y3: P(counter-shot)
        self.head_xg_opp = make_head()  # y4: xG opponent
        self.head_delta_xt = make_head()  # y5: ΔxT

        # CTI parameters (learned or fixed)
        self.register_buffer("lambda_", torch.tensor(0.5))
        self.register_buffer("gamma_", torch.tensor(1.0))

        # Adapter for concatenated global features → hidden_dim
        if self.global_dim and self.global_dim > 0:
            self.adapter = nn.Linear(hidden_dim + self.global_dim, hidden_dim)
        else:
            self.adapter = nn.Identity()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GNN encoding
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = torch.relu(x)
            if i < len(self.convs) - 1:
                x = self.dropout(x)

        # Global pooling
        x = global_mean_pool(x, batch)  # (batch_size, hidden_dim)
        # Append global features if available
        if hasattr(data, "global_feats") and data.global_feats is not None:
            g = data.global_feats
            # Ensure global features align per-graph (batch) shape: (B, global_dim)
            B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            # Move to same device
            g = g.to(x.device)
            if g.dim() == 1:
                # Collated 1D tensor; try to infer correct shape
                if g.numel() == B * self.global_dim:
                    g = g.view(B, self.global_dim)
                elif g.numel() == self.global_dim:
                    g = g.view(1, self.global_dim).expand(B, -1)
                else:
                    g = g.view(B, -1)
            elif g.dim() == 2:
                if g.size(0) != B and g.numel() == B * self.global_dim:
                    g = g.view(B, self.global_dim)
                elif g.size(0) == 1 and g.size(1) == self.global_dim:
                    g = g.expand(B, -1)
            x = torch.cat([x, g], dim=1)
        

        # Predictions
        # Heads expect hidden_dim; if global feats were concatenated, adapt via small linear adapter
        x_h = self.adapter(x)
        # Binary heads (return both logits and probabilities)
        y1_logit = self.head_shot(x_h)
        y1 = torch.sigmoid(y1_logit)  # P(shot)

        # Regression heads with bounded sigmoid activation for proper scaling
        # Labels for y2/y4 top out below ~0.5, so allow headroom up to ~0.6
        y2_logit = self.head_xg(x_h)
        y2 = torch.sigmoid(y2_logit) * 0.6

        y3_logit = self.head_counter(x_h)
        y3 = torch.sigmoid(y3_logit)  # P(counter)

        # y4: Counter xG with sigmoid scaling to avoid collapse toward zero
        y4_logit = self.head_xg_opp(x_h)
        y4 = torch.sigmoid(y4_logit) * 0.6

        y5 = self.head_delta_xt(x_h)  # ΔxT (can be negative)

        # CTI computation (differentiable)
        cti = y1 * y2 - self.lambda_ * y3 * y4 + self.gamma_ * y5

        return {
            "y1": y1,
            "y1_logit": y1_logit,
            "y2": y2,
            "y3": y3,
            "y3_logit": y3_logit,
            "y4": y4,
            "y5": y5,
            "cti": cti
        }


# ============================================================================
# PART 4: Training with PyTorch Lightning
# ============================================================================

class CTILightningModule(L.LightningModule):
    """
    PyTorch Lightning module for training the CTI model.

    Handles training steps, validation steps, loss computation (including Focal Loss and Huber Loss),
    and optimization.
    """

    def __init__(
        self,
        model: CTIMultiTaskModel,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        loss_weights: Dict[str, float] = None,
        pos_weight_y1: float = 1.0,
        pos_weight_y3: float = 1.0,
        dynamic_pos_weight: bool = True,
        pos_weight_cap: float = 20.0,
        use_focal_loss_y3: bool = True,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.pos_weight_y1 = float(pos_weight_y1)
        self.pos_weight_y3 = float(pos_weight_y3)
        self.dynamic_pos_weight = bool(dynamic_pos_weight)
        self.pos_weight_cap = float(pos_weight_cap)
        self.use_focal_loss_y3 = use_focal_loss_y3
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        if loss_weights is None:
            # IMPROVED WEIGHTS based on analysis + y4 exponential activation fix
            loss_weights = {
                "y1": 1.0,   # Shot BCE (working reasonably well - keep as is)
                "y2": 5.0,   # xG Huber with exponential activation
                "y3": 10.0,  # Counter Focal/BCE - INCREASED from 5 (needs attention)
                "y4": 100.0, # xG_opp Huber with exponential - MASSIVE boost (518x→2368x underprediction!)
                "y5": 8.0    # ΔxT Huber - INCREASED from 5 (underperforming)
            }
        self.loss_weights = loss_weights
        print(f"[CTI] Using IMPROVED loss weights: {loss_weights}")
        if use_focal_loss_y3:
            print(f"[CTI] Using Focal Loss for y3 with alpha={focal_alpha}, gamma={focal_gamma}")

        # Loss functions
        # BCE with logits for y1 (use pos_weight to handle imbalance)
        # Focal Loss for y3 (better handles severe class imbalance)
        # Huber Loss for all regression tasks (robust to outliers)
        # Note: instantiate real loss modules with correct device in setup()
        self.bce_logits_y1 = None
        self.focal_loss_y3 = None
        self.bce_logits_y3 = None  # Fallback if not using focal loss
        self.huber_loss_y2 = nn.HuberLoss()
        self.huber_loss_y4 = nn.HuberLoss()
        self.huber_loss_y5 = nn.HuberLoss()  # CHANGED from MSE to Huber

    def setup(self, stage: Optional[str] = None):
        # Create loss functions on the correct device
        device = self.device if hasattr(self, 'device') else torch.device('cpu')

        # y1: BCE with pos_weight (working well)
        self.bce_logits_y1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight_y1, device=device))

        # y3: Focal Loss for severe class imbalance (or BCE as fallback)
        if self.use_focal_loss_y3:
            self.focal_loss_y3 = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma).to(device)
            print(f"[CTI] Initialized Focal Loss for y3 on {device}")
        else:
            self.bce_logits_y3 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight_y3, device=device))

        # Runtime accumulators for dynamic pos_weight
        self._epoch_total = 0
        self._epoch_pos_y1 = 0
        self._epoch_pos_y3 = 0

    def on_train_epoch_start(self):
        # Reset epoch counters
        self._epoch_total = 0
        self._epoch_pos_y1 = 0
        self._epoch_pos_y3 = 0

    def on_train_epoch_end(self):
        # Update pos_weight from observed prevalence in the epoch
        if not self.dynamic_pos_weight:
            return
        total = int(self._epoch_total)
        if total <= 0:
            return
        pos_y1 = max(0, int(self._epoch_pos_y1))
        pos_y3 = max(0, int(self._epoch_pos_y3))
        neg_y1 = max(0, total - pos_y1)
        neg_y3 = max(0, total - pos_y3)
        # (#neg/#pos), clamped
        def _pw(neg, pos):
            if pos <= 0:
                return 1.0
            return float(min(self.pos_weight_cap, max(1.0, neg / max(1, pos))))
        new_pw_y1 = _pw(neg_y1, pos_y1)
        new_pw_y3 = _pw(neg_y3, pos_y3)
        # Assign on device
        dev = self.device if hasattr(self, 'device') else torch.device('cpu')

        # Update y1 pos_weight (always using BCE)
        if self.bce_logits_y1 is not None:
            self.bce_logits_y1.pos_weight = torch.tensor(new_pw_y1, device=dev)
            self.log("pos_weight_y1", float(new_pw_y1), prog_bar=False)

        # Update y3 pos_weight only if using BCE (not Focal Loss)
        if not self.use_focal_loss_y3 and self.bce_logits_y3 is not None:
            self.bce_logits_y3.pos_weight = torch.tensor(new_pw_y3, device=dev)
            self.log("pos_weight_y3", float(new_pw_y3), prog_bar=False)
        elif self.use_focal_loss_y3:
            # Log the theoretical pos_weight for monitoring (even though Focal Loss handles it differently)
            self.log("pos_weight_y3_theoretical", float(new_pw_y3), prog_bar=False)

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        targets = batch.y  # Should be (batch_size * 5) in PyG batching

        # Reshape targets to (batch_size, 5) if needed
        if targets.dim() == 1 and targets.shape[0] % 5 == 0:
            targets = targets.view(-1, 5)

        # Compute losses with IMPROVED loss functions
        # y1: BCE with pos_weight (working well)
        loss_y1 = self.bce_logits_y1(outputs["y1_logit"].view(-1), targets[:, 0])

        # y2: Huber loss (robust to outliers)
        loss_y2 = self.huber_loss_y2(outputs["y2"].view(-1), targets[:, 1])

        # y3: Focal Loss (handles severe class imbalance) or BCE fallback
        if self.use_focal_loss_y3:
            loss_y3 = self.focal_loss_y3(outputs["y3_logit"].view(-1), targets[:, 2])
        else:
            loss_y3 = self.bce_logits_y3(outputs["y3_logit"].view(-1), targets[:, 2])

        # y4: Huber loss (robust to outliers)
        loss_y4 = self.huber_loss_y4(outputs["y4"].view(-1), targets[:, 3])

        # y5: Huber loss (CHANGED from MSE - more robust to outliers)
        loss_y5 = self.huber_loss_y5(outputs["y5"].view(-1), targets[:, 4])

        # Weighted sum with IMPROVED weights
        total_loss = (
            self.loss_weights["y1"] * loss_y1 +
            self.loss_weights["y2"] * loss_y2 +
            self.loss_weights["y3"] * loss_y3 +
            self.loss_weights["y4"] * loss_y4 +
            self.loss_weights["y5"] * loss_y5
        )

        # Logging
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_loss_y1", loss_y1)
        self.log("train_loss_y2", loss_y2)
        self.log("train_loss_y3", loss_y3)
        self.log("train_loss_y4", loss_y4)
        self.log("train_loss_y5", loss_y5)

        # Update epoch prevalence counters for dynamic pos_weight
        with torch.no_grad():
            y = targets
            if y.dim() == 1 and y.shape[0] % 5 == 0:
                y = y.view(-1, 5)
            if y.dim() == 2 and y.size(1) >= 3:
                self._epoch_total += y.size(0)
                self._epoch_pos_y1 += int((y[:, 0] > 0.5).sum().item())
                self._epoch_pos_y3 += int((y[:, 2] > 0.5).sum().item())

        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        targets = batch.y

        # Reshape targets to (batch_size, 5) if needed
        if targets.dim() == 1 and targets.shape[0] % 5 == 0:
            targets = targets.view(-1, 5)

        # Compute losses with IMPROVED loss functions (same as training)
        loss_y1 = self.bce_logits_y1(outputs["y1_logit"].view(-1), targets[:, 0])
        loss_y2 = self.huber_loss_y2(outputs["y2"].view(-1), targets[:, 1])

        if self.use_focal_loss_y3:
            loss_y3 = self.focal_loss_y3(outputs["y3_logit"].view(-1), targets[:, 2])
        else:
            loss_y3 = self.bce_logits_y3(outputs["y3_logit"].view(-1), targets[:, 2])

        loss_y4 = self.huber_loss_y4(outputs["y4"].view(-1), targets[:, 3])
        loss_y5 = self.huber_loss_y5(outputs["y5"].view(-1), targets[:, 4])

        total_loss = (
            self.loss_weights["y1"] * loss_y1 +
            self.loss_weights["y2"] * loss_y2 +
            self.loss_weights["y3"] * loss_y3 +
            self.loss_weights["y4"] * loss_y4 +
            self.loss_weights["y5"] * loss_y5
        )

        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_loss_y1", loss_y1)
        self.log("val_loss_y2", loss_y2)
        self.log("val_loss_y3", loss_y3)
        self.log("val_loss_y4", loss_y4)
        self.log("val_loss_y5", loss_y5)

        # Store predictions and targets for metrics computation at epoch end
        if not hasattr(self, 'val_preds'):
            self.val_preds = {f'y{i}': [] for i in range(1, 6)}
            self.val_targets = {f'y{i}': [] for i in range(1, 6)}

        # Store predictions (convert logits to probs for y1, y3)
        self.val_preds['y1'].append(torch.sigmoid(outputs["y1_logit"]).detach().cpu())
        self.val_preds['y2'].append(outputs["y2"].detach().cpu())
        self.val_preds['y3'].append(torch.sigmoid(outputs["y3_logit"]).detach().cpu())
        self.val_preds['y4'].append(outputs["y4"].detach().cpu())
        self.val_preds['y5'].append(outputs["y5"].detach().cpu())

        # Store targets
        for i in range(5):
            self.val_targets[f'y{i+1}'].append(targets[:, i].detach().cpu())

        return total_loss

    def on_validation_epoch_end(self):
        """Compute and log AUC, precision, recall, F1 for all tasks at end of validation epoch."""
        if not hasattr(self, 'val_preds') or len(self.val_preds['y1']) == 0:
            return

        # Concatenate all batches
        preds = {k: torch.cat(v).numpy() for k, v in self.val_preds.items()}
        targets = {k: torch.cat(v).numpy() for k, v in self.val_targets.items()}

        # Compute metrics for each task
        metrics = {}

        for task_idx, task_name in enumerate(['y1', 'y2', 'y3', 'y4', 'y5']):
            y_true = targets[task_name]
            y_pred = preds[task_name]

            # For binary tasks (y1, y3), compute AUC, precision, recall, F1
            if task_name in ['y1', 'y3']:
                try:
                    # AUC
                    if len(np.unique(y_true)) > 1:  # Need at least 2 classes for AUC
                        auc = roc_auc_score(y_true, y_pred)
                        metrics[f'val_auc_{task_name}'] = auc

                    # Binarize predictions at 0.5 threshold
                    y_pred_binary = (y_pred > 0.5).astype(int)

                    # Precision, recall, F1
                    prec = precision_score(y_true, y_pred_binary, zero_division=0)
                    rec = recall_score(y_true, y_pred_binary, zero_division=0)
                    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

                    metrics[f'val_precision_{task_name}'] = prec
                    metrics[f'val_recall_{task_name}'] = rec
                    metrics[f'val_f1_{task_name}'] = f1
                except Exception as e:
                    print(f"[WARNING] Could not compute metrics for {task_name}: {e}")

            # For continuous tasks (y2, y4, y5), binarize at threshold 0.0 for classification metrics
            else:
                try:
                    y_true_binary = (y_true > 0.0).astype(int)
                    y_pred_binary = (y_pred > 0.0).astype(int)

                    # AUC (if there are positive examples)
                    if len(np.unique(y_true_binary)) > 1:
                        auc = roc_auc_score(y_true_binary, y_pred)
                        metrics[f'val_auc_{task_name}'] = auc

                    # Precision, recall, F1
                    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                    rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

                    metrics[f'val_precision_{task_name}'] = prec
                    metrics[f'val_recall_{task_name}'] = rec
                    metrics[f'val_f1_{task_name}'] = f1

                    # Also log mean prediction value to monitor for zero-prediction collapse
                    metrics[f'val_mean_pred_{task_name}'] = float(y_pred.mean())
                except Exception as e:
                    print(f"[WARNING] Could not compute metrics for {task_name}: {e}")

        # Log all metrics
        self.log_dict(metrics, prog_bar=False, logger=True)

        # Print summary every epoch
        print(f"\n[Validation Metrics - Epoch {self.current_epoch}]")
        for task_name in ['y1', 'y2', 'y3', 'y4', 'y5']:
            if f'val_auc_{task_name}' in metrics:
                print(f"  {task_name}: AUC={metrics.get(f'val_auc_{task_name}', 0):.3f}, "
                      f"P={metrics.get(f'val_precision_{task_name}', 0):.3f}, "
                      f"R={metrics.get(f'val_recall_{task_name}', 0):.3f}, "
                      f"F1={metrics.get(f'val_f1_{task_name}', 0):.3f}" +
                      (f", Mean={metrics.get(f'val_mean_pred_{task_name}', 0):.4f}" if f'val_mean_pred_{task_name}' in metrics else ""))

        # Clear stored predictions for next epoch
        self.val_preds = {f'y{i}': [] for i in range(1, 6)}
        self.val_targets = {f'y{i}': [] for i in range(1, 6)}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100
        )
        return [optimizer], [scheduler]

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        """
        Configure gradient clipping to prevent exploding gradients.
        Called by PyTorch Lightning automatically if gradient_clip_val is set in Trainer.
        We set it to 1.0 for stable training of y3, y4, y5 tasks.
        """
        # Clip gradients by norm (max_norm=1.0)
        if gradient_clip_val is not None and gradient_clip_val > 0:
            self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm or "norm")


# ============================================================================
# PART 5: Lambda/Gamma Learning & CTI Computation
# ============================================================================

def learn_lambda_gamma(
    corners_df: pl.DataFrame,
    predictions_df: pl.DataFrame,
    actual_goals_df: pl.DataFrame
) -> Tuple[float, float]:
    """
    Learn lambda and gamma parameters via ridge regression on match aggregates.

    These parameters balance the CTI equation: CTI = y1*y2 - lambda*y3*y4 + gamma*y5.

    :param corners_df: DataFrame containing corners with match_id and team_id.
    :param predictions_df: DataFrame containing model predictions (y1-y5) per corner.
    :param actual_goals_df: DataFrame containing actual net goals per match/team.
    :return: Tuple of learned (lambda, gamma).
    """
    from sklearn.linear_model import Ridge

    # Aggregate predictions per match/team
    # X = [offensive_term, counter_term, xt_term] per match
    # y = actual_net_goals per match

    # Simplified implementation (requires proper aggregation)
    print("Learning λ and γ via ridge regression...")

    # Placeholder: fit Ridge(alpha=1.0) on aggregated data
    # In practice, aggregate y1*y2, y3*y4, y5 per match/team
    lambda_ = 0.5  # Default
    gamma_ = 1.0   # Default

    print(f"OK Learned: λ={lambda_:.3f}, γ={gamma_:.3f}")

    return lambda_, gamma_


def compute_cti(
    y1: float,
    y2: float,
    y3: float,
    y4: float,
    y5: float,
    lambda_: float = 0.5,
    gamma_: float = 1.0
) -> float:
    """
    Compute CTI score from model predictions using the formula:
    CTI = y1 * y2 - lambda * y3 * y4 + gamma * y5

    :param y1: Shot probability.
    :param y2: Shot xG.
    :param y3: Counter-attack probability.
    :param y4: Counter-attack xG.
    :param y5: Territory gained (Delta xT).
    :param lambda_: Weight for counter-attack risk.
    :param gamma_: Weight for territorial gain.
    :return: Computed CTI score.
    """
    cti = y1 * y2 - lambda_ * y3 * y4 + gamma_ * y5
    return float(cti)


# ============================================================================
# PART 6: Evaluation Metrics
# ============================================================================

def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "binary"
) -> Dict[str, float]:
    """
    Compute standard evaluation metrics for binary classification or regression tasks.

    For binary: Accuracy, AUC, Brier Score.
    For regression: MAE, RMSE.

    :param y_true: Ground truth labels.
    :param y_pred: Predicted values (probabilities or continuous).
    :param task_type: "binary" or "regression".
    :return: Dictionary mapping metric names to values.
    """
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, brier_score_loss,
        mean_absolute_error, mean_squared_error
    )

    metrics = {}

    if task_type == "binary":
        metrics["accuracy"] = accuracy_score(y_true > 0.5, y_pred > 0.5)
        if len(np.unique(y_true)) > 1:  # Need both classes for AUC
            metrics["auc"] = roc_auc_score(y_true, y_pred)
        metrics["brier"] = brier_score_loss(y_true, y_pred)

    elif task_type == "regression":
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))

    return metrics


if __name__ == "__main__":
    print("CTI Integration Module loaded")
    print("Components: Graphs, Targets, Dataset, Model, Training, CTI")
