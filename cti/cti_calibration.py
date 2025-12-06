"""
Author: Tiago
Date: 2025-12-04
Description: CTI Calibration Module. Implements Platt scaling for probabilistic calibration of binary classification tasks.
"""

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import pickle
from typing import Dict


class PlattScaler:
    """
    Platt scaling calibrator for binary classification outputs.

    Fits a logistic regression: P_calibrated = sigmoid(A * logit + B)
    where logit = log(p / (1-p)) for uncalibrated probability p.
    """

    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.is_fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        """
        Fit Platt scaler on validation data.

        Args:
            probs: Uncalibrated probabilities, shape (N,)
            labels: True binary labels, shape (N,)
        """
        # Convert probabilities to logits for more stable fitting
        # Clip to avoid log(0) or log(1)
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs_clipped / (1 - probs_clipped))

        # Fit logistic regression on logits -> labels
        self.model.fit(logits.reshape(-1, 1), labels)
        self.is_fitted = True

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration to uncalibrated probabilities.

        Args:
            probs: Uncalibrated probabilities, shape (N,)

        Returns:
            Calibrated probabilities, shape (N,)
        """
        if not self.is_fitted:
            raise ValueError("PlattScaler must be fitted before transform")

        # Convert to logits
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs_clipped / (1 - probs_clipped))

        # Apply calibration
        calibrated = self.model.predict_proba(logits.reshape(-1, 1))[:, 1]
        return calibrated


def fit_platt_calibrators(model, val_loader, device='cpu') -> Dict[str, PlattScaler]:
    """
    Fit Platt scalers for all binary classification tasks (y1, y3).

    Args:
        model: Trained CTI model (MultiTaskGNN)
        val_loader: Validation data loader
        device: Device for model inference

    Returns:
        Dictionary of fitted PlattScaler objects: {'y1': scaler, 'y3': scaler}
    """
    model.eval()
    model.to(device)

    # Collect all predictions and targets
    all_preds = {'y1': [], 'y3': []}
    all_targets = {'y1': [], 'y3': []}

    print("[Calibration] Collecting predictions on validation set...")

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            targets = batch.y

            # Reshape targets if needed
            if targets.dim() == 1 and targets.shape[0] % 5 == 0:
                targets = targets.view(-1, 5)

            # Get probabilities (sigmoid of logits)
            y1_prob = torch.sigmoid(outputs["y1_logit"]).cpu().numpy()
            y3_prob = torch.sigmoid(outputs["y3_logit"]).cpu().numpy()

            # Get targets
            y1_target = targets[:, 0].cpu().numpy()
            y3_target = targets[:, 2].cpu().numpy()

            all_preds['y1'].append(y1_prob)
            all_preds['y3'].append(y3_prob)
            all_targets['y1'].append(y1_target)
            all_targets['y3'].append(y3_target)

    # Concatenate all batches
    all_preds = {k: np.concatenate(v) for k, v in all_preds.items()}
    all_targets = {k: np.concatenate(v) for k, v in all_targets.items()}

    print(f"  Collected {len(all_preds['y1'])} validation samples")

    # Fit Platt scalers
    calibrators = {}

    for task in ['y1', 'y3']:
        print(f"  Fitting Platt scaler for {task}...")

        probs = all_preds[task]
        labels = all_targets[task]

        # Check if there are both classes
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            print(f"    [WARNING] Only one class present in {task}, skipping calibration")
            calibrators[task] = None
            continue

        # Fit scaler
        scaler = PlattScaler()
        scaler.fit(probs, labels)
        calibrators[task] = scaler

        # Report calibration effect
        probs_before = probs
        probs_after = scaler.transform(probs)

        print(f"    Before calibration: mean P({task})={probs_before.mean():.3f}, "
              f"P(positive)={labels.mean():.3f}")
        print(f"    After calibration:  mean P({task})={probs_after.mean():.3f}")

    return calibrators


def save_calibrators(calibrators: Dict[str, PlattScaler], output_dir: Path):
    """
    Save fitted Platt scalers to disk.

    Args:
        calibrators: Dictionary of PlattScaler objects
        output_dir: Directory to save calibrators.pkl
    """
    output_path = Path(output_dir) / 'calibrators.pkl'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(calibrators, f)

    print(f"[Calibration] Saved calibrators to {output_path}")


def load_calibrators(output_dir: Path) -> Dict[str, PlattScaler]:
    """
    Load fitted Platt scalers from disk.

    Args:
        output_dir: Directory containing calibrators.pkl

    Returns:
        Dictionary of PlattScaler objects
    """
    calibrators_path = Path(output_dir) / 'calibrators.pkl'

    if not calibrators_path.exists():
        raise FileNotFoundError(f"Calibrators not found at {calibrators_path}. "
                                "Run fit_platt_calibrators() first.")

    with open(calibrators_path, 'rb') as f:
        calibrators = pickle.load(f)

    print(f"[Calibration] Loaded calibrators from {calibrators_path}")
    return calibrators


def apply_calibration(predictions: Dict[str, np.ndarray],
                      calibrators: Dict[str, PlattScaler]) -> Dict[str, np.ndarray]:
    """
    Apply calibration to predictions.

    Args:
        predictions: Dictionary with keys 'y1', 'y3' containing uncalibrated probabilities
        calibrators: Dictionary of fitted PlattScaler objects

    Returns:
        Dictionary with calibrated probabilities
    """
    calibrated = {}

    for task in ['y1', 'y3']:
        if task in predictions and task in calibrators and calibrators[task] is not None:
            calibrated[task] = calibrators[task].transform(predictions[task])
        else:
            # No calibration available, return original
            calibrated[task] = predictions[task] if task in predictions else None

    # Copy non-calibrated tasks
    for task in ['y2', 'y4', 'y5']:
        if task in predictions:
            calibrated[task] = predictions[task]

    return calibrated


# Example usage in training pipeline:
"""
# After training completes:
from cti_calibration import fit_platt_calibrators, save_calibrators, load_calibrators, apply_calibration

# Fit on validation set
calibrators = fit_platt_calibrators(model, val_loader, device='cuda')

# Save
save_calibrators(calibrators, output_dir='cti_outputs')

# Later, during inference:
calibrators = load_calibrators('cti_outputs')

# Apply to new predictions
predictions = {'y1': y1_probs, 'y3': y3_probs, ...}
calibrated_predictions = apply_calibration(predictions, calibrators)

# Use calibrated predictions for CTI computation
CTI = calibrated_predictions['y1'] * y2 - 0.5 * calibrated_predictions['y3'] * y4 + y5
"""
