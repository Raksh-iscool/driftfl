"""
DriftFL Federated Client.

This module implements the per-client training loop with integrated
drift detection. Each client:
1. Receives the global model from the server
2. Trains locally for E epochs
3. Extracts features from the penultimate layer
4. Builds a count-min sketch from those features
5. Compares current sketch against baseline (chi-squared divergence)
6. Reports severity to the server alongside model updates

The client never sends raw data or even raw sketches to the server.
Only the severity label (0, 1, or 2) is transmitted.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from sketch import FeatureSketchBank
from drift_scorer import DriftMonitor
import config


def train_local(model, dataloader, epochs, lr, momentum, device,
                fedprox_mu=0.0, global_params=None):
    """
    Train a model locally on one client's data.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    dataloader : DataLoader
        Client's local training data.
    epochs : int
        Number of local epochs (E).
    lr : float
        Learning rate.
    momentum : float
        SGD momentum.
    device : str
        'cuda' or 'cpu'.
    fedprox_mu : float
        If > 0, adds a proximal term to the loss: mu/2 * ||w - w_global||^2.
        This is FedProx's contribution -- it keeps local updates from
        diverging too far from the global model.
    global_params : dict or None
        Global model parameters (needed for FedProx).

    Returns
    -------
    float
        Average training loss over all epochs.
    """
    model.train()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Determine loss function based on output shape
    is_regression = (list(model.parameters())[-1].shape[0] == 1)
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for batch_data in dataloader:
            inputs, targets = batch_data
            inputs = inputs.to(device)
            targets = targets.to(device)
            if is_regression:
                targets = targets.float().unsqueeze(1) if targets.dim() == 1 else targets

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # FedProx proximal term
            if fedprox_mu > 0 and global_params is not None:
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    if name in global_params:
                        proximal_term += ((param - global_params[name].to(device)) ** 2).sum()
                loss += (fedprox_mu / 2.0) * proximal_term

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def extract_features_from_data(model, dataloader, device):
    """
    Run all client data through the model and collect penultimate features.

    This is the step where we get the 512-dim (or 64-dim) feature vectors
    that DriftFL will compress into sketches.

    Parameters
    ----------
    model : nn.Module
        Must have an extract_features() method.
    dataloader : DataLoader
        Client's data.
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    np.ndarray of shape (total_samples, feature_dim)
    """
    model.eval()
    model.to(device)
    all_features = []

    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data[0].to(device)
            features = model.extract_features(inputs)
            all_features.append(features.cpu().numpy())

    if len(all_features) == 0:
        return np.array([])
    return np.concatenate(all_features, axis=0)


class DriftFLClient:
    """
    A single federated client with DriftFL drift detection.

    Parameters
    ----------
    client_id : int
        Unique identifier for this client.
    model : nn.Module
        The model architecture (shared with all clients).
    dataloader : DataLoader
        This client's local training data.
    device : str
        'cuda' or 'cpu'.
    method : str
        'fedavg', 'fedprox', 'driftfl', or 'driftfl+adwin'.
    """

    def __init__(self, client_id, model, dataloader, device='cuda',
                 method='driftfl'):
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.method = method

        # Initialize drift monitor
        feature_dim = model.feature_dim
        self.drift_monitor = DriftMonitor(
            num_features=feature_dim,
            warmup_rounds=config.WARMUP_ROUNDS,
            sketch_depth=config.SKETCH_DEPTH,
            sketch_width=config.SKETCH_WIDTH
        )
        self.sketch_bank = FeatureSketchBank(
            num_features=feature_dim,
            depth=config.SKETCH_DEPTH,
            width=config.SKETCH_WIDTH
        )

        # For ADWIN baseline
        self.loss_history = []

    def update_dataloader(self, new_dataloader):
        """Update the client's data (used when drift is injected)."""
        self.dataloader = new_dataloader

    def fit(self, global_state_dict, current_round):
        """
        Perform one round of local training + drift detection.

        Steps:
        1. Load global model weights
        2. Train locally for E epochs
        3. Extract penultimate features
        4. Build sketch, compare to baseline
        5. Return updated weights and severity

        Parameters
        ----------
        global_state_dict : OrderedDict
            Global model parameters from server.
        current_round : int
            Current communication round number.

        Returns
        -------
        dict with:
            'state_dict': OrderedDict (updated local model weights)
            'num_samples': int (size of local dataset)
            'severity': int (0, 1, or 2)
            'loss': float (average training loss)
            'phi': float (fraction of flagged features)
        """
        # Step 1: Load global model
        self.model.load_state_dict(global_state_dict)
        global_params = {k: v.clone() for k, v in global_state_dict.items()}

        # Step 2: Train locally
        fedprox_mu = config.FEDPROX_MU if self.method == 'fedprox' else 0.0
        avg_loss = train_local(
            self.model, self.dataloader,
            epochs=config.LOCAL_EPOCHS,
            lr=config.LOCAL_LR,
            momentum=config.LOCAL_MOMENTUM,
            device=self.device,
            fedprox_mu=fedprox_mu,
            global_params=global_params if fedprox_mu > 0 else None
        )
        self.loss_history.append(avg_loss)

        # Step 3: Extract features
        features = extract_features_from_data(
            self.model, self.dataloader, self.device
        )
        num_samples = len(features) if len(features) > 0 else len(self.dataloader.dataset)

        severity = 0
        phi = 0.0

        if self.method in ('driftfl',) and len(features) > 0:
            # Step 4: Build sketch
            self.sketch_bank.reset_all()
            self.sketch_bank.update_from_features(features)
            current_tables = self.sketch_bank.get_all_tables()

            # Step 5: Compare to baseline and classify severity
            result = self.drift_monitor.process_round(current_tables)
            severity = result['severity']
            phi = result['phi']

        elif self.method == 'adwin':
            # ADWIN baseline: detect drift from loss signal
            severity = self._adwin_check()

        return {
            'state_dict': self.model.state_dict(),
            'num_samples': num_samples,
            'severity': severity,
            'loss': avg_loss,
            'phi': phi
        }

    def _adwin_check(self):
        """
        Simple ADWIN-like drift detection on loss.

        Compares recent loss window against historical loss.
        If the recent window's mean loss is significantly higher,
        flag as drifted.
        """
        if len(self.loss_history) < 10:
            return 0
        recent = self.loss_history[-5:]
        historical = self.loss_history[-10:-5]
        recent_mean = np.mean(recent)
        hist_mean = np.mean(historical)
        hist_std = np.std(historical) + 1e-8

        # Z-score test
        z = (recent_mean - hist_mean) / hist_std
        if z > 3.0:
            return 2  # severe
        elif z > 2.0:
            return 1  # moderate
        return 0


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model accuracy on the test set.

    Parameters
    ----------
    model : nn.Module
    test_loader : DataLoader
    device : str

    Returns
    -------
    float
        Accuracy (for classification) or negative MSE (for regression).
    """
    model.eval()
    model.to(device)

    is_regression = (list(model.parameters())[-1].shape[0] == 1)

    if is_regression:
        total_mse = 0.0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.float().to(device)
                outputs = model(inputs).squeeze()
                mse = ((outputs - targets) ** 2).sum().item()
                total_mse += mse
                total_samples += len(targets)
        # Return R^2-like metric (higher is better)
        avg_mse = total_mse / total_samples
        return max(0, 100 * (1 - avg_mse))  # scale to 0-100 range
    else:
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        return 100.0 * correct / total
