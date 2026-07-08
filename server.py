"""
DriftFL Federated Server.

Handles:
1. Selecting clients each round
2. Distributing the global model
3. Collecting client updates and severity reports
4. Adjusting aggregation weights based on severity
5. Weighted averaging of model updates

AGGREGATION WEIGHT ADJUSTMENT (the server's response to drift):

When DriftFL is active, the server modifies each client's aggregation
weight based on reported severity:

  severity = 0 (NONE):     alpha = 1.0  (full trust)
  severity = 1 (MODERATE): alpha = 0.5  (half weight)
  severity = 2 (SEVERE):   alpha = 0.0  (excluded)

The adjusted aggregation formula is:
  theta_new = SUM_k (alpha_k * n_k / SUM_j(alpha_j * n_j)) * theta_k

This is just weighted FedAvg where the weights are modified by alpha.

Why not just exclude all drifted clients?
  Because moderate drift means the client's updates contain a MIX of
  useful and corrupted information. Excluding them entirely shrinks the
  effective training pool, which hurts accuracy on the non-drifted parts.
  The 0.5 weight is a compromise: still benefit from their data, but
  reduce the damage from drift.
"""

import torch
import numpy as np
from collections import OrderedDict
import copy

import config


class FederatedServer:
    """
    Central server for DriftFL federated learning.

    Parameters
    ----------
    model : nn.Module
        The global model.
    num_clients : int
        Total number of clients.
    client_fraction : float
        Fraction of clients selected per round.
    method : str
        'fedavg', 'fedprox', 'driftfl', 'adwin', 'oracle'.
    seed : int
        Random seed for client selection.
    """

    def __init__(self, model, num_clients, client_fraction=0.1,
                 method='driftfl', seed=42):
        self.global_model = copy.deepcopy(model)
        self.num_clients = num_clients
        self.client_fraction = client_fraction
        self.method = method
        self.rng = np.random.RandomState(seed)

        # Logging
        self.round_log = []

    def select_clients(self, current_round):
        """
        Randomly select a subset of clients for this round.

        Returns
        -------
        list of int
            Selected client indices.
        """
        num_selected = max(1, int(self.num_clients * self.client_fraction))
        return sorted(self.rng.choice(
            self.num_clients, num_selected, replace=False
        ).tolist())

    def get_global_state_dict(self):
        """Return the current global model parameters."""
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_results, drifted_clients_ground_truth=None,
                  current_round=0):
        """
        Aggregate client updates with severity-based weight adjustment.

        Parameters
        ----------
        client_results : list of dict
            Each dict has 'state_dict', 'num_samples', 'severity'.
        drifted_clients_ground_truth : set or None
            For Oracle method: which clients are actually drifted.
        current_round : int
            Current round (for logging).

        Returns
        -------
        dict
            Aggregation statistics (severity counts, effective clients, etc.)
        """
        if len(client_results) == 0:
            return {'effective_clients': 0}

        # Determine aggregation weights
        weights = []
        for result in client_results:
            n_k = result['num_samples']
            severity = result['severity']
            client_id = result.get('client_id', -1)

            if self.method == 'oracle' and drifted_clients_ground_truth:
                # Oracle knows exactly which clients are drifted
                if client_id in drifted_clients_ground_truth:
                    alpha = 0.0
                else:
                    alpha = 1.0
            elif self.method == 'driftfl':
                # DriftFL: use severity-based weights
                if severity == 0:
                    alpha = config.WEIGHT_NONE
                elif severity == 1:
                    alpha = config.WEIGHT_MODERATE
                else:
                    alpha = config.WEIGHT_SEVERE
            elif self.method == 'adwin':
                # ADWIN: binary exclude if severity > 0
                alpha = 0.0 if severity > 0 else 1.0
            else:
                # FedAvg, FedProx: no drift handling
                alpha = 1.0

            weights.append(alpha * n_k)

        total_weight = sum(weights)
        if total_weight == 0:
            # All clients excluded -- keep current global model
            return {
                'effective_clients': 0,
                'severity_counts': {0: 0, 1: 0, 2: 0}
            }

        # Normalize weights
        weights = [w / total_weight for w in weights]

        # Weighted average of state dicts
        avg_state = OrderedDict()
        for key in client_results[0]['state_dict'].keys():
            avg_state[key] = torch.zeros_like(
                client_results[0]['state_dict'][key], dtype=torch.float32
            )
            for i, result in enumerate(client_results):
                avg_state[key] += weights[i] * result['state_dict'][key].float()

        self.global_model.load_state_dict(avg_state)

        # Log severity distribution
        severities = [r['severity'] for r in client_results]
        severity_counts = {
            0: severities.count(0),
            1: severities.count(1),
            2: severities.count(2)
        }
        effective = sum(1 for w in weights if w > 0)

        log_entry = {
            'round': current_round,
            'severity_counts': severity_counts,
            'effective_clients': effective,
            'total_clients': len(client_results),
            'avg_loss': np.mean([r['loss'] for r in client_results])
        }
        self.round_log.append(log_entry)

        return log_entry
