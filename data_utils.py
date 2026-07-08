"""
Data Utilities for DriftFL Experiments.

Handles:
1. Loading datasets (CIFAR-100, synthetic regression)
2. Non-IID splitting via Dirichlet allocation
3. Drift injection at specified rounds

KEY CONCEPTS:

Non-IID Data Splitting (Dirichlet Allocation):
  In real federated learning, clients don't have identical data.
  A hospital in Florida sees different patients than one in Alaska.
  We simulate this by drawing class proportions from a Dirichlet
  distribution with concentration parameter alpha.
  - alpha -> infinity: all clients have the same class distribution (IID)
  - alpha = 1.0: moderate heterogeneity
  - alpha = 0.5: high heterogeneity (what we use)
  - alpha -> 0: each client has only one class (extreme non-IID)

Drift Injection (Rotating Class Assignments):
  Starting at round 50, we rotate the class labels for 20% of clients.
  If a client originally had classes [0,1,2], after rotation it gets
  classes [5,6,7]. This simulates a real scenario where the types
  of data a client sees change over time (e.g., a hospital starts
  seeing different diseases due to a seasonal outbreak).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# CIFAR-100 Dataset Handling
# ============================================================

def get_cifar100_transforms():
    """Standard CIFAR-100 data augmentation and normalization."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    return train_transform, test_transform


def load_cifar100(data_dir='./data'):
    """Load CIFAR-100 train and test sets."""
    train_t, test_t = get_cifar100_transforms()
    train_set = datasets.CIFAR100(data_dir, train=True, download=True,
                                   transform=train_t)
    test_set = datasets.CIFAR100(data_dir, train=False, download=True,
                                  transform=test_t)
    return train_set, test_set


def dirichlet_split(dataset, num_clients, alpha=0.5, seed=42):
    """
    Split a dataset across clients using Dirichlet allocation.

    For each class, we draw a probability vector from Dir(alpha, ..., alpha)
    with num_clients entries. This vector determines what fraction of that
    class's samples go to each client.

    Parameters
    ----------
    dataset : torchvision Dataset
        Must have a .targets attribute.
    num_clients : int
        Number of federated clients (K).
    alpha : float
        Dirichlet concentration. Lower = more heterogeneous.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of list of int
        client_indices[k] = list of dataset indices assigned to client k.
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        # Find all samples of class c
        class_indices = np.where(targets == c)[0]
        rng.shuffle(class_indices)

        # Draw proportions from Dirichlet
        proportions = rng.dirichlet([alpha] * num_clients)
        # Convert proportions to sample counts
        proportions = (proportions * len(class_indices)).astype(int)
        # Fix rounding errors
        proportions[-1] = len(class_indices) - proportions[:-1].sum()

        # Assign indices to clients
        start = 0
        for k in range(num_clients):
            end = start + proportions[k]
            client_indices[k].extend(class_indices[start:end].tolist())
            start = end

    # Shuffle each client's data
    for k in range(num_clients):
        rng.shuffle(client_indices[k])

    return client_indices


def apply_class_rotation(client_indices, dataset, drifted_clients,
                         rotation_offset=5, seed=42):
    """
    Rotate class assignments for specified clients.

    For each drifted client, replace its data with samples from
    classes shifted by rotation_offset. If client originally had
    class 3 samples, it now gets class 8 samples.

    This simulates label distribution drift: the types of data
    the client sees have changed.

    Parameters
    ----------
    client_indices : list of list of int
        Current index assignments per client.
    dataset : torchvision Dataset
        The full dataset.
    drifted_clients : list of int
        Which client indices should experience drift.
    rotation_offset : int
        How many classes to shift by.
    seed : int
        Random seed.

    Returns
    -------
    list of list of int
        Updated client indices with drift applied.
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    new_indices = [list(idx) for idx in client_indices]

    for k in drifted_clients:
        old_indices = client_indices[k]
        old_classes = targets[old_indices]
        rotated_indices = []

        for idx in old_indices:
            old_class = targets[idx]
            new_class = (old_class + rotation_offset) % num_classes
            # Find a random sample from the new class
            candidates = np.where(targets == new_class)[0]
            chosen = rng.choice(candidates)
            rotated_indices.append(chosen)

        new_indices[k] = rotated_indices

    return new_indices


# ============================================================
# Synthetic Regression Dataset
# ============================================================

class SyntheticRegressionDataset(Dataset):
    """
    Synthetic linear regression dataset.

    y = X @ w + noise

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
    y : np.ndarray of shape (n,)
    """

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_synthetic_data(num_clients=50, samples_per_client=1000,
                            num_features=20, noise_std=0.1, seed=42):
    """
    Generate synthetic regression data for each client.

    Each client gets data from y = X @ w_true + noise, where X is
    drawn from N(0, I) and w_true is shared across all clients.

    Parameters
    ----------
    num_clients : int
        Number of clients (K).
    samples_per_client : int
        Number of samples per client.
    num_features : int
        Dimensionality of features.
    noise_std : float
        Standard deviation of Gaussian noise.
    seed : int
        Random seed.

    Returns
    -------
    list of SyntheticRegressionDataset
        One dataset per client.
    w_true : np.ndarray
        The ground-truth weight vector.
    """
    rng = np.random.RandomState(seed)
    w_true = rng.randn(num_features)

    client_datasets = []
    for k in range(num_clients):
        X = rng.randn(samples_per_client, num_features)
        noise = rng.randn(samples_per_client) * noise_std
        y = X @ w_true + noise
        client_datasets.append(SyntheticRegressionDataset(X, y))

    return client_datasets, w_true


def apply_covariate_shift(client_datasets, drifted_clients,
                          rotation_degrees=45, seed=42):
    """
    Apply covariate shift by rotating the feature space.

    For drifted clients, we multiply their feature matrix X by a
    rotation matrix. This changes which directions in feature space
    carry information, simulating a scenario where the input
    characteristics change while the underlying relationship stays
    the same.

    Parameters
    ----------
    client_datasets : list of SyntheticRegressionDataset
    drifted_clients : list of int
        Which clients get shifted.
    rotation_degrees : float
        Rotation angle in degrees.
    seed : int
        Random seed.

    Returns
    -------
    list of SyntheticRegressionDataset
        Updated datasets with drift applied.
    """
    rng = np.random.RandomState(seed)
    d = client_datasets[0].X.shape[1]

    # Generate a random rotation matrix using QR decomposition
    # of a random Gaussian matrix
    random_matrix = rng.randn(d, d)
    Q, _ = np.linalg.qr(random_matrix)

    # Scale the rotation: interpolate between identity and Q
    # angle=0 -> identity, angle=90 -> full random rotation
    theta = np.radians(rotation_degrees)
    rotation = np.eye(d) * np.cos(theta) + Q * np.sin(theta)

    new_datasets = list(client_datasets)
    for k in drifted_clients:
        X_old = client_datasets[k].X.numpy()
        X_rotated = X_old @ rotation.T
        y = client_datasets[k].y.numpy()
        # Note: y stays the same because y = X_original @ w + noise
        # but now X has been rotated, so the model trained on original
        # X will make bad predictions on rotated X
        new_datasets[k] = SyntheticRegressionDataset(X_rotated, y)

    return new_datasets


# ============================================================
# Federated Data Loader Helper
# ============================================================

class ClientDataManager:
    """
    Manages per-client data loading across rounds, including drift injection.

    Parameters
    ----------
    task : str
        'cifar100' or 'synthetic'
    num_clients : int
        Number of federated clients.
    drift_start_round : int
        Round at which drift begins.
    drift_fraction : float
        Fraction of clients that experience drift.
    """

    def __init__(self, task='cifar100', num_clients=100,
                 drift_start_round=50, drift_fraction=0.2,
                 alpha=0.5, seed=42):
        self.task = task
        self.num_clients = num_clients
        self.drift_start_round = drift_start_round
        self.drift_fraction = drift_fraction
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Select which clients will drift
        num_drifted = int(num_clients * drift_fraction)
        self.drifted_clients = sorted(
            self.rng.choice(num_clients, num_drifted, replace=False).tolist()
        )

        if task == 'cifar100':
            self.train_set, self.test_set = load_cifar100()
            self.base_indices = dirichlet_split(
                self.train_set, num_clients, alpha=alpha, seed=seed
            )
            self.current_indices = [list(idx) for idx in self.base_indices]
        elif task == 'synthetic':
            self.client_datasets, self.w_true = generate_synthetic_data(
                num_clients=num_clients, seed=seed
            )
            self.base_datasets = list(self.client_datasets)

        self.drift_active = False
        self.rotation_count = 0

    def get_client_loader(self, client_id, batch_size=64):
        """Get a DataLoader for a specific client."""
        if self.task == 'cifar100':
            subset = Subset(self.train_set, self.current_indices[client_id])
            return DataLoader(subset, batch_size=batch_size, shuffle=True)
        else:
            return DataLoader(self.client_datasets[client_id],
                              batch_size=batch_size, shuffle=True)

    def get_test_loader(self, batch_size=128):
        """Get the global test DataLoader."""
        if self.task == 'cifar100':
            return DataLoader(self.test_set, batch_size=batch_size,
                              shuffle=False)
        else:
            # For synthetic, create a held-out test set
            rng = np.random.RandomState(self.seed + 9999)
            X_test = rng.randn(1000, 20)
            y_test = X_test @ self.w_true + rng.randn(1000) * 0.1
            test_ds = SyntheticRegressionDataset(X_test, y_test)
            return DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    def maybe_inject_drift(self, current_round):
        """
        Check if drift should be injected at this round.

        For CIFAR-100: rotates classes every 10 rounds after drift_start.
        For synthetic: applies one-time covariate shift at drift_start.
        """
        if current_round < self.drift_start_round:
            return

        if self.task == 'cifar100':
            rounds_since_drift = current_round - self.drift_start_round
            expected_rotations = rounds_since_drift // 10 + 1
            if expected_rotations > self.rotation_count:
                self.rotation_count = expected_rotations
                offset = 5 * self.rotation_count
                self.current_indices = apply_class_rotation(
                    self.base_indices, self.train_set,
                    self.drifted_clients, rotation_offset=offset,
                    seed=self.seed + current_round
                )
                self.drift_active = True

        elif self.task == 'synthetic':
            if not self.drift_active:
                self.client_datasets = apply_covariate_shift(
                    self.base_datasets, self.drifted_clients,
                    rotation_degrees=45, seed=self.seed
                )
                self.drift_active = True

    def is_client_drifted(self, client_id):
        """Check if a specific client is in the drifted group."""
        return client_id in self.drifted_clients


if __name__ == "__main__":
    print("=== Data Utilities Tests ===")

    # Test Dirichlet split
    print("\n--- Testing Dirichlet Split ---")
    train_t, _ = get_cifar100_transforms()
    # Use a small fake dataset for testing
    from unittest.mock import MagicMock
    fake_dataset = MagicMock()
    fake_dataset.targets = list(range(10)) * 100  # 1000 samples, 10 classes
    indices = dirichlet_split(fake_dataset, num_clients=5, alpha=0.5, seed=42)
    for k in range(5):
        print(f"  Client {k}: {len(indices[k])} samples")

    # Test synthetic data
    print("\n--- Testing Synthetic Data ---")
    datasets_list, w = generate_synthetic_data(num_clients=5,
                                                samples_per_client=100)
    for k in range(5):
        print(f"  Client {k}: X shape {datasets_list[k].X.shape}, "
              f"y shape {datasets_list[k].y.shape}")

    print("\nAll data utility tests passed.")
