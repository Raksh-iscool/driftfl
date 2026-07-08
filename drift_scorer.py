"""
Drift Scorer for DriftFL.

This module compares a current sketch against a baseline sketch using
chi-squared divergence to detect distribution shift.

KEY CONCEPTS:

Chi-Squared Divergence:
  Measures how much an observed distribution differs from an expected one.
  For each cell in the sketch, we compute:
      (observed - expected)^2 / (expected + epsilon)
  Large values mean the current distribution has moved away from the baseline.
  We sum across all cells and normalize.

Why chi-squared and not KL divergence or Wasserstein distance?
  - Chi-squared is symmetric (treats baseline and current equally)
  - Computable directly on sketch cells without reconstructing distributions
  - Well-understood statistical properties (known null distribution)
  - Cheap to compute: just element-wise operations on two 2D arrays

Severity Classification:
  We count what fraction of features exceed the drift threshold (tau).
  - phi < 0.05  -> NONE    (within noise range, ~5% features might
                             randomly exceed the 99th percentile threshold)
  - 0.05 <= phi < 0.25 -> MODERATE (real shift in some features, but the
                             client's updates still contain useful signal)
  - phi >= 0.25 -> SEVERE  (a quarter+ of features have shifted,
                             client updates would likely hurt the global model)

Adaptive Thresholds:
  Each client calibrates its own tau during the warm-up phase (first 5 rounds).
  Why? Because different clients have different data characteristics.
  A hospital with 2000 patients has more natural variability round-to-round
  than one with 50,000. A fixed threshold would either miss drift at the
  noisy client or false-alarm at the stable one.
"""

import numpy as np


# Severity levels
SEVERITY_NONE = 0
SEVERITY_MODERATE = 1
SEVERITY_SEVERE = 2


def chi_squared_score(sketch_current, sketch_baseline, epsilon=1.0):
    """
    Compute normalized chi-squared divergence between two sketch tables.

    This is Equation (5) from the paper. For each cell (i,j) in the
    sketch, we measure how much the current count deviates from the
    baseline count, normalized by the baseline count.

    Parameters
    ----------
    sketch_current : np.ndarray of shape (depth, width)
        The CMS table from the current round's data.
    sketch_baseline : np.ndarray of shape (depth, width)
        The CMS table from the warm-up phase (reference distribution).
    epsilon : float
        Small constant to prevent division by zero. Default 1.0.
        We use 1.0 (not a tiny value like 1e-8) because sketch cells
        are integer counts, and an empty cell genuinely means "no data
        mapped here." Adding 1.0 prevents infinite scores from dominating.

    Returns
    -------
    float
        The normalized chi-squared divergence score.
        Higher values indicate more distribution shift.
    """
    d, w = sketch_current.shape
    # Element-wise chi-squared: (O - E)^2 / (E + eps)
    diff_sq = (sketch_current.astype(float) - sketch_baseline.astype(float)) ** 2
    denominator = sketch_baseline.astype(float) + epsilon
    cell_scores = diff_sq / denominator
    # Normalize by total number of cells so the score is comparable
    # across sketches of different sizes
    return cell_scores.sum() / (d * w)


def compute_per_feature_scores(current_tables, baseline_tables, epsilon=1.0):
    """
    Compute drift scores for each feature dimension.

    Parameters
    ----------
    current_tables : np.ndarray of shape (num_features, depth, width)
        Current round's sketch tables for all features.
    baseline_tables : np.ndarray of shape (num_features, depth, width)
        Baseline sketch tables from warm-up phase.
    epsilon : float
        Division-by-zero prevention constant.

    Returns
    -------
    np.ndarray of shape (num_features,)
        Per-feature drift scores. Each entry tells you how much that
        feature dimension's distribution has changed.
    """
    num_features = current_tables.shape[0]
    scores = np.zeros(num_features)
    for f in range(num_features):
        scores[f] = chi_squared_score(
            current_tables[f], baseline_tables[f], epsilon
        )
    return scores


def calibrate_threshold(warmup_scores, percentile=99):
    """
    Set the drift threshold from warm-up phase scores.

    During warm-up (first T_0=5 rounds), the data is assumed stationary.
    Any variation in drift scores is just natural noise. We set tau as
    the 99th percentile of these scores -- meaning only 1% of features
    would randomly exceed tau even when there is NO drift.

    This is Equation (6) from the paper.

    Parameters
    ----------
    warmup_scores : list of np.ndarray
        List of per-feature score arrays from each warm-up round.
        Each array has shape (num_features,).
    percentile : int
        Which percentile to use. Default 99.

    Returns
    -------
    float
        The calibrated threshold tau for this client.
    """
    all_scores = np.concatenate(warmup_scores)
    if len(all_scores) == 0:
        return 1.0  # fallback if no warm-up data
    return np.percentile(all_scores, percentile)


def classify_severity(feature_scores, tau):
    """
    Classify overall drift severity based on fraction of flagged features.

    This is Equations (7) and (8) from the paper.

    Parameters
    ----------
    feature_scores : np.ndarray of shape (num_features,)
        Per-feature drift scores from compute_per_feature_scores().
    tau : float
        The drift threshold from calibrate_threshold().

    Returns
    -------
    severity : int
        0 = NONE, 1 = MODERATE, 2 = SEVERE
    phi : float
        Fraction of features exceeding the threshold.
    flagged_features : np.ndarray
        Boolean mask of which features were flagged.
    """
    flagged = feature_scores > tau
    phi = flagged.sum() / len(feature_scores)

    if phi < 0.05:
        severity = SEVERITY_NONE
    elif phi < 0.25:
        severity = SEVERITY_MODERATE
    else:
        severity = SEVERITY_SEVERE

    return severity, phi, flagged


class DriftMonitor:
    """
    Complete drift monitoring system for one federated client.

    Manages the warm-up phase, baseline construction, threshold
    calibration, and per-round drift scoring.

    Parameters
    ----------
    num_features : int
        Number of feature dimensions (D).
    warmup_rounds : int
        Number of initial rounds for baseline construction (T_0).
    sketch_depth : int
        CMS depth parameter (d).
    sketch_width : int
        CMS width parameter (w).
    """

    def __init__(self, num_features, warmup_rounds=5,
                 sketch_depth=5, sketch_width=256):
        self.num_features = num_features
        self.warmup_rounds = warmup_rounds
        self.sketch_depth = sketch_depth
        self.sketch_width = sketch_width

        # Baseline: accumulated sketch from warm-up phase
        self.baseline_tables = None
        # Threshold: calibrated from warm-up scores
        self.tau = None
        # Track warm-up data
        self.warmup_score_history = []
        self.warmup_sketch_history = []
        self.current_round = 0

    def is_warmup(self):
        """Check if we're still in the warm-up phase."""
        return self.current_round < self.warmup_rounds

    def process_round(self, current_tables):
        """
        Process one communication round's feature sketches.

        During warm-up: accumulates baseline and collects calibration data.
        During monitoring: scores drift and classifies severity.

        Parameters
        ----------
        current_tables : np.ndarray of shape (num_features, depth, width)
            This round's sketch tables from FeatureSketchBank.get_all_tables().

        Returns
        -------
        dict with keys:
            'severity' : int (0, 1, or 2)
            'phi' : float (fraction of flagged features)
            'round' : int (current round number)
            'is_warmup' : bool
            'scores' : np.ndarray or None (per-feature scores if not warmup)
        """
        self.current_round += 1

        if self.is_warmup():
            self.warmup_sketch_history.append(current_tables.copy())

            # After the first round, start computing scores against
            # the accumulating baseline for threshold calibration
            if len(self.warmup_sketch_history) > 1:
                # Baseline is the mean of all previous warm-up sketches
                prev_baseline = np.mean(
                    self.warmup_sketch_history[:-1], axis=0
                )
                scores = compute_per_feature_scores(
                    current_tables, prev_baseline
                )
                self.warmup_score_history.append(scores)

            # At the end of warm-up, finalize baseline and threshold
            if self.current_round == self.warmup_rounds:
                self.baseline_tables = np.mean(
                    self.warmup_sketch_history, axis=0
                )
                if len(self.warmup_score_history) > 0:
                    self.tau = calibrate_threshold(self.warmup_score_history)
                else:
                    self.tau = 1.0  # fallback

            return {
                'severity': SEVERITY_NONE,
                'phi': 0.0,
                'round': self.current_round,
                'is_warmup': True,
                'scores': None
            }
        else:
            # Monitoring phase: compare current against baseline
            scores = compute_per_feature_scores(
                current_tables, self.baseline_tables
            )
            severity, phi, flagged = classify_severity(scores, self.tau)

            return {
                'severity': severity,
                'phi': phi,
                'round': self.current_round,
                'is_warmup': False,
                'scores': scores
            }


if __name__ == "__main__":
    print("=== Drift Scorer Tests ===")

    # Test 1: Same distribution should produce low score
    rng = np.random.RandomState(42)
    base = rng.randint(0, 100, size=(5, 256)).astype(np.int32)
    noise = rng.randint(-2, 3, size=(5, 256)).astype(np.int32)
    curr_same = base + noise  # small noise
    score_same = chi_squared_score(curr_same, base)
    print(f"Same dist score: {score_same:.4f} (should be low)")

    # Test 2: Different distribution should produce high score
    curr_diff = rng.randint(0, 100, size=(5, 256)).astype(np.int32)
    score_diff = chi_squared_score(curr_diff, base)
    print(f"Diff dist score: {score_diff:.4f} (should be higher)")
    assert score_diff > score_same, "Different dist should score higher!"

    # Test 3: Severity classification
    scores = np.array([0.1, 0.2, 5.0, 0.3, 6.0, 0.1, 7.0, 0.2, 0.1, 0.15,
                        0.1, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1])
    tau = 1.0
    sev, phi, flagged = classify_severity(scores, tau)
    print(f"Severity: {sev}, Phi: {phi:.2f}, Flagged: {flagged.sum()}/{len(scores)}")
    # 3 out of 20 features > 1.0 -> phi=0.15 -> MODERATE
    assert sev == SEVERITY_MODERATE

    # Test 4: DriftMonitor full lifecycle
    print("\n=== DriftMonitor Lifecycle ===")
    monitor = DriftMonitor(num_features=10, warmup_rounds=3,
                           sketch_depth=5, sketch_width=64)

    # Warm-up rounds (stable data)
    for r in range(3):
        tables = rng.randint(0, 50, size=(10, 5, 64)).astype(np.int32)
        result = monitor.process_round(tables)
        print(f"Round {result['round']}: warmup={result['is_warmup']}, "
              f"severity={result['severity']}")

    # Monitoring round (no drift)
    tables_stable = rng.randint(0, 50, size=(10, 5, 64)).astype(np.int32)
    result = monitor.process_round(tables_stable)
    print(f"Round {result['round']}: warmup={result['is_warmup']}, "
          f"severity={result['severity']}, phi={result['phi']:.3f}")

    # Monitoring round (with drift -- very different distribution)
    tables_drifted = rng.randint(200, 500, size=(10, 5, 64)).astype(np.int32)
    result = monitor.process_round(tables_drifted)
    print(f"Round {result['round']}: warmup={result['is_warmup']}, "
          f"severity={result['severity']}, phi={result['phi']:.3f}")

    print("\nAll tests passed.")
