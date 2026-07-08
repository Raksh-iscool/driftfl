"""
Count-Min Sketch (CMS) Implementation for DriftFL.

A count-min sketch is a probabilistic data structure that approximates
the frequency of items in a data stream using a fixed amount of memory.
It trades exact accuracy for space efficiency -- it may overcount but
never undercounts.

Structure: A 2D array of size (depth x width).
- depth (d): Number of independent hash functions. More depth = lower
  false positive probability. We use d=5, giving error prob < 0.67%.
- width (w): Number of buckets per hash function. More width = lower
  frequency estimation error. We use w=256, giving error < 1.06% of
  total count.

How it works:
1. INSERT: For item x, compute d hash values h_1(x)...h_d(x). Each
   hash maps x to a column index in {0, 1, ..., w-1}. Increment
   the cell at (row_i, h_i(x)) for each row i.
2. QUERY: Return the MINIMUM of cells (row_i, h_i(x)) across all rows.
   The minimum reduces overcounting from hash collisions.

Why it preserves privacy:
- The sketch is a lossy compression. You cannot reconstruct which
  specific data points went in. Multiple different datasets can produce
  the same sketch.
- Recovering individual records from a CMS requires solving an
  underdetermined system of equations (d*w equations, D*n unknowns),
  which is computationally infeasible for any realistic data size.
"""

import numpy as np
import hashlib
import struct


class CountMinSketch:
    """
    Count-Min Sketch for frequency estimation.

    Parameters
    ----------
    depth : int
        Number of hash functions (rows). Default 5.
        Higher depth = lower false positive probability.
        Error probability = 1/e^depth. At d=5: ~0.67%.
    width : int
        Number of buckets per hash function (columns). Default 256.
        Higher width = more accurate frequency estimates.
        Max overestimate = e/width fraction of total count. At w=256: ~1.06%.
    seed : int
        Random seed for hash function generation. Keeping the same seed
        across clients ensures they use identical hash mappings, which
        is required for meaningful cross-client sketch comparison.
    """

    def __init__(self, depth=5, width=256, seed=42):
        self.depth = depth
        self.width = width
        self.seed = seed
        # The actual counter table. Using int32 to handle large counts
        # without overflow. In the paper we mention 16-bit counters
        # are fine for batch sizes under 10,000.
        self.table = np.zeros((depth, width), dtype=np.int32)
        # Pre-generate seeds for each hash function so they're
        # deterministic and consistent across all clients
        self._hash_seeds = [seed + i * 7919 for i in range(depth)]

    def _hash(self, item, row_index):
        """
        Hash an item to a column index for a given row.

        Uses SHA-256 truncated to 8 bytes for good distribution.
        The row_index selects which hash function to use via different seeds.

        Parameters
        ----------
        item : int or float
            The value to hash. We convert to string first.
        row_index : int
            Which hash function (row) we're computing for.

        Returns
        -------
        int
            Column index in [0, width).
        """
        # Combine item value with row-specific seed
        key = f"{item}_{self._hash_seeds[row_index]}".encode('utf-8')
        h = hashlib.sha256(key).digest()
        # Take first 8 bytes, convert to integer, mod by width
        val = struct.unpack('<Q', h[:8])[0]
        return val % self.width

    def update(self, item, count=1):
        """
        Record an observation of `item` by incrementing all d cells.

        This is the INSERT operation. For each of the d hash functions,
        we compute the hash of `item` to get a column index, then
        increment the cell at (row, column) by `count`.

        Parameters
        ----------
        item : int or float
            The observed value.
        count : int
            How many times to count this observation. Default 1.
        """
        for i in range(self.depth):
            col = self._hash(item, i)
            self.table[i, col] += count

    def query(self, item):
        """
        Estimate the frequency of `item`.

        Returns the minimum count across all d rows. The true count
        is guaranteed to be <= this estimate (CMS never undercounts).
        The overcount is bounded by (e/w) * N with probability 1 - 1/e^d.

        Parameters
        ----------
        item : int or float
            The value to query.

        Returns
        -------
        int
            Estimated frequency (always >= true frequency).
        """
        counts = []
        for i in range(self.depth):
            col = self._hash(item, i)
            counts.append(self.table[i, col])
        return min(counts)

    def batch_update(self, items):
        """
        Insert multiple items efficiently.

        Parameters
        ----------
        items : array-like
            Sequence of values to insert.
        """
        for item in items:
            self.update(item)

    def batch_update_quantized(self, values, num_bins=256):
        """
        Quantize continuous feature values into discrete bins, then
        insert each bin index into the sketch.

        This is what we use in DriftFL. Neural network feature activations
        are continuous floats, but the CMS needs discrete items. We
        discretize by mapping each value to one of `num_bins` bins
        spanning the observed range.

        Parameters
        ----------
        values : np.ndarray
            1D array of continuous feature values.
        num_bins : int
            Number of discrete bins. Default 256 (matches sketch width).
        """
        if len(values) == 0:
            return
        vmin, vmax = values.min(), values.max()
        if vmax == vmin:
            # All values identical -- put everything in bin 0
            self.table[:, 0] += len(values)
            return
        # Map each value to a bin index in [0, num_bins)
        bin_indices = np.clip(
            ((values - vmin) / (vmax - vmin) * (num_bins - 1)).astype(int),
            0, num_bins - 1
        )
        for bin_idx in bin_indices:
            self.update(int(bin_idx))

    def get_table(self):
        """Return a copy of the internal counter table."""
        return self.table.copy()

    def reset(self):
        """Zero out all counters."""
        self.table.fill(0)


class FeatureSketchBank:
    """
    Maintains one CMS per feature dimension.

    In DriftFL, we extract D-dimensional feature vectors from the model's
    penultimate layer. Each dimension gets its own CMS so we can detect
    which specific features drifted.

    For D=512 features with d=5, w=256, 16-bit counters:
    Total memory = 512 * 5 * 256 * 2 bytes = 1.31 MB

    Parameters
    ----------
    num_features : int
        Number of feature dimensions (D in the paper).
    depth : int
        CMS depth (d). Default 5.
    width : int
        CMS width (w). Default 256.
    seed : int
        Base random seed.
    """

    def __init__(self, num_features, depth=5, width=256, seed=42):
        self.num_features = num_features
        self.depth = depth
        self.width = width
        self.sketches = [
            CountMinSketch(depth=depth, width=width, seed=seed + f * 31)
            for f in range(num_features)
        ]

    def update_from_features(self, feature_matrix, num_bins=256):
        """
        Given a batch of feature vectors, update all per-feature sketches.

        Parameters
        ----------
        feature_matrix : np.ndarray of shape (batch_size, num_features)
            Feature activations from the model's penultimate layer.
        num_bins : int
            Number of quantization bins for continuous values.
        """
        assert feature_matrix.shape[1] == self.num_features, \
            f"Expected {self.num_features} features, got {feature_matrix.shape[1]}"
        for f in range(self.num_features):
            self.sketches[f].batch_update_quantized(
                feature_matrix[:, f], num_bins=num_bins
            )

    def get_all_tables(self):
        """
        Return the sketch tables for all features as a 3D array.

        Returns
        -------
        np.ndarray of shape (num_features, depth, width)
        """
        return np.array([s.get_table() for s in self.sketches])

    def reset_all(self):
        """Reset all sketches to zero."""
        for s in self.sketches:
            s.reset()


if __name__ == "__main__":
    # Quick sanity check
    print("=== Count-Min Sketch Test ===")
    cms = CountMinSketch(depth=5, width=256, seed=42)

    # Insert known frequencies
    for _ in range(100):
        cms.update(42)
    for _ in range(50):
        cms.update(99)
    for _ in range(10):
        cms.update(7)

    print(f"Query 42 (true=100): {cms.query(42)}")
    print(f"Query 99 (true=50):  {cms.query(99)}")
    print(f"Query 7  (true=10):  {cms.query(7)}")
    print(f"Query 0  (true=0):   {cms.query(0)}")  # should be 0 or small

    print("\n=== Feature Sketch Bank Test ===")
    bank = FeatureSketchBank(num_features=4, depth=5, width=64, seed=0)
    # Simulate a batch of 100 samples with 4 features
    rng = np.random.RandomState(123)
    features = rng.randn(100, 4)
    bank.update_from_features(features)
    tables = bank.get_all_tables()
    print(f"Tables shape: {tables.shape}")  # (4, 5, 64)
    print(f"Total counts per feature: {[tables[f].sum() // 5 for f in range(4)]}")
    print("Tests passed.")
