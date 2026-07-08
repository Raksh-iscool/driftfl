"""
DriftFL Configuration.

All hyperparameters in one place for reproducibility.
"""

# ============================================================
# Federated Learning Configuration
# ============================================================
NUM_ROUNDS = 100           # Total communication rounds (T)
NUM_CLIENTS = 100          # Total clients (K) for CIFAR-100
NUM_CLIENTS_SYNTHETIC = 50 # Total clients for synthetic task
CLIENT_FRACTION = 0.1      # Fraction of clients per round (C)
LOCAL_EPOCHS = 5           # Local SGD epochs (E)
LOCAL_LR = 0.01            # Local learning rate
LOCAL_MOMENTUM = 0.9       # SGD momentum
BATCH_SIZE = 64            # Local batch size
NUM_RUNS = 5               # Independent runs for averaging
FEDPROX_MU = 0.01          # Proximal coefficient for FedProx

# ============================================================
# DriftFL Sketch Configuration
# ============================================================
SKETCH_DEPTH = 5           # CMS depth (d) -- 5 hash functions
SKETCH_WIDTH = 256         # CMS width (w) -- 256 buckets per hash
WARMUP_ROUNDS = 5          # Warm-up phase duration (T_0)
QUANTIZATION_BINS = 256    # Bins for continuous feature discretization

# ============================================================
# DriftFL Severity Thresholds
# ============================================================
# These define the boundaries between severity tiers
# based on the fraction of flagged features (phi)
PHI_MODERATE = 0.05        # phi >= 0.05 -> MODERATE
PHI_SEVERE = 0.25          # phi >= 0.25 -> SEVERE
THRESHOLD_PERCENTILE = 99  # Percentile for tau calibration

# ============================================================
# Aggregation Weight Multipliers
# ============================================================
WEIGHT_NONE = 1.0          # alpha for SEVERITY_NONE
WEIGHT_MODERATE = 0.5      # alpha for SEVERITY_MODERATE
WEIGHT_SEVERE = 0.0        # alpha for SEVERITY_SEVERE (excluded)

# ============================================================
# Drift Injection Configuration
# ============================================================
DRIFT_START_ROUND = 50     # Round at which drift begins
DRIFT_FRACTION = 0.2       # Fraction of clients that drift
ROTATION_OFFSET = 5        # Class rotation offset for CIFAR-100
ROTATION_DEGREES = 45      # Feature rotation for synthetic

# ============================================================
# Hardware / Reproducibility
# ============================================================
DEVICE = "cuda"            # or "cpu"
SEED = 42                  # Base random seed
