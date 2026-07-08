# DriftFL

**DriftFL** is a privacy-preserving federated learning framework that detects client-side feature distribution drift and dynamically adjusts aggregation weights based on drift severity. Unlike conventional federated learning methods that assume stationary client data distributions, DriftFL enables robust learning under non-IID and evolving data streams without compromising data privacy.

The framework performs drift detection entirely on-device using compact probabilistic data structures and communicates only a lightweight severity label to the central server, ensuring minimal communication overhead while maintaining strong privacy guarantees.

---

## Overview

In real-world federated learning deployments, client data distributions often evolve over time due to changing environments, user behavior, or sensor characteristics. Traditional aggregation algorithms such as FedAvg treat every client equally, allowing drifted clients to negatively influence the global model.

DriftFL addresses this challenge through:

- Local feature-level drift monitoring
- Privacy-preserving drift detection
- Severity-aware aggregation
- Adaptive client weighting
- Constant communication overhead

Rather than transmitting feature statistics or raw data, each client communicates only a **4-byte drift severity label**, allowing the server to determine how much trust should be assigned to each model update.

---

## Key Features

- Privacy-preserving client-side drift detection
- Count-Min Sketch based feature distribution summarization
- Normalized Chi-Squared divergence for drift scoring
- Adaptive per-client threshold calibration
- Three-level drift severity classification
- Severity-weighted federated aggregation
- Extremely low communication overhead (4 bytes per round)
- Compatible with existing weighted federated learning algorithms

---

## Methodology

DriftFL consists of four major components:

### 1. Feature Extraction

Each client extracts feature representations from the penultimate layer of the local neural network after local training.

### 2. Count-Min Sketch Representation

Instead of storing complete feature histograms, DriftFL compresses each feature distribution using Count-Min Sketches, providing an efficient and privacy-preserving summary.

### 3. Drift Detection

Current feature distributions are compared with warm-up baseline distributions using a normalized Chi-Squared divergence.

Each client independently determines whether drift has occurred based on adaptive thresholds learned during the warm-up phase.

### 4. Severity-Weighted Aggregation

Detected drift is classified into three severity levels:

| Severity | Aggregation Weight |
|----------|-------------------:|
| Stable | 1.0 |
| Moderate Drift | 0.5 |
| Severe Drift | 0.0 |

The federated server uses these weights during model aggregation, reducing the influence of drifted clients while preserving useful updates.

---

## Architecture

```
                Global Server
                      │
         Broadcast Global Model
                      │
     ┌────────────────┴────────────────┐
     │                                 │
 Client 1                         Client K
     │                                 │
 Local Training                 Local Training
     │                                 │
 Feature Extraction            Feature Extraction
     │                                 │
 Count-Min Sketch             Count-Min Sketch
     │                                 │
 Chi-Squared Drift Detection  Chi-Squared Drift Detection
     │                                 │
 Severity Classification      Severity Classification
     │                                 │
     └────────── Severity Label ───────┘
                      │
        Severity-Weighted Aggregation
                      │
             Updated Global Model
```

---

## Highlights

- Detects feature-level distribution shifts before model degradation becomes severe.
- Preserves client privacy by keeping feature statistics entirely local.
- Adds only a constant-size communication cost regardless of model size.
- Supports heterogeneous and non-IID federated learning environments.
- Improves robustness against concept drift through adaptive aggregation.

---

## Project Structure

```
driftfl/
│── sketch.py
│── drift_scorer.py
│── models.py
│── data_utils.py
│── config.py
│── client.py
│── server.py
│── run_experiment.py
│── plot_results.py
└── README.md
```

---

## Core Components

| File | Description |
|------|-------------|
| `sketch.py` | Count-Min Sketch implementation for feature summarization |
| `drift_scorer.py` | Chi-Squared drift scoring and severity estimation |
| `models.py` | Neural network architectures used in federated learning |
| `data_utils.py` | Dataset preparation, non-IID partitioning, and drift injection |
| `config.py` | Experiment configuration and hyperparameters |
| `client.py` | Federated client with local drift detection |
| `server.py` | Federated server with severity-weighted aggregation |
| `run_experiment.py` | Main experiment pipeline |
| `plot_results.py` | Performance visualization and result generation |

---

## Applications

- Healthcare Federated Learning
- Edge AI Systems
- Mobile Device Learning
- IoT Sensor Networks
- Smart Manufacturing
- Financial Fraud Detection
- Autonomous Systems

---

