# Neurosymbolic Device Discovery

A research proof-of-concept exploring network device identification under incomplete, noisy, and conflicting observations.

The project investigates how probabilistic classification, symbolic reasoning, and adaptive evidence gathering can be combined to identify simulated network devices.

> **Status:** Synthetic dataset generation and preliminary symbolic-probabilistic baseline. A PyTorch neural classifier and uncertainty-aware query selection are planned as subsequent experiments.

## Table of Contents

1. [Motivation](#1-motivation)
2. [Research Questions](#2-research-questions)
3. [Scope and Assumptions](#3-scope-and-assumptions)
4. [Device Taxonomy](#4-device-taxonomy)
5. [Architecture](#5-architecture)
6. [Project Structure](#6-project-structure)
7. [Installation](#7-installation)
8. [Running Experiments](#8-running-experiments)
9. [Evaluation Plan](#9-evaluation-plan)
10. [Limitations](#10-limitations)
11. [Future Work](#11-future-work)

## 1. Motivation

This project is motivated by practical challenges encountered in multi-vendor network discovery, where device information may be incomplete, inconsistent, or unavailable.

Traditional heuristic-based identification can be useful, but it raises questions about how to combine uncertain evidence and decide which additional information is worth collecting.

The goal of this POC is to investigate these questions through controlled experiments rather than assume that a hybrid approach will necessarily outperform simpler baselines.

## 2. Research Questions

The main research question is:

> To what extent can combining neural predictions, symbolic constraints, and uncertainty-aware query selection improve network device identification under incomplete or conflicting observations?

The experiments will investigate:

1. When symbolic constraints improve or degrade classification accuracy.
2. Whether adaptive query selection offers advantages over fixed or random querying.
3. How different approaches behave under missing, noisy, and conflicting evidence.

## 3. Scope and Assumptions

All devices and observations are synthetic. No real hardware, network scanning, or live traffic is involved.

The simulation uses:

- Four vendors: Cisco, Juniper, Arista, and HPE.
- Two device roles: Router and Switch.
- Eight vendor-role classification labels.
- Seven simulated observation features.
- Configurable missing and noisy observations.
- Reproducible random seeds.

> **Note:** The feature distributions are illustrative assumptions, not empirical measurements of real vendor equipment.

## 4. Device Taxonomy

| Vendor  | Roles           | Example Labels                |
|---------|-----------------|-------------------------------|
| Cisco   | Router, Switch  | `cisco_router`, `cisco_switch`   |
| Juniper | Router, Switch  | `juniper_router`, `juniper_switch` |
| Arista  | Router, Switch  | `arista_router`, `arista_switch`  |
| HPE     | Router, Switch  | `hpe_router`, `hpe_switch`      |

### Observation Features

| Feature                | Type                  |
|------------------------|-----------------------|
| `vendor_hint`          | Categorical           |
| `routing_enabled`      | Boolean               |
| `lldp_enabled`         | Boolean               |
| `snmp_available`       | Boolean               |
| `interface_count_band` | Low / Medium / High   |
| `vlan_capable`         | Boolean               |
| `firmware_family`      | Categorical           |

By default, two of the seven possible probes are initially selected for each device. Unavailable or unrevealed observations are represented as missing values in the generated CSV.

## 5. Architecture

The project separates data generation, observation simulation, inference, and evaluation.

### Data Generation and Preliminary Inference Baseline

```text
Partial Observations
         |
         +----> Symbolic Classifier
         |      (domain-rule scores)
         |
         +----> Naive Bayes Classifier
                (class probabilities)
         |
         v
Weighted Score Fusion
         |
         v
Predicted Device Class
```

The Naive Bayes classifier is a probabilistic baseline, not a neural network. The preliminary fusion uses a weighted geometric mean of the component scores.

### Planned Extensions

The subsequent research implementation will introduce:

- A PyTorch multilayer perceptron (MLP).
- Hard and soft symbolic constraints.
- Predictive entropy as an uncertainty heuristic.
- Query selection based on estimated information gain and simulated cost.
- Fixed-order and random-query baselines.
- Ablation studies across the proposed components.

## 6. Project Structure

```text
neurosymbolic-device-discovery/
├── src/
│   ├── core/
│   │   ├── domain/
│   │   └── utils/constants/
│   ├── data/
│   │   └── builder.py
│   └── device_discovery/
│       ├── cli.py
│       ├── simulation/
│       │   ├── simulator.py
│       │   └── profiles.py
│       ├── data/
│       │   └── materializer.py
│       └── engine/
│           ├── symbolic.py
│           ├── neural.py
│           ├── fusion.py
│           ├── evaluator.py
│           └── runner.py
├── tests/
├── data/generated/
├── conftest.py
├── pyproject.toml
└── .gitignore
```

## 7. Installation

**Requirements:**

- Python 3.10 or later.
- pip.

Clone the repository:

```bash
git clone <repository-url>
cd neurosymbolic-device-discovery
```

Create a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

The preliminary implementation uses the Python standard library for its classifiers. Later neural experiments will require PyTorch and related dependencies.

## 8. Running Experiments

### Generate the Synthetic Dataset

```bash
python -m device_discovery.cli --seed 42
```

**Default configuration:**

| Parameter               | Value |
|-------------------------|-------|
| Random seed             | 42    |
| Samples per class       | 200   |
| Training split          | 70%   |
| Validation split        | 15%   |
| Test split              | 15%   |
| Initial selected probes | 2     |
| Missing rate            | 0.20  |
| Noise rate              | 0.05  |

**Expected dataset sizes:**

| Split      | Devices |
|------------|--------:|
| Train      | 1,120   |
| Validation | 240     |
| Test       | 240     |
| **Total**  | **1,600** |

**Generated artifacts:**

- `train.csv`, `val.csv`, `test.csv`: Partial observations and labels.
- `ground_truth.json`: True device profiles used by the simulator and evaluation.
- `probe_outcomes.json`: Precomputed simulated query outcomes.
- `manifest.json`: Dataset configuration and metadata.

> **Important:** Ground-truth labels and unrevealed probe outcomes must not be accessible to inference or query-selection logic.

### Run the Preliminary Classifier Evaluation

```bash
python -m device_discovery.engine.runner
```

The intended comparison includes:

- Symbolic-only baseline.
- Naive Bayes-only baseline.
- Symbolic-probabilistic fusion.

An initial fusion configuration uses symbolic weight `0.40` and probabilistic weight `0.60`. These weights are experimental settings, not established optimal values.

> **Note:** Training-set results are diagnostic. Validation data should be used for model selection and tuning, while the held-out test set should be reserved for final evaluation.

### Run Tests

```bash
python -m pytest tests/ -v
```

Tests cover reproducibility, observation noise, missing values, split integrity, dataset materialization, and inference components where implemented.

## 9. Evaluation Plan

The full POC will report:

- Classification accuracy.
- Macro-F1 score.
- Average number of queries.
- Average simulated query cost.
- Unresolved prediction rate.
- Accuracy versus query budget.

Experiments will vary observation availability, noise, conflicting evidence, and query budgets.

The study will also examine cases where symbolic constraints or adaptive querying fail to improve results.

## 10. Limitations

- The dataset is synthetic and may not represent production network environments.
- Device classes and features are simplified.
- Vendor and firmware indicators may make some classification cases unusually easy.
- Naive Bayes assumes conditional feature independence.
- Symbolic rules depend on manually specified assumptions.
- Fusion weights and scores require validation and potentially calibration.
- Current baseline results do not establish the effectiveness of the planned neural-symbolic or adaptive-query approaches.

## 11. Future Work

The next steps are to implement a PyTorch MLP, compare it with the probabilistic baseline, integrate symbolic constraints, and evaluate adaptive query selection under controlled query budgets.

The long-term goal is to understand when learned predictions, explicit domain knowledge, and selective evidence gathering complement one another, and when simpler methods are sufficient.