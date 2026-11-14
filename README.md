# Neurosymbolic Device Discovery

A research proof-of-concept exploring network device identification under incomplete, noisy, and conflicting observations.

---
 
## At a Glance
 
| | |
|---|---|
| **Core question** | How can a system decide reliably when its observations are incomplete, noisy, or conflicting, and how does it decide what extra evidence is worth paying for? |
| **Testbed** | Identifying network devices (vendor + role) in a controlled simulator |
| **Approaches compared** | Probabilistic models, neural networks, symbolic rules, and adaptive (cost-aware) querying, alone and combined |
| **What this is not** | A production network-discovery tool. Networking is only a convenient, controllable test domain. |
| **Reporting policy** | Negative and inconclusive results will be reported explicitly. |
 
### The idea in plain words
 
Imagine identifying a device by asking it a few questions (probes). Some questions get no answer, some answers are wrong, and each question costs something. The system must choose between three actions at every step:
 
1. **Predict** the device type now.
2. **Ask** one more question (pay its cost).
3. **Abstain** because the evidence is not enough.
The project builds a simulator where this situation can be reproduced exactly, then measures which reasoning methods handle it best.
 
### Worked example
 
A device shows `lldp_enabled = true` and `vendor_hint = Cisco`, but its other features are missing. The evidence is consistent with `cisco_router` and `cisco_switch`. The system can:
 
- guess, with a risk of error;
- query `routing_enabled` (cost 1–4) to separate router from switch;
- abstain.
The research asks which choice is best, and how a model can *know* it is uncertain in the first place.
 
---
 
## Table of Contents
 
1. [Research Motivation](#1-research-motivation)
2. [Problem Formulation](#2-problem-formulation)
3. [Research Questions and Hypotheses](#3-research-questions-and-hypotheses)
4. [Experimental Testbed](#4-experimental-testbed)
5. [Research Methodology](#5-research-methodology)
6. [System Architecture](#6-system-architecture)
7. [Project Structure](#7-project-structure)
8. [Installation and Running Experiments](#8-installation-and-running-experiments)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Current Progress](#10-current-progress)
11. [Limitations and Threats to Validity](#11-limitations-and-threats-to-validity)
12. [Future Research Directions](#12-future-research-directions)
13. [Glossary](#13-glossary)
---
 
## 1. Research Motivation
 
Real-world intelligent systems rarely have perfect information. Observations can be:
 
| Problem | Meaning |
|---|---|
| **Incomplete** | Important features are unavailable or not yet collected |
| **Noisy** | Measurements contain errors |
| **Ambiguous** | Several explanations fit the same evidence |
| **Conflicting** | Different observations support incompatible conclusions |
| **Costly** | Getting more evidence takes time, computation, or communication |
 
Under these conditions, models can become unstable (small changes in evidence flip the prediction), overconfident despite missing data, or too dependent on unreliable inputs.
 
Known tools each help in part: probabilistic models represent uncertainty, neural networks learn patterns, symbolic rules encode domain knowledge, and active learning chooses what to observe next. **Combining them does not automatically improve results.** This project tests each tool, and their combinations, under controlled conditions.
 
### Why network device discovery?
 
- Different devices share similar observable traits.
- Vendor indicators can be missing or misleading.
- Different probes give different amounts of information at different costs.
- Ground truth is known in simulation.
- Uncertainty conditions are reproducible through fixed random seeds.
The domain is deliberately simplified. **The research focus is robust reasoning and evidence acquisition, not networking itself.**
 
---
 
## 2. Problem Formulation
 
A device has an unknown class $Y$. The system starts with a partial set of observations $O_0$ and has a set of probes $Q$ it may run. Each probe $q$ has a cost $c(q)$ and may fail.
 
At each step the system chooses to **predict**, **query**, or **abstain**.
 
An illustrative objective is:
 
$$
\min_{\pi, f} \; \mathbb{E}\Big[ \mathcal{L}\big(f(O_T), Y\big) + \lambda \sum_{t=1}^{T} c(q_t) \Big]
$$
 
| Symbol | Meaning |
|---|---|
| $f$ | the inference model |
| $\pi$ | the query-selection policy |
| $O_T$ | evidence available after querying |
| $\mathcal{L}$ | prediction loss |
| $c(q_t)$ | cost of the probe chosen at step $t$ |
| $\lambda$ | trade-off between accuracy and cost |
 
This is a research formulation. The current implementation does **not** yet optimize it.
 
### What "robustness" means here
 
Robustness is not a single number. It is examined through:
 
- accuracy as missing data increases;
- sensitivity to noisy or conflicting evidence;
- stability under small controlled perturbations;
- confidence calibration;
- ability to recognize insufficient evidence;
- performance under a limited query budget;
- generalization across simulated scenarios.
Accuracy, confidence, stability, and acquisition cost are reported separately.
 
---
 
## 3. Research Questions and Hypotheses
 
Hypotheses are questions to test, not expected positive outcomes.
 
| RQ | Question | Related hypothesis |
|---|---|---|
| **RQ1** Robust inference | How do probabilistic, neural, and symbolic methods behave as observations become incomplete, noisy, or contradictory? | **H1:** Missing and noisy data reduce reliability, differently across models. |
| **RQ2** Symbolic knowledge | When do symbolic constraints help, and when do wrong or over-restrictive ones hurt? | **H2:** Correct constraints help when evidence is insufficient.<br>**H3:** Incorrect constraints cause systematic errors. |
| **RQ3** Uncertainty estimation | Can uncertainty estimates flag unreliable predictions and guide further querying? | (supports H4) |
| **RQ4** Adaptive acquisition | Does uncertainty-aware, cost-sensitive querying beat random or fixed-order querying on the accuracy–cost trade-off? | **H4:** Uncertainty-aware querying can beat random querying in some settings.<br>**H5:** Cost-sensitive selection can cut cost at similar quality. |
| **RQ5** Combined reasoning | Does combining learning, symbolic rules, and adaptive querying beat each component alone? | **H6:** The combination may improve robustness, depending on conditions and assumptions. |
 
---
 
## 4. Experimental Testbed
 
### 4.1 Device taxonomy
 
Four vendors × two roles = eight classes.
 
| Vendor | Labels |
|---|---|
| Cisco | `cisco_router`, `cisco_switch` |
| Juniper | `juniper_router`, `juniper_switch` |
| Arista | `arista_router`, `arista_switch` |
| HPE | `hpe_router`, `hpe_switch` |
 
### 4.2 Observation features
 
| Feature | Type |
|---|---|
| `vendor_hint` | Categorical |
| `routing_enabled` | Boolean |
| `lldp_enabled` | Boolean |
| `snmp_available` | Boolean |
| `interface_count_band` | Low / Medium / High |
| `vlan_capable` | Boolean |
| `firmware_family` | Categorical |
 
The distributions are **illustrative modeling assumptions**, not measurements of real equipment. `vendor_hint` and `firmware_family` are intentionally imperfect, so some devices stay ambiguous.
 
### 4.3 Uncertainty scenarios
 
| Scenario | Missing rate | Noise rate |
|---|---:|---:|
| Easy | 10% | 2% |
| Moderate | 20% | 5% |
| Hard | 35% | 12% |
 
- **Missing** = a probe result is unavailable.
- **Noise** = a successful observation contains a wrong value.
- **Ambiguity** comes from overlapping feature distributions.
- Explicit *conflicting-evidence* scenarios are a planned extension.
### 4.4 Probe costs
 
- **Uniform:** every probe costs the same.
- **Heterogeneous:** costs range from 1 to 4, separating informative probes from cheap ones.
- Failed probes still incur their cost.
### 4.5 Dataset
 
Default: 200 devices per class, class-balanced, with disjoint device IDs across splits.
 
| Split | Devices | Share |
|---|---:|---:|
| Training | 1,120 | 70% |
| Validation | 240 | 15% |
| Test | 240 | 15% |
| **Total** | **1,600** | **100%** |
 
By default two probes are chosen as the initial observation. Some may fail, so fewer than two values may be visible.
 
---
 
## 5. Research Methodology
 