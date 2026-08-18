# RiskTraf

**RiskTraf: Risk-Extrapolated Residual Learning for Multi-Variate Traffic Flow Prediction**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation of our paper accepted at **CIKM 2026** (*the 35th ACM International Conference on Information and Knowledge Management, November 7–11, 2026, Rome, Italy*), together with the **PEMSB-3V** benchmark.

---

## Overview

Traffic sensors commonly record **flow**, **speed**, and **occupancy**, but standard traffic flow forecasting benchmarks and models rarely exploit all three raw measurements reliably. Although speed and occupancy provide sensor-native traffic-state information beyond flow alone, existing releases often omit these variables, replace them with proxies, or contain logically inconsistent records. Moreover, direct empirical risk minimization over three-variable inputs may exploit regime-dependent shortcuts, as the relationships among flow, speed, and occupancy vary substantially between free-flow and congested states.

This work makes two contributions:

- **PEMSB-3V**: a public benchmark suite of four district-level traffic datasets (PEMS03-B, PEMS04-B, PEMS07-B, PEMS08-B) that preserve raw flow, speed, and occupancy measurements from PeMS detectors, built with a transparent metadata-based screening pipeline.
- **RiskTraf**: a model-agnostic risk-extrapolated residual plug-in. For each trained spatio-temporal backbone, RiskTraf freezes the validation-selected checkpoint and learns a lightweight zero-start residual head from historical speed and occupancy. The residual head constructs ordered traffic-risk environments and optimizes horizon-wise flow corrections with a risk extrapolation (REx) objective, mitigating regime-specific shortcut correlations without modifying the backbone.

Extensive experiments show that RiskTraf consistently improves diverse forecasting backbones (STGCN, DCRNN, AGCRN, Graph WaveNet, GMAN, STEMGNN, STNorm, GTS, STWA, MegaCRN, HimNet, STDN, and more) and outperforms debiasing and distribution-shift adaptation methods.

---

## Project Structure

```
RiskTraf/
├── risktraf/                 # RiskTraf plug-in and training pipeline
│   ├── data.py               # PEMSB-3V loading, scaling, flow-only labels
│   ├── models.py             # BackboneAdapter + CovariateDebiasHead (residual head)
│   ├── risk.py               # Risk environments and REx-style objective
│   ├── trainer.py            # Two-stage paired trainer (Stage I backbone / Stage II RiskTraf)
│   ├── run_matrix.py         # Multi-GPU dataset x backbone matrix runner
│   ├── summarize.py          # Baseline vs RiskTraf result tables
│   └── local_staeformer.py   # STAEformer implementation
├── models/                   # Spatio-temporal backbone implementations
├── lib/
│   └── losses.py             # Backbone-specific auxiliary losses (GTS, MegaCRN)
├── data/                     # PEMSB-3V benchmark data (download separately)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
conda create -n risktraf python=3.10
conda activate risktraf
pip install -r requirements.txt
```

---

## PEMSB-3V Benchmark

PEMSB-3V consists of four district-level datasets built from PeMS detectors, each preserving raw flow, speed, and occupancy measurements:

| Dataset | Nodes | Interval | Variables |
|---------|-------|----------|-----------|
| PEMS03-B | 1,013 | 5 mins | flow, speed, occupancy |
| PEMS04-B | 2,474 | 5 mins | flow, speed, occupancy |
| PEMS07-B | 2,788 | 5 mins | flow, speed, occupancy |
| PEMS08-B | 1,515 | 5 mins | flow, speed, occupancy |

The data is hosted on Google Drive (link will be added here). After downloading, place the files under `data/` with the following layout:

```
data/
├── PEMS03-B.npz              # array `data` of shape (nodes, time, 3): [flow, speed, occupancy]
├── PEMS04-B.npz
├── PEMS07-B.npz
├── PEMS08-B.npz
└── adj_files/
    ├── PEMS03-B_adj.pkl      # adjacency matrices for graph-based backbones
    ├── PEMS04-B_adj.pkl
    ├── PEMS07-B_adj.pkl
    └── PEMS08-B_adj.pkl
```

The `data/aux/` directory is created automatically at training time for derived auxiliary files (e.g., GMAN spatial embeddings).

---

## Quick Start

All commands are run from the repository root.

### Full paired matrix (paper protocol)

For every dataset/backbone pair, this first trains the vanilla three-variable backbone (Stage I), then freezes that checkpoint and trains the RiskTraf residual head on top of it (Stage II):

```bash
python -m risktraf.run_matrix \
  --paired \
  --datasets PEMS03-B PEMS04-B PEMS07-B PEMS08-B \
  --plugins baseline risk \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --workers 4 --epochs 8 --patience 3 \
  --output_dir runs
```

On a single GPU, pass `--devices cuda:0 --workers 1`.

### Single dataset/backbone pair

```bash
# Stage I: vanilla backbone
PYTHONPATH=. python -m risktraf.trainer --dataset PEMS03-B --model STDN --plugin baseline --output_dir runs

# Stage II: RiskTraf residual head on the frozen Stage-I checkpoint
PYTHONPATH=. python -m risktraf.trainer --dataset PEMS03-B --model STDN --plugin risk \
  --init_checkpoint runs/PEMS03-B/STDN/baseline/seed2026_h12/best.pt \
  --skip_backbone_train --calibrate --output_dir runs
```

### Summarize results

```bash
python -m risktraf.summarize --run_dir runs --out runs/summary.md
```

### Smoke test

```bash
python -m risktraf.run_matrix --datasets PEMS03-B --plugins baseline risk --workers 4 --smoke
```

---

## Experimental Setting

- Input: 12 historical steps (1 hour) of flow, speed, and occupancy.
- Output: the next 12 steps of flow only.
- Metrics: MAE, RMSE, and MAPE on inverse-scaled flow values.
- Paired protocol: each `+RiskTraf` result starts from the exact same validation-selected backbone checkpoint as its baseline; if the residual head does not improve validation MAE, it rolls back to the original checkpoint.

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{risktraf2026,
  title     = {RiskTraf: Risk-Extrapolated Residual Learning for Multi-Variate Traffic Flow Prediction},
  author    = {Wang, Guangyu and Liu, Zhidan},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  address   = {Rome, Italy},
  publisher = {ACM}
}
```

---

## Acknowledgements

The backbone implementations in `models/` and `lib/` are adapted from [Torch-MTS](https://github.com/XDZhelheim/Torch-MTS) (Apache License 2.0; see `models/LICENSE-Torch-MTS`). We thank the authors of all backbone models for releasing their code.

## License

This project is licensed under the MIT License.
