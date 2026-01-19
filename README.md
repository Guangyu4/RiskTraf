# 🚦 RiskTraf

**Risk Extrapolation Framework for Multi-Variate Traffic Prediction**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

Traffic speed and occupancy can be simultaneously collected alongside traffic flow by modern sensors, yet they have long been overlooked in flow prediction tasks. Since these variables originate from the same physical traffic process and jointly observe the underlying road state, information theory suggests that incorporating them should yield information gain and enhance prediction accuracy.

However, **naively incorporating them as covariates paradoxically degrades model performance**. We identify the root cause:

> During traffic incidents, speed drops sharply, causing vehicles to pass over sensors more slowly and thus occupy them for longer durations—this mechanically inflates occupancy readings even when flow remains relatively stable, creating **spurious correlations** that confuse conventional models.

Traditional deep learning methods, built upon empirical risk minimization, inherently fail to handle such distribution shifts between normal and abnormal traffic conditions.

---

## 🎯 RiskTraf Framework

To address this challenge, we propose **RiskTraf**, a risk extrapolation framework for traffic prediction that:

- 📊 **Partitions data** into distinct traffic environments
- ⚖️ **Minimizes worst-case risk** across all environments
- 🔄 **Handles distribution shifts** between normal and abnormal conditions

---

## 📂 Project Structure

```
PEMSB/
├── core/                 # Core components
│   ├── dataset.py        # Data loading and preprocessing
│   ├── STAEformer.py     # STAEformer backbone
│   ├── steve_model.py    # STEVE model implementation
│   └── train.py          # Base training script
│
├── risktraf/             # RiskTraf model
│   ├── train_risktraf.py # Training script
│   └── eval_risktraf.py  # Evaluation script
│
├── baseline/             # Baseline experiments
├── rex/                  # Risk extrapolation experiments
├── ablation/             # Ablation studies
├── causal/               # Causal analysis experiments
├── ood/                  # Out-of-distribution experiments
├── backbone/             # Backbone comparison experiments
├── megacrn/              # MegaCRN experiments
├── evaluation/           # Evaluation utilities
└── scripts/              # Shell scripts for running experiments
```

---

## 📊 Dataset

We contribute the **first publicly available multi-variate traffic dataset** that includes:

| Variable | Description |
|----------|-------------|
| 🚗 **Flow** | Number of vehicles passing the sensor |
| 🏎️ **Speed** | Average speed of vehicles |
| ⏱️ **Occupancy** | Percentage of time the sensor is occupied |

---

## 🚀 Quick Start

### Training RiskTraf

```bash
cd risktraf
python train_risktraf.py --dataset PEMS04-B
```

### Evaluation

```bash
python eval_risktraf.py --dataset PEMS04-B
```

---

## 📈 Results

Our approach consistently outperforms:
- ✅ Classical spatio-temporal models
- ✅ Debiasing-based approaches  
- ✅ Distribution shift adaptation methods

---

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{risktraf2026,
  title={RiskTraf: Risk Extrapolation for Multi-Variate Traffic Prediction},
  author={Anonymous},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License.
