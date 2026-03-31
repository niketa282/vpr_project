# 🗺️ Visual Place Recognition (VPR) Pipeline

A modular Python pipeline for Visual Place Recognition using image descriptors,
cosine similarity matching, and standard retrieval evaluation metrics.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Layout](#project-layout)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This pipeline takes a **database** of reference images and a **query set**, extracts
global descriptors using a pretrained backbone (DINOv2), builds a pairwise similarity
matrix, performs top-K matching, and evaluates results using Precision\@K, Recall\@K,
and mAP.

```
Query Image  ──┐
               ├──► Descriptor ──► Similarity Matrix S ──► Matching ──► Evaluation
Database DB  ──┘    Computation     ℝ|DB|×|Q|
```

---

## Project Layout

```
vpr_project/
│
├── main.py                     # Entry point — orchestrates the full pipeline
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
└── vpr/                        # Core package (note the __init__.py)
    ├── __init__.py             # Makes vpr/ a importable Python package
    ├── dataset.py              # read_images()
    ├── descriptor.py           # compute_descriptor()
    ├── similarity.py           # compute_similarity_matrix()
    ├── matching.py             # match()
    ├── metrics.py              # compute_precision_recall()
    └── evaluation.py           # evaluate(), print_metrics()
```

---

## Installation

> Tested on **WSL2 + Windows 11** with Python 3.10+

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/vpr_project.git
cd vpr_project
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate        # WSL / Linux / macOS
# .venv\Scripts\activate         # Windows CMD
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

**Run on a real dataset**

```bash
python main.py --db-dir path/to/database --query-dir path/to/queries
```

**Run with a generated dummy dataset (no real data needed)**

```bash
python main.py --generate-dummy
```

**All options**

```
--db-dir          Path to reference/database images  (default: dataset/database)
--query-dir       Path to query images               (default: dataset/queries)
--gt-file         Path to ground truth .npy file     (optional)
--top-k           Number of top matches to retrieve  (default: 10)
--batch-size      Images per forward pass            (default: 16)
--out-dir         Where to save outputs              (default: vpr_output/)
```

---

## Modules

| File | Function | Description |
|------|----------|-------------|
| `vpr/dataset.py` | `read_images()` | Loads and preprocesses images from database and query directories |
| `vpr/descriptor.py` | `compute_descriptor()` | Extracts L2-normalised global descriptors via DINOv2 backbone |
| `vpr/similarity.py` | `compute_similarity_matrix()` | Builds **S ∈ ℝ\|DB\|×\|Q\|** using cosine similarity |
| `vpr/matching.py` | `match()` | Returns top-K database candidates per query |
| `vpr/metrics.py` | `compute_precision_recall()` | Computes Precision\@K and Recall\@K |
| `vpr/evaluation.py` | `evaluate()`, `print_metrics()` | Full evaluation against ground truth + mAP |

---

## Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Recall\@K** | `hits(K) / N_queries` | Did at least one correct match appear in top-K? |
| **Precision\@K** | `true_positives(K) / K` | Of K returned results, how many were correct? |
| **mAP** | `mean AP over all queries` | Area under per-query Precision–Recall curve |

---

## Contributing

Pull requests are welcome. For major changes please open an issue first.

---

## License

[MIT](https://choosealicense.com/licenses/mit/)