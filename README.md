[README.md](https://github.com/user-attachments/files/28395541/README.md)
# Data Leakage and a Rigorous Benchmark of ML Methods for Predicting Standard Enthalpy of Formation (ΔH<sub>f</sub>°)

Reproducible code for the data pipeline, the **data‑leakage analysis**, and the **SchNet** graph‑neural‑network benchmark used in:

> **Foundation Models, Graph Neural Networks, and Data Leakage: A Rigorous Benchmark of ML Methods for Predicting Standard Enthalpy of Formation**
> Ali A. Khairbek and Renjith Thomas.

The whole study can be reproduced by running **one file, top to bottom** — either the notebook or the script. They contain the **same** pipeline: they download the datasets, detect and remove data leakage, build a leakage‑free external benchmark, train SchNet, and save every result automatically. **No manual steps in between.**

---

## ⚡ Run it in one go

> **You only need to run ONE of the two files below — they are identical.** Pick the notebook *or* the script.
> A **GPU is strongly recommended** (SchNet trains for 100 epochs). The easiest, zero‑install option is a free GPU notebook on **Google Colab** or **Kaggle**.

### Option 1 — Notebook  ▸ *recommended, zero setup*

1. Open **`reproduce_all_clean.ipynb`** in **Google Colab** (`File ▸ Upload notebook`) or upload it to a **Kaggle** notebook.
2. Turn on the GPU:
   * **Colab:** `Runtime ▸ Change runtime type ▸ T4 GPU`
   * **Kaggle:** `Settings ▸ Accelerator ▸ GPU`
3. Run everything at once:
   * **Colab:** `Runtime ▸ Run all`
   * **Kaggle:** `Run All`

That is the entire process. The notebook installs its own dependencies, downloads the data, trains the model, and writes the outputs. Just wait for it to finish.

### Option 2 — Script

`reproduce_all_clean.py` is the same pipeline flattened into a single script.

**Easiest (Colab / Kaggle / Jupyter — same one‑click experience):**
open a new notebook with a **GPU** runtime and load the whole script into one cell, then run it:

```python
%load reproduce_all_clean.py
# press Run/Shift+Enter once more to execute the whole file
```

**On your own machine (Linux/Mac/Windows with a CUDA GPU):**
install the dependencies once, then run the notebook end‑to‑end with a single command:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace reproduce_all_clean.ipynb
```

> ℹ️ The `.py` file keeps the Colab/Kaggle install shortcuts (lines starting with `!pip …`) and writes to the Colab/Kaggle working folder, so running it as a *plain* `python reproduce_all_clean.py` outside a notebook will need those `!pip` lines removed and the data path adjusted first. If you want a version that runs anywhere with just `python reproduce_all.py`, open an issue — a portable single‑file runner can be provided.

---

## Requirements

* **Python ≥ 3.10**
* **A CUDA GPU** (training on CPU is possible but very slow)
* Python packages (installed automatically inside the notebook; for a local run use the list below):

```text
numpy
pandas
requests
tqdm
pyarrow
rdkit
thermochem
torch
torch-geometric
matplotlib
```

> `torch` and `torch-geometric` are sensitive to your CUDA version. On Colab/Kaggle this is handled for you. For a local install, follow the official guides for
> [PyTorch](https://pytorch.org/get-started/locally/) and
> [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## What the pipeline does

Running the file executes four stages in order:

1. **Build the dataset.** Downloads and parses three sources and computes the regression target ΔH<sub>f</sub>°(298 K):
   * **QM9** — ~134 k DFT molecules (H, C, N, O, F); ΔH<sub>f</sub>° is derived from the reported U₀ / H₂₉₈ using atomic reference energies and thermal corrections.
   * **tmQM (2024)** — ~108 k organometallic molecules (kept for reference/extension).
   * **Burcat database** — ~3 000 experimental species; ΔH<sub>f</sub>°(298) is computed from the NASA‑7 polynomials.
   The three sources are written as `.parquet` files.

2. **Detect and remove data leakage.** Every benchmark molecule is matched against QM9 by **InChIKey connectivity skeleton**. Molecules present in QM9 are flagged as *leaked* and **excluded from training**.

3. **Build a leakage‑free external benchmark** of **25 molecules** — 7 small clean molecules (≤ 9 heavy atoms, confirmed absent from QM9) and 18 large molecules (> 9 heavy atoms, guaranteed outside QM9). The PyG dataset is saved to disk.

4. **Train and evaluate SchNet.** A SchNet GNN (`hidden_channels=128`, `num_filters=128`, `num_interactions=6`, `num_gaussians=50`, `cutoff=10 Å`, with atomic references) is trained for **100 epochs** on the leakage‑corrected QM9 split, then evaluated **in‑distribution** and on the **leakage‑free external benchmark**.

---

## Outputs

Everything is written to the working directory (`/content/…` on Colab, `/kaggle/working/…` on Kaggle):

| File | Description |
|------|-------------|
| `enthalpy_data/qm9_with_deltaHf.parquet` | QM9 with computed ΔH<sub>f</sub>° (training target) |
| `enthalpy_data/tmqm.parquet` | tmQM organometallic dataset |
| `enthalpy_data/burcat.parquet` | Burcat experimental ΔH<sub>f</sub>° values |
| `enthalpy_data.zip` | The full data folder, zipped |
| `ml_results/pyg_data_clean.pt` | Leakage‑free PyG dataset (train/val/test + clean benchmark) |
| `ml_results/schnet_clean.pt` | Trained SchNet weights + training history |
| `ml_results/external_clean.csv` | Per‑molecule SchNet predictions on the external benchmark |
| `ml_results/final_results.png` | Summary figure: training curve, in‑distribution parity, external‑benchmark parity |

**Expected key numbers** (reproduced by this code): SchNet reaches an in‑distribution test MAE of ≈ 0.79 kcal mol⁻¹ (R² ≈ 0.9995), but its MAE on the leakage‑free external benchmark rises to ≈ 13.5 kcal mol⁻¹ — the central result demonstrating how leakage inflates apparent accuracy.

---

## Repository contents

```
.
├── reproduce_all_clean.ipynb   # the full pipeline as a notebook  (run this)
├── reproduce_all_clean.py      # the same pipeline as a script    (or run this)
├── requirements.txt            # Python dependencies (for local runs)
└── README.md
```

---

## Scope / notes

* This repository reproduces the **dataset construction**, the **leakage analysis**, and the **SchNet** benchmark (the SchNet rows of the method comparison and the SchNet figure).
* The **foundation‑model calibration** (MACE‑OFF23 + Gaussian‑process correction) described in the paper’s Methods is a **separate procedure** and is *not* part of these two files.
* All datasets are downloaded directly from their original public sources at runtime — nothing large is stored in this repository.

---

## Data sources & licenses

* **QM9** — Ramakrishnan et al., *Sci. Data* **1**, 140022 (2014). Distributed via Figshare.
* **tmQM** — Balcells & Skjelstad, *J. Chem. Inf. Model.* **60**, 6135 (2020); 2024 update by the UiO CompCat group.
* **Burcat thermochemical database** — A. Burcat & B. Ruscic (hosted at ELTE).

Please cite the original dataset papers if you use them. Each dataset is subject to its own license/terms.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{Khairbek_Thomas_thermochem,
  title  = {Foundation Models, Graph Neural Networks, and Data Leakage:
            A Rigorous Benchmark of ML Methods for Predicting
            Standard Enthalpy of Formation},
  author = {Khairbek, Ali A. and Thomas, Renjith},
  year   = {2025}
}
```

*(Update the year / journal / DOI once published.)*
