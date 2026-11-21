# ASBD End-Semester Hackathon — Power Consumption of Tetouan City

**Student:** Jeswin Tobias J | **Roll No:** CS23I1013
**Course:** Analytics & Systems of Big Data Practice, IIITDM Kancheepuram (Nov 2025)
**Dataset:** [UCI #849 — Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city) (52,416 rows, 10-min intervals, 2017)

## Layout
| Path | Contents |
|---|---|
| `Dataset/` | Raw CSV (untouched; UCI, CC BY 4.0) |
| `src/common.py` | Shared paths, column names, loader, `save_fig`, `timer` |
| `notebooks/` | One executed notebook per phase |
| `plots/` | Every figure, numbered `NN_name.png` |
| `outputs/` | Cleaned data, transactions, rule tables, timings (CSV) |
| `report/` | Final PDF report + sources |

## Phases
| # | Notebook | Task |
|---|---|---|
| 1 | `01_preprocessing.ipynb` | Cleaning, feature engineering, selection, transformation (plots 01–06) |
| 2 | `02_visualization.ipynb` | 14 course plot types with inferences (plots 07–20) |
| 3 | `03_dic.ipynb` | DIC from scratch (`src/arm.py`), rules, sensitivity (plots 21–25) |
| 4 | `04_diana.ipynb` | DIANA from scratch (`src/diana.py`) on 72-col daily profiles, dendrogram, 5 day types (plots 26–30) |
| 5 | `05_cofi.ipynb` | COFI from scratch (`src/cofi.py`), CFI/MFI, closed (Min-Max) rules, performance (plots 31–34) |
| 6 | `06_pyspark.ipynb` | DIC + COFI on PySpark (`src/spark_arm.py`), 1×–40× scaling, naive vs combiner COFI, 1–10 core scaling (plots 35–37) |
| 7 | `report/` | PDF report |

## Phase 1 outputs (`outputs/`)
| File | Used by |
|---|---|
| `clean_features.csv` | cleaned 52,416 rows + engineered features, `load_level` target |
| `transactions.txt` | one transaction per line (35 distinct items) — DIC, COFI |
| `daily_profiles.csv` | 364 days × (72 zone-hour clustering columns `z1_h00…z3_h23` + 24 city columns for plots + day labels) — DIANA |
| `feature_sets.csv` | selected KNN features and ARM attributes |
| `cleaning_log.csv` | rows changed per cleaning step |

## Setup (after cloning)
**Needs:** Python 3.12 and Java 17 or 21 (for PySpark). Check with `python --version` and `java -version`.

```bash
git clone https://github.com/jeswintobias/asbd-tetouan-power-analytics.git
cd asbd-tetouan-power-analytics
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name asbd-tetouan
```

No Java yet? macOS: `brew install openjdk@21` · Ubuntu: `sudo apt install openjdk-21-jdk` · Windows: install Temurin 21 and set `JAVA_HOME`.

## Run
Open the notebooks from `notebooks/` (they use `sys.path.append('..')` and relative paths) and run them **in order**, 01 → 06: 01 writes the `outputs/` files the others read.

```bash
cd notebooks
jupyter lab                         # interactive, or re-run everything:
for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```
Every plot is saved to `plots/` (300 dpi) and every table to `outputs/`. Run times in 06 depend on the machine; itemsets and counts do not.
