"""
ASBD End-Semester Hackathon — Power Consumption of Tetouan City
Student : Jeswin Tobias J  |  Roll No : CS23I1013
Course  : Analytics & Systems of Big Data, IIITDM Kancheepuram

Shared paths, constants and helpers used by every notebook.
"""
import os
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

STUDENT = "Jeswin Tobias J"
ROLL_NO = "CS23I1013"
SEED = 42

# Project paths (resolved from this file, so notebooks can run from anywhere)
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "Dataset" / "Tetuan City power consumption.csv"
PLOTS = ROOT / "plots"
OUTPUTS = ROOT / "outputs"
for d in (PLOTS, OUTPUTS):
    os.makedirs(d, exist_ok=True)

# Short column names: raw file has double spaces and long names
RENAME = {
    "DateTime": "datetime",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Wind Speed": "wind_speed",
    "general diffuse flows": "gen_diffuse",
    "diffuse flows": "diffuse",
    "Zone 1 Power Consumption": "zone1",
    "Zone 2  Power Consumption": "zone2",
    "Zone 3  Power Consumption": "zone3",
}
WEATHER = ["temperature", "humidity", "wind_speed", "gen_diffuse", "diffuse"]
ZONES = ["zone1", "zone2", "zone3"]


# Plot palette (colorblind-validated order; first 3 safe for all-pairs use)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ZONE_COLORS = dict(zip(ZONES, CAT[:3]))
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]          # one-hue blue ramp
DIV = ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#c22f2f"]          # blue - gray - red


def set_style():
    """Uniform, recessive plot styling for all notebooks."""
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    sns.set_theme(style="whitegrid", palette=CAT, font_scale=0.95)
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": "#e6e5e1", "axes.edgecolor": "#b9b8b2", "lines.linewidth": 1.6})
    for name, cols in (("seq_blue", SEQ), ("div_br", DIV)):
        if name not in plt.colormaps():
            plt.colormaps.register(LinearSegmentedColormap.from_list(name, cols))


def load_raw() -> pd.DataFrame:
    """Read the raw CSV, rename columns, parse datetime."""
    df = pd.read_csv(DATA_RAW).rename(columns=RENAME)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%m/%d/%Y %H:%M")
    return df


def save_fig(name: str, fig=None):
    """Save the current (or given) figure to plots/<name>.png at 300 dpi."""
    (fig or plt.gcf()).savefig(PLOTS / f"{name}.png", dpi=300, bbox_inches="tight")


@contextmanager
def timer(label: str, store: dict | None = None):
    """Time a code block; optionally record seconds in `store[label]`."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if store is not None:
        store[label] = dt
    print(f"[{label}] {dt:.3f} s")


def header():
    """Print the student banner at the top of each notebook."""
    print(f"{STUDENT} | {ROLL_NO} | ASBD Hackathon — Tetouan Power Consumption")
