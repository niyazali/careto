"""Publication figures from saved predictions; no additional model fits."""

import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

B = Path(__file__).resolve().parent
O = B / "robustness_results"
O.mkdir(exist_ok=True)
F = B / "figures"
F.mkdir(exist_ok=True)
s = json.loads((B / "saved_results_core/RESULT_SUMMARY.json").read_text())
schemes = ["random_row", "original_study", "category_unpurged", "category_study_purged"]
labels = [
    "Random\nrows",
    "Held-out\nstudies",
    "Held-out\nmatrix",
    "Matrix and\nstudies withheld",
]
models = [
    "Ridge",
    "ElasticNet",
    "RandomForest",
    "GradientBoosting",
    "SVR",
    "KNN",
    "XGBoost",
]
names = [
    "Ridge",
    "Elastic net",
    "Random forest",
    "Gradient boosting",
    "Support vector regression",
    "Nearest neighbors",
    "XGBoost",
]
get = lambda m, k: next(r for r in s["summary"] if r["model"] == m and r["scheme"] == k)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def save(fig, n):
    for e in ["png", "pdf", "svg"]:
        fig.savefig(F / (n + "." + e), dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


fig, (a, b) = plt.subplots(
    2,
    1,
    figsize=(8.8, 7.4),
    gridspec_kw={"height_ratios": [1, 1.75]},
    layout="constrained",
)
x = np.arange(4)
for m, c, off, name in [
    ("RandomForest", "#0072B2", -0.08, "Random forest"),
    ("training_median", "#555555", 0.08, "Training median"),
]:
    vals = np.array([get(m, k)["study_MAE"] for k in schemes])
    ci = np.array([get(m, k)["study_MAE_conditional_CI"] for k in schemes])
    a.errorbar(
        x + off,
        vals,
        yerr=[vals - ci[:, 0], ci[:, 1] - vals],
        fmt="o-",
        color=c,
        capsize=3,
        label=name,
    )
a.set_xticks(x, labels)
a.set_ylabel("Equal-study MAE (pp)")
a.set_title("A  Validation performance", loc="left", fontweight="bold")
a.legend(loc="upper left", frameon=False, ncol=2)
a.set_ylim(
    0,
    max(
        get(m, k)["study_MAE_conditional_CI"][1]
        for m in ["RandomForest", "training_median"]
        for k in schemes
    )
    * 1.18,
)
a.grid(axis="y", alpha=0.15)
vals = np.array(
    [
        [
            next(
                r["delta_study_MAE"]
                for r in s["paired_comparisons"]
                if r["A"] == k + "/" + m and r["B"] == k + "/training_median"
            )
            for k in schemes
        ]
        for m in models
    ]
)
lim = max(abs(vals.min()), abs(vals.max()))
im = b.pcolormesh(
    np.arange(5) - 0.5,
    np.arange(8) - 0.5,
    vals,
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
    shading="flat",
)
b.set_xlim(-0.5, 3.5)
b.set_ylim(6.5, -0.5)
b.set_xticks(x, labels)
b.set_yticks(np.arange(7), names)
b.tick_params(length=0)
b.set_title("B  Learner minus median MAE", loc="left", fontweight="bold")
for i in range(7):
    for j in range(4):
        b.text(
            j,
            i,
            f"{vals[i,j]:.2f}",
            ha="center",
            va="center",
            color="white" if abs(vals[i, j]) > 0.6 * lim else "black",
            fontsize=12.5,
        )
fig.colorbar(
    im, ax=b, label="Difference (pp); negative favors learner", fraction=0.05, pad=0.03
)
save(fig, "Figure_2")
with (B / "saved_results_core/heldout_predictions.csv").open() as f:
    p = [
        {**r, "target": float(r["target"]), "prediction": float(r["prediction"])}
        for r in csv.DictReader(f)
        if r["scheme"] == "category_study_purged"
    ]
cats = [
    "algae_microbial",
    "animal_dairy",
    "cereals_legumes",
    "engineered_carriers",
    "extracted_oils",
    "fruits_fruiting_vegetables",
    "leafy_brassica",
    "mixed_meals",
    "roots_tubers",
]
cn = [
    "Algae / microbial",
    "Animal / dairy",
    "Cereals / legumes",
    "Engineered carriers",
    "Extracted oils",
    "Fruits / fruiting\nvegetables",
    "Leafy / brassica",
    "Mixed meals",
    "Roots / tubers",
]
rf = get("RandomForest", "category_study_purged")["by_category"]
base = get("training_median", "category_study_purged")["by_category"]
fig, (a, b) = plt.subplots(
    1,
    2,
    figsize=(10.5, 6.6),
    sharey=True,
    gridspec_kw={"width_ratios": [1, 1.35]},
    layout="constrained",
)
y = np.arange(len(cats))
a.plot(
    [rf[c]["study_MAE"] for c in cats],
    y - 0.09,
    "o",
    color="#0072B2",
    label="Random forest",
)
a.plot(
    [base[c]["study_MAE"] for c in cats],
    y + 0.09,
    "s",
    color="#555555",
    label="Training median",
)
a.set_yticks(
    y,
    [
        f'{n}\n{rf[c]["rows"]} observations, {rf[c]["studies"]} {"study" if rf[c]["studies"]==1 else "studies"}'
        for n, c in zip(cn, cats)
    ],
    fontsize=12.5,
)
a.invert_yaxis()
a.set_xlim(0, max(max(rf[c]["study_MAE"], base[c]["study_MAE"]) for c in cats) * 1.2)
a.set_title("A  Category mean error", loc="left", fontweight="bold")
a.set_xlabel("Equal-study MAE (pp)")
fig.legend(
    *a.get_legend_handles_labels(),
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.65, -0.065),
    ncol=2,
    fontsize=11,
)
spread = []
for i, c in enumerate(cats):
    groups = defaultdict(list)
    for r in p:
        if r["matrix_category"] == c and r["model"] == "RandomForest":
            groups[r["study_group"]].append(r)
    gs = sorted(groups)
    jitter = np.zeros(len(gs)) if len(gs) == 1 else np.linspace(-0.22, 0.22, len(gs))
    for g, j in zip(gs, jitter):
        rr = groups[g]
        mae = float(np.mean([abs(r["target"] - r["prediction"]) for r in rr]))
        spread.append(
            {
                "Category": c,
                "Study group": g,
                "Model": "RandomForest",
                "Observations": len(rr),
                "MAE (pp)": mae,
            }
        )
        b.scatter(
            mae,
            i + j,
            s=30,
            color="#0072B2",
            alpha=0.8,
            edgecolors="white",
            linewidths=0.35,
        )
b.set_xlim(0, max(r["MAE (pp)"] for r in spread) * 1.12)
b.set_xlabel("Within-study MAE (pp)")
b.set_title("B  Errors of contributing studies", loc="left", fontweight="bold")
for ax in [a, b]:
    ax.grid(axis="x", alpha=0.15)
    ax.tick_params(axis="y", length=0)
save(fig, "Figure_3")
with (O / "category_study_errors.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(spread[0]))
    w.writeheader()
    w.writerows(spread)
print("Two figures saved in PNG, PDF and SVG; category-study errors saved.")
