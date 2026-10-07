"""Check sensitivity to removing the largest studies using nested validation.
Run from the released code directory. Results are saved separately from original fits.
Five studies are selected by observation count before inspecting deletion results.
"""

import csv, json, hashlib, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import run_validation as v

B = Path(__file__).resolve().parent
O = B / "robustness_results"
O.mkdir(exist_ok=True)


def save(name, obj):
    (O / name).write_text(json.dumps(obj, indent=2, allow_nan=False))


def write(name, rows):
    with (O / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


rows = [
    r
    for r in json.loads((B / "COHORT.json").read_text())["rows"]
    if r["cohort"] == "core"
]
cfg = json.loads((B / "analysis_config.json").read_text())
counts = Counter(r["study_group"] for r in rows)
top = sorted(counts, key=lambda g: (-counts[g], g))[:5]
plan = {
    "design": "post hoc largest-study deletion; retune and refit within every remaining primary fold",
    "selection": "five largest original-study groups, count descending, identifier ascending for ties",
    "selected_studies": [{"study_group": g, "observations": counts[g]} for g in top],
    "models": ["RandomForest", "SVR", "training_median"],
    "seed": cfg["seed"],
    "scope": "Results for the previously specified random forest and SVR sensitivity families; conditional intervals omit refitting uncertainty",
    "input_hashes": {
        n: hashlib.sha256((B / n).read_bytes()).hexdigest()
        for n in [
            "COHORT.json",
            "analysis_config.json",
            "saved_results_core/heldout_predictions.csv",
        ]
    },
}
save("PLAN.json", plan)
with (B / "saved_results_core/heldout_predictions.csv").open() as f:
    preds = [
        {**r, "target": float(r["target"]), "prediction": float(r["prediction"])}
        for r in csv.DictReader(f)
        if r["scheme"] == "category_study_purged"
    ]
by = defaultdict(list)
for r in preds:
    by[r["model"]].append(r)
study = []
fixed = []
for g in sorted(counts):
    for m in by:
        a = [r for r in by[m] if r["study_group"] == g]
        met = v.metrics(
            [r["target"] for r in a], [r["prediction"] for r in a], [g] * len(a)
        )
        study.append(
            {
                "Study group": g,
                "Model": m,
                "Observations": len(a),
                "Categories": "; ".join(sorted(set(r["matrix_category"] for r in a))),
                "Study MAE (pp)": met["study_MAE"],
                "Mean observed (%)": float(np.mean([r["target"] for r in a])),
                "Mean predicted (%)": float(np.mean([r["prediction"] for r in a])),
            }
        )
        aa = [r for r in by[m] if r["study_group"] != g]
        bb = [r for r in by["training_median"] if r["study_group"] != g]
        pair = v.paired_delta(aa, bb, cfg["bootstrap_replicates"], cfg["seed"])
        fixed.append(
            {
                "Omitted study": g,
                "Model": m,
                "Refitted": False,
                "Remaining observations": len(aa),
                "Remaining studies": pair["studies"],
                "Difference from median (pp)": pair["delta_study_MAE"],
                "Interval lower (pp)": pair["conditional_CI"][0],
                "Interval upper (pp)": pair["conditional_CI"][1],
            }
        )
write("study_errors.csv", study)
write("fixed_prediction_deletions.csv", fixed)
allp = []
summaries = []
folds = []
for g in top:
    reduced = [r for r in rows if r["study_group"] != g]
    out = []
    for fold, tr, te in v.splits(reduced, "category_study_purged", cfg):
        train = [reduced[i] for i in tr]
        test = [reduced[i] for i in te]
        assert not (
            set(r["study_group"] for r in train) & set(r["study_group"] for r in test)
        )
        assert not (
            set(r["matrix_category"] for r in train)
            & set(r["matrix_category"] for r in test)
        )
        for m in plan["models"]:
            fitted, info = v.fit_tuned(train, m, cfg)
            p = fitted.predict(v.matrix(test, cfg))
            folds.append(
                {
                    "omitted_study": g,
                    "fold": fold,
                    "model": m,
                    "train_row_ids": [r["row_id"] for r in train],
                    "test_row_ids": [r["row_id"] for r in test],
                    "tuning": info,
                }
            )
            out.extend(
                [
                    {
                        **{
                            k: r[k]
                            for k in [
                                "row_id",
                                "target",
                                "study_group",
                                "matrix_category",
                            ]
                        },
                        "prediction": float(z),
                        "model": m,
                        "fold": fold,
                        "omitted_study": g,
                        "scheme": "category_study_purged",
                    }
                    for r, z in zip(test, p)
                ]
            )
        save("folds.json", folds)
        print(g, fold, "complete", flush=True)
    for m in plan["models"]:
        a = [r for r in out if r["model"] == m]
        baseline = [r for r in out if r["model"] == "training_median"]
        assert len(a) == len(reduced) and len(set(r["row_id"] for r in a)) == len(
            reduced
        )
        met = v.metrics(
            [r["target"] for r in a],
            [r["prediction"] for r in a],
            [r["study_group"] for r in a],
        )
        pair = v.paired_delta(a, baseline, cfg["bootstrap_replicates"], cfg["seed"])
        summaries.append(
            {
                "Omitted study": g,
                "Removed observations": counts[g],
                "Model": m,
                "Remaining observations": len(a),
                "Remaining studies": met["studies"],
                "Study MAE (pp)": met["study_MAE"],
                "Row MAE (pp)": met["row_MAE"],
                "Prediction R2": met["R2"],
                "Difference from median (pp)": pair["delta_study_MAE"],
                "Interval lower (pp)": pair["conditional_CI"][0],
                "Interval upper (pp)": pair["conditional_CI"][1],
            }
        )
    allp += out
    write("refit_deletions.csv", summaries)
    write("refit_predictions.csv", allp)
save(
    "COMPLETION.json",
    {
        "status": "complete",
        "deletions": len(top),
        "fold_model_fits": len(folds),
        "prediction_rows": len(allp),
        "plan": plan,
    },
)
print(json.dumps(summaries, indent=2))
