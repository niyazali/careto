"""Check study-weighted fitting across alternative inner partitions.
Original data, outer folds, candidate grid and model randomness are unchanged.
"""

from pathlib import Path
from collections import Counter
import json, csv, time
import numpy as np
import run_validation as v

D = Path(__file__).resolve().parent
O = D / "training_weight_results"
O.mkdir(exist_ok=True)
cfg = json.loads((D / "analysis_config.json").read_text())
rows = [
    r
    for r in json.loads((D / "COHORT.json").read_text())["rows"]
    if r["cohort"] == "core"
]
seeds = [20261005, 20261006, 20261007, 20261008, 20261009]
assert (
    cfg["seed"] in seeds
), "Prespecified sensitivity seed set must include the primary seed"
n_rows = len(rows)
row_ids = {r["row_id"] for r in rows}
assert len(row_ids) == n_rows
plan = {
    "scope": "Post hoc sensitivity; no changes to original analysis",
    "rows": len(rows),
    "studies": len(set(r["study_group"] for r in rows)),
    "models": ["RandomForest", "SVR"],
    "inner_seeds": seeds,
    "model_seed": cfg["seed"],
    "outer_scheme": "category_study_purged",
    "weight_formula": "N/(G*n_g), recomputed within each training partition",
    "preprocessing": "Training-fitted encoding and scaling unchanged and unweighted",
    "RF_behavior": "scikit-learn 1.9.1 uses weighted bootstrap draw probabilities; tree weights are draw counts",
    "SVR_behavior": "sample weights multiply per-observation C",
    "intervals": "2000 paired fixed-prediction study resamples; conditional and unadjusted",
    "inputs": {
        n: v.hash_file(D / n)
        for n in ["COHORT.json", "analysis_config.json", "run_validation.py"]
    },
}
if (O / "PLAN.json").exists():
    assert json.loads((O / "PLAN.json").read_text()) == plan
else:
    v.save_json(O / "PLAN.json", plan)


def weights(rr):
    c = Counter(r["study_group"] for r in rr)
    w = np.array([len(rr) / (len(c) * c[r["study_group"]]) for r in rr])
    assert np.isclose(w.mean(), 1)
    masses = [sum(w[i] for i, r in enumerate(rr) if r["study_group"] == g) for g in c]
    assert np.allclose(masses, len(rr) / len(c))
    return w


def fit(train, name, weighted, seed):
    x = v.matrix(train, cfg)
    y = np.array([r["target"] for r in train])
    obj, inner = v.inner_splits(train, {**cfg, "seed": seed})
    assert len(inner) >= 2
    trials = []
    for index, params in enumerate(cfg["hyperparameters"][name]):
        yy = []
        pp = []
        gg = []
        for fold, tr, te in inner:
            assert not (
                set(train[i]["study_group"] for i in tr)
                & set(train[i]["study_group"] for i in te)
            )
            rr = [train[i] for i in tr]
            m = v.pipeline(name, params, cfg)
            kw = {"model__sample_weight": weights(rr)} if weighted else {}
            m.fit(x[tr], y[tr], **kw)
            p = m.predict(x[te])
            assert np.all(np.isfinite(p))
            yy.extend(y[te])
            pp.extend(p)
            gg.extend(train[i]["study_group"] for i in te)
        trials.append(
            {
                "candidate": index,
                "parameters": params,
                "study_MAE": v.study_mae(yy, pp, gg),
            }
        )
    best = min(trials, key=lambda z: (z["study_MAE"], z["candidate"]))
    m = v.pipeline(name, best["parameters"], cfg)
    m.fit(x, y, **({"model__sample_weight": weights(train)} if weighted else {}))
    return m, {
        "parameters": best["parameters"],
        "trials": trials,
        "inner_objective": obj,
        "inner_splits": [
            {
                "fold": f,
                "train_ids": [train[i]["row_id"] for i in tr],
                "test_ids": [train[i]["row_id"] for i in te],
            }
            for f, tr, te in inner
        ],
    }


folds = v.splits(rows, "category_study_purged", cfg)
assert len(folds) == len({r["matrix_category"] for r in rows})
expected_fits = len(seeds) * 2 * len(plan["models"]) * len(folds)
base = []
predictions = []
details = []
for fold, tr, te in folds:
    train = [rows[i] for i in tr]
    test = [rows[i] for i in te]
    assert not (
        set(r["study_group"] for r in train) & set(r["study_group"] for r in test)
    )
    m, _ = v.fit_tuned(train, "training_median", cfg)
    for r, p in zip(test, m.predict(v.matrix(test, cfg))):
        base.append(
            {**r, "prediction": float(p), "fold": fold, "model": "training_median"}
        )
for seed in seeds:
    for weighted in [False, True]:
        for name in plan["models"]:
            for fold, tr, te in folds:
                train = [rows[i] for i in tr]
                test = [rows[i] for i in te]
                m, info = fit(train, name, weighted, seed)
                details.append(
                    {
                        "inner_seed": seed,
                        "weighted": weighted,
                        "model": name,
                        "fold": fold,
                        **info,
                    }
                )
                for r, p in zip(test, m.predict(v.matrix(test, cfg))):
                    predictions.append(
                        {
                            **r,
                            "prediction": float(p),
                            "fold": fold,
                            "model": name,
                            "weighted": weighted,
                            "inner_seed": seed,
                        }
                    )
                v.save_json(
                    O / "CHECKPOINT.json",
                    {
                        "completed_fits": len(details),
                        "expected_fits": expected_fits,
                        "latest": [seed, weighted, name, fold],
                    },
                )
                print(seed, weighted, name, fold, flush=True)
# Check that the original seed reproduces the saved results and baseline.
saved = list(csv.DictReader((D / "saved_results_core/heldout_predictions.csv").open()))
saved = {
    (r["model"], r["row_id"]): float(r["prediction"])
    for r in saved
    if r["scheme"] == "category_study_purged"
}
maxdiff = max(
    abs(r["prediction"] - saved[(r["model"], r["row_id"])])
    for r in predictions
    if not r["weighted"] and r["inner_seed"] == cfg["seed"]
)
bd = max(abs(r["prediction"] - saved[(r["model"], r["row_id"])]) for r in base)
assert maxdiff < 1e-10 and bd < 1e-10
summary = []
for seed in seeds:
    for weighted in [False, True]:
        for name in plan["models"]:
            p = [
                r
                for r in predictions
                if r["inner_seed"] == seed
                and r["weighted"] == weighted
                and r["model"] == name
            ]
            assert len(p) == n_rows and {r["row_id"] for r in p} == row_ids
            un = [
                r
                for r in predictions
                if r["inner_seed"] == seed and not r["weighted"] and r["model"] == name
            ]
            summary.append(
                {
                    "inner_seed": seed,
                    "weighted": weighted,
                    "model": name,
                    **v.metrics(
                        [r["target"] for r in p],
                        [r["prediction"] for r in p],
                        [r["study_group"] for r in p],
                    ),
                    "versus_median": v.paired_delta(
                        p, base, cfg["bootstrap_replicates"], cfg["seed"]
                    ),
                    "versus_unweighted": (
                        v.paired_delta(p, un, cfg["bootstrap_replicates"], cfg["seed"])
                        if weighted
                        else None
                    ),
                }
            )
v.save_json(
    O / "RESULTS.json",
    {
        "plan": plan,
        "baseline": v.metrics(
            [r["target"] for r in base],
            [r["prediction"] for r in base],
            [r["study_group"] for r in base],
        ),
        "summary": summary,
        "reproduction_max_absolute_error": maxdiff,
        "baseline_reproduction_max_absolute_error": bd,
        "completed_outer_fits": len(details),
        "completed_baseline_fits": len(folds),
    },
)
v.save_json(O / "FOLDS.json", details)
fields = [
    "row_id",
    "target",
    "prediction",
    "study_group",
    "matrix_category",
    "fold",
    "model",
    "weighted",
    "inner_seed",
]
with (O / "heldout_predictions.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(predictions)
print(json.dumps(summary, indent=2))
