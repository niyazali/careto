"""Compare category holdouts with matched training-set sizes.
Twenty subsets sampled without using outcomes match each primary
study-excluded fold's training size. Average losses over repetitions.
"""

from pathlib import Path
import json, csv
from collections import defaultdict
import numpy as np
import run_validation as v

D = Path(__file__).resolve().parent
O = D / "training_size_results"
O.mkdir(exist_ok=True)
cfg = json.loads((D / "analysis_config.json").read_text())
rows = [
    r
    for r in json.loads((D / "COHORT.json").read_text())["rows"]
    if r["cohort"] == "core"
]
models = ["RandomForest", "SVR", "training_median"]
seeds = list(range(20261010, 20261030))
n_rows = len(rows)
row_ids = {r["row_id"] for r in rows}
assert len(row_ids) == n_rows
plan = {
    "analysis": "Post hoc training-row-count-matched category holdout",
    "models": models,
    "sampling_seeds": seeds,
    "sampling": "Uniform rows without replacement from category-unpurged training pool; no target-based selection; preserve original input order",
    "reference": "Saved primary category/study-purged predictions",
    "test": "Exactly the same held-out category observations in both arms",
    "tuning": "Original nested, study-purged inner rule and candidates; fixed original seed",
    "aggregation": "Mean absolute loss per row across 20 repetitions, then equal original-study mean; not ensemble predictions",
    "interval": "2000 paired original-study resamples of repetition-averaged fixed losses; conditional, unadjusted; does not include refitting uncertainty",
    "limits": "Matches observation count only; study count, feature support and composition differ; not a pure leakage estimate",
    "inputs": {
        n: v.hash_file(D / n)
        for n in [
            "COHORT.json",
            "analysis_config.json",
            "saved_results_core/heldout_predictions.csv",
            "run_validation.py",
        ]
    },
}
if (O / "PLAN.json").exists():
    assert json.loads((O / "PLAN.json").read_text()) == plan
else:
    v.save_json(O / "PLAN.json", plan)
primary = {f: (tr, te) for f, tr, te in v.splits(rows, "category_study_purged", cfg)}
unpurged = {f: (tr, te) for f, tr, te in v.splits(rows, "category_unpurged", cfg)}
saved = list(csv.DictReader((D / "saved_results_core/heldout_predictions.csv").open()))
ref = {
    m: {
        r["row_id"]: r
        for r in saved
        if r["scheme"] == "category_study_purged" and r["model"] == m
    }
    for m in models
}
assert all(set(x) == row_ids for x in ref.values())
assert set(primary) == set(unpurged)
expected_fold_repetitions = len(seeds) * len(primary)
allpred = []
details = []
for rep, seed in enumerate(seeds):
    for fi, fold in enumerate(primary):
        ptr, te = primary[fold]
        pool, ute = unpurged[fold]
        assert np.array_equal(te, ute)
        rng = np.random.default_rng(np.random.SeedSequence([seed, fi]))
        idx = np.sort(rng.choice(pool, len(ptr), replace=False))
        assert len(idx) == len(ptr) and not (set(idx) & set(te))
        train = [rows[i] for i in idx]
        test = [rows[i] for i in te]
        tg = {r["study_group"] for r in test}
        pg = {rows[i]["study_group"] for i in ptr}
        assert not (pg & tg) and all(r["matrix_category"] != fold for r in train)
        record = {
            "repetition": rep,
            "sampling_seed": seed,
            "fold": fold,
            "training_rows": len(train),
            "primary_training_rows": len(ptr),
            "training_studies": len({r["study_group"] for r in train}),
            "primary_training_studies": len(pg),
            "shared_test_studies": sorted({r["study_group"] for r in train} & tg),
            "shared_test_study_rows": sum(r["study_group"] in tg for r in train),
            "train_ids": [r["row_id"] for r in train],
            "test_ids": [r["row_id"] for r in test],
            "models": {},
        }
        cache = O / f"fit_{rep:02d}_{fi}.json"
        if cache.exists():
            got = json.loads(cache.read_text())
            assert got["detail"]["train_ids"] == record["train_ids"]
            details.append(got["detail"])
            allpred.extend(got["predictions"])
            continue
        preds = []
        for name in models:
            fitted, info = v.fit_tuned(train, name, cfg)
            for inner in info.get("inner_splits", []):
                assert not inner["study_overlap"]
            record["models"][name] = info
            for r, p in zip(test, fitted.predict(v.matrix(test, cfg))):
                original = ref[name][r["row_id"]]
                assert float(original["target"]) == r["target"]
                assert (
                    original["study_group"] == r["study_group"]
                    and original["fold"] == fold
                )
                preds.append(
                    {
                        "row_id": r["row_id"],
                        "study_group": r["study_group"],
                        "target": r["target"],
                        "prediction": float(p),
                        "primary_prediction": float(original["prediction"]),
                        "model": name,
                        "fold": fold,
                        "repetition": rep,
                        "sampling_seed": seed,
                    }
                )
        v.save_json(cache, {"detail": record, "predictions": preds})
        details.append(record)
        allpred.extend(preds)
        v.save_json(
            O / "CHECKPOINT.json",
            {
                "completed_fold_repetitions": len(details),
                "expected": expected_fold_repetitions,
            },
        )
        print("completed", len(details), "/", expected_fold_repetitions, flush=True)
summary = []
repeatstats = []
lossrows = []
for name in models:
    for rep in range(len(seeds)):
        pp = [r for r in allpred if r["model"] == name and r["repetition"] == rep]
        assert len(pp) == n_rows and {r["row_id"] for r in pp} == row_ids
        baseline = {
            r["row_id"]: r
            for r in allpred
            if r["model"] == "training_median" and r["repetition"] == rep
        }
        rr = [{**r, "prediction": r["primary_prediction"]} for r in pp]
        repeatstats.append(
            {
                "model": name,
                "repetition": rep,
                "sampling_seed": seeds[rep],
                "study_MAE": v.study_mae(
                    [r["target"] for r in pp],
                    [r["prediction"] for r in pp],
                    [r["study_group"] for r in pp],
                ),
                "versus_primary": v.paired_delta(
                    pp, rr, cfg["bootstrap_replicates"], cfg["seed"]
                ),
                "versus_own_median": v.paired_delta(
                    pp,
                    list(baseline.values()),
                    cfg["bootstrap_replicates"],
                    cfg["seed"],
                ),
            }
        )
    for rid in ref[name]:
        pp = [r for r in allpred if r["model"] == name and r["row_id"] == rid]
        assert len(pp) == len(seeds)
        lossrows.append(
            {
                "model": name,
                "row_id": rid,
                "study_group": pp[0]["study_group"],
                "matched_loss": float(
                    np.mean([abs(r["prediction"] - r["target"]) for r in pp])
                ),
                "primary_loss": abs(pp[0]["primary_prediction"] - pp[0]["target"]),
            }
        )
matchedmedian = {
    r["row_id"]: r["matched_loss"] for r in lossrows if r["model"] == "training_median"
}


def contrast(rr, baseline):
    grouped = defaultdict(list)
    for r in rr:
        grouped[r["study_group"]].append(r["matched_loss"] - baseline(r))
    vals = np.array([np.mean(grouped[g]) for g in sorted(grouped)])
    rng = np.random.default_rng(cfg["seed"])
    boot = vals[
        rng.integers(0, len(vals), (cfg["bootstrap_replicates"], len(vals)))
    ].mean(axis=1)
    return {
        "delta_study_MAE": float(vals.mean()),
        "conditional_CI": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
    }


for name in models:
    rr = [r for r in lossrows if r["model"] == name]
    groups = sorted({r["study_group"] for r in rr})
    agg = lambda key: float(
        np.mean(
            [np.mean([r[key] for r in rr if r["study_group"] == g]) for g in groups]
        )
    )
    reps = [r["study_MAE"] for r in repeatstats if r["model"] == name]
    summary.append(
        {
            "model": name,
            "primary_study_MAE": agg("primary_loss"),
            "matched_study_MAE": agg("matched_loss"),
            "repetition_min": min(reps),
            "repetition_max": max(reps),
            "versus_primary": contrast(rr, lambda r: r["primary_loss"]),
            "versus_own_median": contrast(rr, lambda r: matchedmedian[r["row_id"]]),
        }
    )
for fn, data in [
    (
        "RESULTS.json",
        {
            "plan": plan,
            "summary": summary,
            "repetitions": repeatstats,
            "fold_diagnostics": [
                {
                    k: r[k]
                    for k in [
                        "repetition",
                        "sampling_seed",
                        "fold",
                        "training_rows",
                        "primary_training_rows",
                        "training_studies",
                        "primary_training_studies",
                        "shared_test_studies",
                        "shared_test_study_rows",
                    ]
                }
                for r in details
            ],
        },
    ),
    ("FOLDS.json", details),
]:
    v.save_json(O / fn, data)
for fn, data in [("heldout_predictions.csv", allpred), ("row_losses.csv", lossrows)]:
    with (O / fn).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
print(json.dumps(summary, indent=2), flush=True)
