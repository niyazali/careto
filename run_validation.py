"""Run nested validation with row, study and food-category holdouts.

Requires Python 3.12 or newer. Preprocessing fits within each training split.
Save results after each outer fold and check input hashes before fitting.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn import __version__ as sklearn_version
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    # Atomic checkpoint avoids truncated JSON on interruption.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def study_mae(y, pred, groups):
    errors = defaultdict(list)
    for a, b, g in zip(y, pred, groups):
        errors[str(g)].append(abs(float(a) - float(b)))
    return float(np.mean([np.mean(e) for e in errors.values()]))


def metrics(y, pred, groups):
    y, pred, groups = np.asarray(y, float), np.asarray(pred, float), np.asarray(groups)
    counts = Counter(groups)
    weights = np.array([1 / counts[g] for g in groups], float)
    weights /= weights.sum()
    err = y - pred
    sst = float(np.sum((y - np.mean(y)) ** 2))
    vp = float(np.sum(weights * (pred - np.sum(weights * pred)) ** 2))
    slope = (
        float(
            np.sum(
                weights * (pred - np.sum(weights * pred)) * (y - np.sum(weights * y))
            )
            / vp
        )
        if vp > 1e-12
        else None
    )
    return {
        "rows": len(y),
        "studies": len(counts),
        "study_MAE": study_mae(y, pred, groups),
        "row_MAE": float(np.mean(abs(err))),
        "row_RMSE": float(np.sqrt(np.mean(err**2))),
        "R2": float(1 - np.sum(err**2) / sst) if len(y) > 1 and sst > 0 else None,
        "calibration_slope": slope,
        "calibration_joint_intercept": (
            float(np.sum(weights * y) - slope * np.sum(weights * pred))
            if slope is not None
            else None
        ),
        "calibration_in_the_large": float(np.sum(weights * err)),
    }


def splits(rows, scheme, config):
    n = len(rows)
    groups = np.array([r["study_group"] for r in rows])
    cats = np.array([r["matrix_category"] for r in rows])
    indexes = np.arange(n)
    if scheme == "random_row":
        if n < 2:
            return []
        return [
            (f"row_{i}", tr, te)
            for i, (tr, te) in enumerate(
                KFold(
                    min(config["outer_folds"], n),
                    shuffle=True,
                    random_state=config["seed"],
                ).split(indexes)
            )
        ]
    if scheme == "original_study":
        ng = len(set(groups))
        if ng < 2:
            return []
        return [
            (f"study_{i}", tr, te)
            for i, (tr, te) in enumerate(
                GroupKFold(min(config["outer_folds"], ng)).split(indexes, groups=groups)
            )
        ]
    result = []
    for category in sorted(set(cats) - {"unknown"}):
        te = indexes[cats == category]
        tr = indexes[cats != category]
        if scheme == "category_study_purged":
            tr = tr[~np.isin(groups[tr], groups[te])]
        result.append((category, tr, te))
    return result


def inner_splits(rows, config):
    # Use category holdouts where possible, with at most three inner folds.
    candidates = [
        (name, tr, te)
        for name, tr, te in splits(rows, "category_study_purged", config)
        if len(tr) >= 2 and len(te) >= 1
    ]
    if len(candidates) >= 2:
        rng = np.random.default_rng(config["seed"])
        picked = sorted(
            rng.choice(
                len(candidates),
                min(config["max_inner_folds"], len(candidates)),
                replace=False,
            )
        )
        return "study_purged_category", [candidates[i] for i in picked]
    groups = np.array([r["study_group"] for r in rows])
    ng = len(set(groups))
    if ng >= 2:
        k = min(config["max_inner_folds"], ng)
        candidates = [
            (f"group_{i}", tr, te)
            for i, (tr, te) in enumerate(
                GroupKFold(k).split(np.arange(len(rows)), groups=groups)
            )
            if len(tr) >= 2
        ]
        if len(candidates) >= 2:
            return "original_study_fallback", candidates
    return "infeasible", []


def estimator(name, params, config):
    seed, nt = config["seed"], config["fixed_model_settings"]["tree_estimators"]
    if name == "training_mean":
        return DummyRegressor(strategy="mean")
    if name == "training_median":
        return DummyRegressor(strategy="median")
    if name == "Ridge":
        return Ridge(**params)
    if name == "ElasticNet":
        return ElasticNet(
            **params,
            max_iter=config["fixed_model_settings"]["elasticnet_max_iter"],
            random_state=seed,
        )
    if name == "RandomForest":
        return RandomForestRegressor(
            **params, n_estimators=nt, n_jobs=1, random_state=seed
        )
    if name == "GradientBoosting":
        return GradientBoostingRegressor(**params, n_estimators=nt, random_state=seed)
    if name == "SVR":
        return SVR(**params)
    if name == "KNN":
        return KNeighborsRegressor(**params)
    if name == "XGBoost":
        return XGBRegressor(
            **params,
            n_estimators=nt,
            objective="reg:squarederror",
            n_jobs=1,
            random_state=seed,
        )
    raise ValueError(name)


def pipeline(name, params, config):
    steps = [("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    if name in {"Ridge", "ElasticNet", "SVR", "KNN"}:
        steps.append(("scaling", StandardScaler()))
    steps.append(("model", estimator(name, params, config)))
    return Pipeline(steps)


def matrix(rows, config):
    return np.array(
        [[str(r.get(k, "unknown")) for k in config["features"]] for r in rows],
        dtype=object,
    )


def fit_tuned(train, name, config):
    x = matrix(train, config)
    y = np.array([r["target"] for r in train], float)
    if name.startswith("training_"):
        counts = Counter(r["study_group"] for r in train)
        weights = np.array([1 / counts[r["study_group"]] for r in train], float)
        fitted = pipeline(name, {}, config).fit(x, y, model__sample_weight=weights)
        return fitted, {
            "parameters": {},
            "inner_objective": "no_tuning_baseline",
            "candidates": [],
            "training_weights": "Equal total mass per original study; row weight1/n_rows_in_study",
        }
    objective, inner = inner_splits(train, config)
    if not inner:
        raise ValueError(
            "No two feasible category-purged or original-study inner splits"
        )
    trials = []
    for index, params in enumerate(config["hyperparameters"][name]):
        ys, preds, groups, failures = [], [], [], []
        for fold, tr, te in inner:
            try:
                model = pipeline(name, params, config)
                model.fit(x[tr], y[tr])
                pred = model.predict(x[te])
                if not np.all(np.isfinite(pred)):
                    raise ValueError("Nonfinite prediction")
                ys.extend(y[te])
                preds.extend(pred)
                groups.extend(train[i]["study_group"] for i in te)
            except Exception as e:
                failures.append(
                    {"inner_fold": fold, "error": f"{type(e).__name__}: {e}"}
                )
        score = study_mae(ys, preds, groups) if not failures and len(ys) else None
        trials.append(
            {
                "candidate": index,
                "parameters": params,
                "study_MAE": score,
                "failures": failures,
            }
        )
    valid = [t for t in trials if t["study_MAE"] is not None]
    if not valid:
        raise ValueError("Every prespecified candidate failed: " + json.dumps(trials))
    best = min(valid, key=lambda t: (t["study_MAE"], t["candidate"]))
    fitted = pipeline(name, best["parameters"], config).fit(x, y)
    return fitted, {
        "parameters": best["parameters"],
        "inner_objective": objective,
        "candidates": trials,
        "inner_splits": [
            {
                "fold": f,
                "train_row_ids": [train[i]["row_id"] for i in tr],
                "test_row_ids": [train[i]["row_id"] for i in te],
                "study_overlap": sorted(
                    set(train[i]["study_group"] for i in tr)
                    & set(train[i]["study_group"] for i in te)
                ),
            }
            for f, tr, te in inner
        ],
    }


def bootstrap_study_mae(predictions, reps, seed):
    errors = defaultdict(list)
    for r in predictions:
        errors[r["study_group"]].append(abs(r["target"] - r["prediction"]))
    vals = np.array([np.mean(errors[g]) for g in sorted(errors)])
    if len(vals) < 2:
        return None
    rng = np.random.default_rng(seed)
    boot = vals[rng.integers(0, len(vals), (reps, len(vals)))].mean(axis=1)
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def paired_delta(a, b, reps, seed):
    a, b = {r["row_id"]: r for r in a}, {r["row_id"]: r for r in b}
    shared = sorted(a.keys() & b.keys())
    by_study = defaultdict(list)
    for rid in shared:
        ar, br = a[rid], b[rid]
        assert ar["target"] == br["target"] and ar["study_group"] == br["study_group"]
        by_study[ar["study_group"]].append(
            abs(ar["prediction"] - ar["target"]) - abs(br["prediction"] - br["target"])
        )
    if not by_study:
        return {
            "rows": 0,
            "studies": 0,
            "delta_study_MAE": None,
            "conditional_CI": None,
        }
    values = np.array([np.mean(by_study[g]) for g in sorted(by_study)])
    if len(values) < 2:
        return {
            "rows": len(shared),
            "studies": len(values),
            "delta_study_MAE": float(np.mean(values)),
            "conditional_CI": None,
            "reason": "Only one original study; no cluster interval",
        }
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (reps, len(values)))].mean(axis=1)
    return {
        "rows": len(shared),
        "studies": len(values),
        "delta_study_MAE": float(np.mean(values)),
        "conditional_CI": [float(v) for v in np.quantile(draws, [0.025, 0.975])],
        "direction": "A minus B; positive means A has larger error",
    }


def summarize(result, config):
    grouped = defaultdict(list)
    for r in result["predictions"]:
        grouped[(r["scheme"], r["model"])].append(r)
    summary, pairs = [], []
    for (scheme, name), rs in grouped.items():
        item = {
            "scheme": scheme,
            "model": name,
            **metrics(
                [r["target"] for r in rs],
                [r["prediction"] for r in rs],
                [r["study_group"] for r in rs],
            ),
            "study_MAE_conditional_CI": bootstrap_study_mae(
                rs, config["bootstrap_replicates"], config["seed"]
            ),
        }
        categories = defaultdict(list)
        for r in rs:
            categories[r["matrix_category"]].append(r)
        item["by_category"] = {
            cat: metrics(
                [r["target"] for r in a],
                [r["prediction"] for r in a],
                [r["study_group"] for r in a],
            )
            for cat, a in categories.items()
        }
        item["category_macro_study_MAE"] = float(
            np.mean([m["study_MAE"] for m in item["by_category"].values()])
        )
        summary.append(item)
        baseline = grouped.get((scheme, "training_median"), [])
        if name != "training_median" and baseline:
            pairs.append(
                {
                    "A": f"{scheme}/{name}",
                    "B": f"{scheme}/training_median",
                    **paired_delta(
                        rs, baseline, config["bootstrap_replicates"], config["seed"]
                    ),
                }
            )
    for name in sorted({k[1] for k in grouped}):
        for a, b in [
            ("category_study_purged", "category_unpurged"),
            ("original_study", "random_row"),
            ("category_study_purged", "original_study"),
        ]:
            if (a, name) in grouped and (b, name) in grouped:
                pairs.append(
                    {
                        "A": f"{a}/{name}",
                        "B": f"{b}/{name}",
                        **paired_delta(
                            grouped[(a, name)],
                            grouped[(b, name)],
                            config["bootstrap_replicates"],
                            config["seed"],
                        ),
                    }
                )
    result["summary"], result["paired_comparisons"] = summary, pairs
    return result


def diagnostics(train, test, fitted, config):
    # Raw feature permutation keeps all its one-hot columns together through pipeline.transform.
    x = matrix(test, config)
    y = [r["target"] for r in test]
    groups = [r["study_group"] for r in test]
    original = study_mae(y, fitted.predict(x), groups)
    rng = np.random.default_rng(config["seed"])
    importance, shift = {}, {}
    for name, members in config["permutation_groups"].items():
        columns = [config["features"].index(member) for member in members]
        changes = []
        for _ in range(5):
            changed = x.copy()
            indexes = rng.permutation(len(test))
            changed[:, columns] = x[indexes][:, columns]
            changes.append(study_mae(y, fitted.predict(changed), groups) - original)
        importance[name] = {
            "delta_study_MAE_mean": float(np.mean(changes)),
            "repeats": changes,
            "raw_features_permuted_jointly": members,
            "interpretation": "Held-out predictive reliance; not causal importance",
        }
    for col, name in enumerate(config["features"]):
        ca, cb = Counter(r[name] for r in train), Counter(r[name] for r in test)
        levels = sorted(ca.keys() | cb.keys())
        p = [ca[k] / len(train) for k in levels]
        q = [cb[k] / len(test) for k in levels]
        shift[name] = {
            "categorical_JS_distance_base2": float(jensenshannon(p, q, base=2)),
            "unseen_test_levels": sorted(cb.keys() - ca.keys()),
        }
    return {"heldout_permutation": importance, "covariate_support": shift}


def run(cohort_path, output, config_path, smoke=False, sensitivity="core"):
    config = json.loads(config_path.read_text())
    cohort = json.loads(cohort_path.read_text())
    if not smoke:
        freeze_path = BASE / (
            "FREEZE_with_conversions.json"
            if cohort_path.name == "COHORT_with_conversions.json"
            else "FREEZE.json"
        )
        assert (
            freeze_path.exists()
        ), "Source-method review/cohort must be frozen before scientific fit"
        freeze = json.loads(freeze_path.read_text())
        assert freeze["cohort_sha256"] == hash_file(cohort_path)
        assert freeze["config_sha256"] == hash_file(config_path)
        assert not cohort[
            "duplicate_candidates"
        ], "Adjudicate duplicate candidates before fitting"
    rows = [
        r
        for r in cohort["rows"]
        if r["cohort"] == "core" or sensitivity == "core_plus_supernatant"
    ]
    if sensitivity == "numeric_text_table_only":
        rows = [r for r in rows if "graph" not in r["value_type"]]
    assert rows and len({r["row_id"] for r in rows}) == len(rows)
    output.mkdir(exist_ok=True, parents=True)
    result = {
        "status": "running",
        "cohort": sensitivity,
        "smoke_synthetic_only": smoke,
        "config": config,
        "config_sha256": hash_file(config_path),
        "cohort_sha256": hash_file(cohort_path),
        "runner_sha256": hash_file(__file__),
        "python": sys.version,
        "sklearn": sklearn_version,
        "rows": len(rows),
        "studies": len(set(r["study_group"] for r in rows)),
        "predictions": [],
        "folds": [],
        "failures": [],
        "warnings": [],
    }
    if (output / "CHECKPOINT.json").exists():
        previous = json.loads((output / "CHECKPOINT.json").read_text())
        assert (
            previous["cohort_sha256"] == result["cohort_sha256"]
            and previous["config_sha256"] == result["config_sha256"]
        ), "Changed inputs; use a new results directory"
        assert (
            previous.get("runner_sha256") == result["runner_sha256"]
        ), "Runner changed; use a new results directory"
        assert (
            previous["cohort"] == sensitivity
            and previous["smoke_synthetic_only"] == smoke
        )
        result = previous
    models = ["training_mean", "training_median"] + list(config["hyperparameters"])
    started = time.time()
    for scheme in config["outer_schemes"]:
        outer = splits(rows, scheme, config)
        if not outer:
            result["failures"].append(
                {"scheme": scheme, "reason": "No feasible outer split"}
            )
        for fold_name, tr, te in outer:
            if any(
                f["scheme"] == scheme and f["fold"] == fold_name
                for f in result["folds"]
            ):
                continue
            train, test = [rows[i] for i in tr], [rows[i] for i in te]
            overlap = set(r["study_group"] for r in train) & set(
                r["study_group"] for r in test
            )
            actually_removed = []
            if scheme == "category_study_purged":
                test_ids = {r["row_id"] for r in test}
                test_groups = {r["study_group"] for r in test}
                actually_removed = [
                    r
                    for r in rows
                    if r["row_id"] not in test_ids and r["study_group"] in test_groups
                ]
            if scheme in {"original_study", "category_study_purged"}:
                assert not overlap, f"Original study leakage {scheme}/{fold_name}"
            receipt = {
                "scheme": scheme,
                "fold": fold_name,
                "train_row_ids": [r["row_id"] for r in train],
                "test_row_ids": [r["row_id"] for r in test],
                "train_studies": sorted(set(r["study_group"] for r in train)),
                "test_studies": sorted(set(r["study_group"] for r in test)),
                "study_overlap": sorted(overlap),
                "purged_studies": sorted({r["study_group"] for r in actually_removed}),
                "purged_row_ids": [r["row_id"] for r in actually_removed],
                "models": {},
            }
            if len(train) < 2:
                result["failures"].append(
                    {
                        "scheme": scheme,
                        "fold": fold_name,
                        "reason": f"Fewer than two training rows after split/purge ({len(train)})",
                    }
                )
            else:
                for name in models:
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            fitted, tuning = fit_tuned(train, name, config)
                            pred = fitted.predict(matrix(test, config))
                        if not np.all(np.isfinite(pred)):
                            raise ValueError("Nonfinite outer prediction")
                        result["warnings"].extend(
                            {
                                "scheme": scheme,
                                "fold": fold_name,
                                "model": name,
                                "message": str(w.message),
                            }
                            for w in caught
                        )
                        receipt["models"][name] = tuning
                        if name not in {"training_mean", "training_median"}:
                            receipt["models"][name].update(
                                diagnostics(train, test, fitted, config)
                            )
                        for r, prediction in zip(test, pred):
                            result["predictions"].append(
                                {
                                    "row_id": r["row_id"],
                                    "target": r["target"],
                                    "prediction": float(prediction),
                                    "study_group": r["study_group"],
                                    "matrix_category": r["matrix_category"],
                                    "scheme": scheme,
                                    "fold": fold_name,
                                    "model": name,
                                }
                            )
                        with (output / f"{scheme}__{fold_name}__{name}.pkl").open(
                            "wb"
                        ) as f:
                            pickle.dump(fitted, f)
                    except Exception as e:
                        result["failures"].append(
                            {
                                "scheme": scheme,
                                "fold": fold_name,
                                "model": name,
                                "reason": f"{type(e).__name__}: {e}",
                            }
                        )
            result["folds"].append(receipt)
            result["elapsed_seconds"] = time.time() - started
            save_json(output / "CHECKPOINT.json", result)
            print(
                f'{scheme}/{fold_name}: train {len(train)} rows, test {len(test)} rows; {len(result["predictions"])} predictions saved',
                flush=True,
            )
    result["status"] = "complete_with_failures" if result["failures"] else "complete"
    summarize(result, config)
    save_json(output / "RESULTS.json", result)
    with (output / "oof_predictions.intermediate.csv").open("w", newline="") as f:
        if result["predictions"]:
            w = csv.DictWriter(f, fieldnames=list(result["predictions"][0]))
            w.writeheader()
            w.writerows(result["predictions"])
    save_json(
        output / "RESULT_SUMMARY.json",
        {
            "summary": result["summary"],
            "paired_comparisons": result["paired_comparisons"],
            "failures": result["failures"],
        },
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=BASE / "COHORT.json")
    parser.add_argument("--output", type=Path, default=BASE / "results_core")
    parser.add_argument("--config", type=Path, default=BASE / "analysis_config.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--sensitivity",
        choices=["core", "numeric_text_table_only", "core_plus_supernatant"],
        default="core",
    )
    args = parser.parse_args()
    if args.smoke and args.cohort == BASE / "COHORT.json":
        raise ValueError("Smoke test must use explicitly separate synthetic input")
    run(args.cohort, args.output, args.config, args.smoke, args.sensitivity)
