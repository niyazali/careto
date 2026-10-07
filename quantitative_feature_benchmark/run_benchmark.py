"""Compare categorical and quantitative predictors using nested study holdouts."""

import argparse, hashlib, json, platform, os, io
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

CATS = ["analyte", "chemical_class", "food_form", "processing", "added_fat", "protocol"]
SEED = 20261005
RF = [
    dict(max_depth=3, min_samples_leaf=2),
    dict(max_depth=6, min_samples_leaf=2),
    dict(max_depth=None, min_samples_leaf=4),
]
SV = [
    dict(C=1.0, gamma="scale", epsilon=0.1),
    dict(C=10.0, gamma="scale", epsilon=0.5),
    dict(C=100.0, gamma="scale", epsilon=1.0),
]


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def balanced_mae(y, p, g):
    return float(pd.DataFrame({"e": abs(y - p), "g": g}).groupby("g").e.mean().mean())


def median(y, g):
    _, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    weights = 1 / counts[inv]
    baseline = DummyRegressor(strategy="median").fit(
        np.zeros((len(y), 1)), y, sample_weight=weights
    )
    return float(baseline.predict(np.zeros((1, 1)))[0])


def build(family, params, features):
    cat = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    branches = [("categorical", cat, CATS)]
    if features:
        num = Pipeline(
            [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        )
        branches.append(("quantitative", num, features))
    steps = [("preprocess", ColumnTransformer(branches))]
    if family == "SVR":
        steps.append(("scale", StandardScaler()))
    model = (
        RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=SEED, **params)
        if family == "RandomForest"
        else SVR(**params)
    )
    return Pipeline(steps + [("model", model)])


def run(input_path, features, output):
    input_path = Path(input_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (
        not features
        or len(set(features)) != len(features)
        or set(features) & set(CATS + ["target", "study_group", "row_id"])
    ):
        raise ValueError(
            "Specify distinct quantitative features, excluding target/identifiers/categorical predictors"
        )
    raw = input_path.read_bytes()
    df = pd.read_csv(io.BytesIO(raw))
    required = ["row_id", "target", "study_group"] + CATS + features
    missing = set(required) - set(df)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if (
        df.row_id.isna().any()
        or df.row_id.duplicated().any()
        or df.study_group.isna().any()
    ):
        raise ValueError("Unique nonmissing row IDs and original study groups required")
    if len(df.study_group.unique()) < 5:
        raise ValueError(
            "At least five original-study groups required (feasibility rule, not power adequacy)"
        )
    y = pd.to_numeric(df.target, errors="raise").to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError("Finite target required")
    for f in features:
        df[f] = pd.to_numeric(df[f], errors="raise")
        if np.isinf(df[f]).any():
            raise ValueError(f"Infinite feature: {f}")
        if df[f].notna().sum() == 0:
            raise ValueError(f"Entirely missing feature: {f}")
    for f in CATS:
        df[f] = df[f].fillna("unknown").astype(str)
    g = df.study_group.astype(str).to_numpy()
    config = {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "features": features,
        "categorical_features": CATS,
        "seed": SEED,
        "outer": "leave-one-original-study-out",
        "inner": "GroupKFold min(3, number of outer-training studies)",
        "metric": "Equal original-study MAE",
        "bootstrap_replicates": 2000,
        "bootstrap_scope": "Paired study resampling of fixed cross-fitted losses; conditional intervals exclude model refitting uncertainty",
        "feasibility_min_groups": 5,
        "feasibility_not_power_adequacy": True,
        "feature_selection": "None; features fixed before this benchmark fit; post hoc to original analysis",
        "prediction_policy": "unbounded",
        "baseline_implementation": "sklearn DummyRegressor(strategy=median), sample_weight=1/n_rows_in_training_study, matching original analysis",
        "hyperparameters": {"RandomForest": RF, "SVR": SV},
        "fixed_RF": {"n_estimators": 100, "n_jobs": 1},
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rows": len(df),
        "groups": len(set(g)),
    }
    dump(output / "config.json", config)  # frozen before first fit
    pred = df[["row_id", "study_group", "target"]].copy()
    pred["input_row_index"] = np.arange(len(df))
    names = ["weighted_median"] + [
        f"{fam}_{kind}"
        for fam in ["RandomForest", "SVR"]
        for kind in ["categorical", "quantitative"]
    ]
    for n in names:
        pred[n] = np.nan
    logs = []
    seen = np.zeros(len(df), int)
    for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(df, y, g), 1):
        assert not set(g[tr]) & set(g[te])
        seen[te] += 1
        inner = list(
            GroupKFold(n_splits=min(3, len(set(g[tr])))).split(
                df.iloc[tr], y[tr], g[tr]
            )
        )
        for a, b in inner:
            assert not set(g[tr[a]]) & set(g[tr[b]]) and not set(g[tr[a]]) & set(g[te])
        rec = {
            "fold": fold,
            "train_studies": sorted(set(g[tr])),
            "test_studies": sorted(set(g[te])),
            "train_row_ids": df.row_id.iloc[tr].tolist(),
            "test_row_ids": df.row_id.iloc[te].tolist(),
            "inner_splits": [
                {
                    "train_studies": sorted(set(g[tr[a]])),
                    "validation_studies": sorted(set(g[tr[b]])),
                }
                for a, b in inner
            ],
            "models": {},
        }
        pred.loc[te, "weighted_median"] = median(y[tr], g[tr])
        for fam, grid in [("RandomForest", RF), ("SVR", SV)]:
            for kind, fs in [("categorical", []), ("quantitative", features)]:
                scores = []
                for params in grid:
                    # Pool inner held-out losses then average equally over studies.
                    iy = []
                    ip = []
                    ig = []
                    for a, b in inner:
                        model = build(fam, params, fs)
                        model.fit(df.iloc[tr[a]], y[tr[a]])
                        iy.extend(y[tr[b]])
                        ip.extend(model.predict(df.iloc[tr[b]]))
                        ig.extend(g[tr[b]])
                    scores.append(
                        balanced_mae(np.array(iy), np.array(ip), np.array(ig))
                    )
                best = int(np.argmin(scores))
                model = build(fam, grid[best], fs)
                model.fit(df.iloc[tr], y[tr])
                name = f"{fam}_{kind}"
                pred.loc[te, name] = model.predict(df.iloc[te])
                rec["models"][name] = {
                    "candidate_study_balanced_mae": scores,
                    "selected_index": best,
                    "selected_settings": grid[best],
                }
        logs.append(rec)
        dump(output / "fold_logs.json", logs)
        print(f"Completed outer fold {fold}/{len(set(g))}: {g[te][0]}", flush=True)
    assert (
        (seen == 1).all()
        and pred.row_id.tolist() == df.row_id.tolist()
        and np.isfinite(pred[names].to_numpy()).all()
    )
    pred.to_csv(output / "heldout_predictions.csv", index=False)
    errors = pd.DataFrame({"study_group": g})
    for n in names:
        errors[n] = abs(y - pred[n].to_numpy())
    errors = errors.groupby("study_group", sort=True).mean()
    errors["n_rows"] = pd.Series(g).value_counts()
    errors.to_csv(output / "study_errors.csv")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(errors), size=(2000, len(errors)))
    metrics = {
        "rows": len(df),
        "studies": len(errors),
        "interval_scope": config["bootstrap_scope"],
        "models": {},
        "paired_differences": {},
    }
    for n in names:
        a = errors[n].to_numpy()
        err = y - pred[n].to_numpy()
        denom = float(((y - y.mean()) ** 2).sum())
        metrics["models"][n] = {
            "study_balanced_MAE": float(a.mean()),
            "conditional_95pct_interval": np.quantile(
                a[idx].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "descriptive_row_MAE": float(abs(err).mean()),
            "descriptive_row_RMSE": float(np.sqrt((err**2).mean())),
            "descriptive_pooled_R2": (
                float(1 - (err**2).sum() / denom) if denom else None
            ),
        }
    for fam in ["RandomForest", "SVR"]:
        a = (errors[f"{fam}_quantitative"] - errors[f"{fam}_categorical"]).to_numpy()
        metrics["paired_differences"][fam] = {
            "direction": "quantitative minus categorical; negative favors quantitative",
            "delta_MAE": float(a.mean()),
            "conditional_95pct_interval": np.quantile(
                a[idx].mean(axis=1), [0.025, 0.975]
            ).tolist(),
        }
    for n in names[1:]:
        a = (errors[n] - errors["weighted_median"]).to_numpy()
        metrics["paired_differences"][f"{n}_minus_weighted_median"] = {
            "direction": "learner minus weighted median; negative favors learner",
            "delta_MAE": float(a.mean()),
            "conditional_95pct_interval": np.quantile(
                a[idx].mean(axis=1), [0.025, 0.975]
            ).tolist(),
        }
    dump(output / "metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.input, [f.strip() for f in a.features.split(",") if f.strip()], a.output)
