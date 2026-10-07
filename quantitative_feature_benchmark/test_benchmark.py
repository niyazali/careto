"""Synthetic structural checks. Outputs here are test fixtures, not research results."""

import tempfile, json
from pathlib import Path
import numpy as np
import pandas as pd
from run_benchmark import run, CATS, build, median
from sklearn.dummy import DummyRegressor

with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    rows = []
    for group in range(5):
        for j in range(3 + group):
            rows.append(
                dict(
                    row_id=f"{group}-{j}",
                    study_group=f"S{group}",
                    target=float(10 + group * 2 + j),
                    q=float(j) if j else np.nan,
                    **{f: f"cat{group}" for f in CATS},
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(d / "synthetic.csv", index=False)
    m1 = run(d / "synthetic.csv", ["q"], d / "one")
    m2 = run(d / "synthetic.csv", ["q"], d / "two")
    assert len(m1["paired_differences"]) == 6
    for name, result in m1["models"].items():
        assert all(
            np.isfinite(result[k])
            for k in [
                "descriptive_row_MAE",
                "descriptive_row_RMSE",
                "descriptive_pooled_R2",
            ]
        )
    for name in [
        "config.json",
        "metrics.json",
        "heldout_predictions.csv",
        "study_errors.csv",
        "fold_logs.json",
    ]:
        assert (d / "one" / name).read_bytes() == (d / "two" / name).read_bytes(), name
    preds = pd.read_csv(d / "one/heldout_predictions.csv")
    assert preds.row_id.tolist() == frame.row_id.tolist()
    logs = json.loads((d / "one/fold_logs.json").read_text())
    for log in logs:
        assert set(log["train_studies"]).isdisjoint(log["test_studies"])
        for inner in log["inner_splits"]:
            assert set(inner["train_studies"]).isdisjoint(inner["validation_studies"])
            assert set(inner["validation_studies"]).isdisjoint(log["test_studies"])
    # Ensure held-out numeric extremes cannot change train-fitted imputation/scaling.
    model = build("SVR", dict(C=1, gamma="scale", epsilon=0.1), ["q"])
    model.fit(frame.iloc[:7], frame.target.iloc[:7])
    imputer = (
        model.named_steps["preprocess"]
        .named_transformers_["quantitative"]
        .named_steps["impute"]
    )
    assert imputer.statistics_[0] == np.nanmedian(frame.q.iloc[:7])
    for yy, gg in [
        (np.array([1.0, 100.0, 100.0, 100.0]), np.array(["a", "b", "b", "b"])),
        (
            np.array([1.0, 2.0, 2.0, 3.0, 10.0, 20.0, 30.0]),
            np.array(["a", "a", "b", "b", "b", "c", "c"]),
        ),
    ]:
        _, inv, counts = np.unique(gg, return_inverse=True, return_counts=True)
        reference = (
            DummyRegressor(strategy="median")
            .fit(np.zeros((len(yy), 1)), yy, sample_weight=1 / counts[inv])
            .predict(np.zeros((1, 1)))[0]
        )
        assert median(yy, gg) == reference
    bad = frame[frame.study_group != "S4"]
    bad.to_csv(d / "bad.csv", index=False)
    try:
        run(d / "bad.csv", ["q"], d / "badout")
    except ValueError as e:
        assert "five" in str(e)
    else:
        raise AssertionError("Must reject fewer than five groups")
    print(
        "PASS: exact deterministic rerun, matched row IDs, study disjointness, training imputation, weighted median, feasibility rejection"
    )
