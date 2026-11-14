from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

PROBES = ["vendor_hint", "routing_enabled", "lldp_enabled", "snmp_available", "interface_count_band", "vlan_capable", "firmware_family"]

LABEL_CANDIDATES = ["label", "device_class", "class_label", "target", "true_label"]

MISSING_TOKEN = "__MISSING__"
UNKNOWN_TOKEN = "__UNKNOWN__"


def read_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def find_label_column(df: pd.DataFrame) -> str:
    for name in LABEL_CANDIDATES:
        if name in df.columns:
            return name

    raise ValueError("Cannot find label column. " f"Available columns: {list(df.columns)}")


def normalize_probe_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [name for name in PROBES if name not in df.columns]

    if missing_columns:
        raise ValueError("Missing probe columns: " + ", ".join(missing_columns))

    result = df[PROBES].copy()

    for name in PROBES:
        values = result[name].astype(str).str.strip()
        values = values.replace({"": MISSING_TOKEN, "None": MISSING_TOKEN, "none": MISSING_TOKEN, "null": MISSING_TOKEN, "NULL": MISSING_TOKEN, "nan": MISSING_TOKEN, "NaN": MISSING_TOKEN})
        result[name] = values

    return result


def make_nb_pipeline(alpha: float = 1.0) -> Pipeline:
    """Categorical Naive Bayes for discrete probes.

    The encoder learns categories only from training data.
    Unknown validation categories get a reserved integer code.
    """

    categorical = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64))]
    )

    transformer = ColumnTransformer(transformers=[("categorical", categorical, PROBES)], remainder="drop")

    # CategoricalNB requires non-negative integer codes.
    # OrdinalEncoder returns -1 for unseen categories.
    # Adding 1 maps unknown -> 0 and known -> 1, 2, ...
    return Pipeline(steps=[("preprocess", transformer), ("shift", AddOneTransformer()), ("classifier", CategoricalNB(alpha=alpha))])


from sklearn.base import BaseEstimator, TransformerMixin


class AddOneTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = np.asarray(X, dtype=np.int64) + 1

        if np.any(result < 0):
            raise ValueError("CategoricalNB received negative categories")

        return result


def evaluate_predictions(y_true, y_pred) -> dict:
    labels = sorted(set(y_true) | set(y_pred))

    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def train_evaluate_nb(train_df: pd.DataFrame, eval_df: pd.DataFrame, alpha: float = 1.0) -> tuple[Pipeline, dict]:
    train_label = find_label_column(train_df)
    eval_label = find_label_column(eval_df)

    X_train = normalize_probe_frame(train_df)
    X_eval = normalize_probe_frame(eval_df)

    y_train = train_df[train_label].astype(str)
    y_eval = eval_df[eval_label].astype(str)

    if y_train.eq("").any() or y_eval.eq("").any():
        raise ValueError("Empty labels found in dataset")

    model = make_nb_pipeline(alpha=alpha)

    model.fit(X_train, y_train)

    predictions = model.predict(X_eval)

    metrics = evaluate_predictions(y_eval, predictions)

    return model, metrics


def missingness_report(df: pd.DataFrame) -> dict:
    X = normalize_probe_frame(df)

    present = X.ne(MISSING_TOKEN)

    n_rows = len(X)

    return {
        "n_observations": n_rows,
        "n_probes": len(PROBES),
        "overall_missing_rate": float(1.0 - present.to_numpy().mean()),
        "mean_observed_probes": float(present.sum(axis=1).mean()),
        "per_probe": {probe: {"present": int(present[probe].sum()), "missing": int((~present[probe]).sum()), "missing_rate": float((~present[probe]).mean())} for probe in PROBES},
        "observed_probe_count_distribution": {str(int(count)): int(frequency) for count, frequency in (present.sum(axis=1).value_counts().sort_index().items())},
    }


def conditional_distribution(df: pd.DataFrame, probe: str, target: str) -> dict:
    """P(target | observed probe value), training data only."""
    if probe not in PROBES:
        raise ValueError(f"Unknown probe: {probe}")

    X = normalize_probe_frame(df)

    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    frame = pd.DataFrame({"probe": X[probe], "target": df[target].astype(str)})

    # Exclude missing probe values: a missing observation
    # is not a statement about the device property.
    frame = frame[frame["probe"] != MISSING_TOKEN]

    if frame.empty:
        return {}

    counts = pd.crosstab(frame["probe"], frame["target"])

    proportions = counts.div(counts.sum(axis=1), axis=0)

    result = {}

    for probe_value in proportions.index:
        result[str(probe_value)] = {"n": int(counts.loc[probe_value].sum()), "distribution": {str(label): float(probability) for label, probability in (proportions.loc[probe_value].items())}}

    return result


def derive_vendor_role(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    label_col = find_label_column(result)

    labels = result[label_col].astype(str)

    valid_roles = labels.str.endswith(("_router", "_switch"))

    if not valid_roles.all():
        bad = labels[~valid_roles].head().tolist()
        raise ValueError(f"Unexpected device labels: {bad}")

    result["vendor"] = labels.str.rsplit("_", n=1).str[0]

    result["role"] = labels.str.rsplit("_", n=1).str[-1]

    return result


def probe_diagnostics(train_df: pd.DataFrame) -> dict:
    derived = derive_vendor_role(train_df)

    return {
        "firmware_to_vendor": conditional_distribution(derived, "firmware_family", "vendor"),
        "interface_to_role": conditional_distribution(derived, "interface_count_band", "role"),
        "routing_to_role": conditional_distribution(derived, "routing_enabled", "role"),
        "vlan_to_role": conditional_distribution(derived, "vlan_capable", "role"),
        "vendor_hint_to_vendor": conditional_distribution(derived, "vendor_hint", "vendor"),
    }


def full_evidence_report(df: pd.DataFrame) -> dict:
    """Measure label ambiguity among fully observed signatures.

    If identical 7-probe signatures have different labels,
    no deterministic classifier using only these probes can
    correctly classify all such observations.
    """
    label_col = find_label_column(df)
    X = normalize_probe_frame(df)

    fully_observed = X.ne(MISSING_TOKEN).all(axis=1)

    complete_X = X.loc[fully_observed]
    complete_y = df.loc[fully_observed, label_col].astype(str)

    if complete_X.empty:
        return {"n_full_observations": 0, "n_unique_signatures": 0, "empirical_signature_upper_bound": None, "ambiguous_signatures": 0}

    combined = complete_X.copy()
    combined["_label"] = complete_y.to_numpy()

    counts = combined.groupby(PROBES + ["_label"], dropna=False).size().rename("count").reset_index()

    signature_counts = counts.groupby(PROBES, dropna=False)["count"]

    max_correct = int(signature_counts.max().sum())

    n_signatures = int(signature_counts.ngroups)

    ambiguous = int((signature_counts.size() > 1).sum())

    return {"n_full_observations": int(len(complete_X)), "n_unique_signatures": n_signatures, "ambiguous_signatures": ambiguous, "empirical_signature_upper_bound": (max_correct / len(complete_X))}


def print_conditional_table(name: str, values: dict) -> None:
    print(f"\n{name}")
    print("-" * 60)

    if not values:
        print("No observed values")
        return

    for probe_value, entry in values.items():
        distribution = entry["distribution"]

        probabilities = ", ".join(f"{label}={probability:.3f}" for label, probability in sorted(distribution.items()))

        print(f"{probe_value} (n={entry['n']}): " f"{probabilities}")


def main() -> None:
    parser = argparse.ArgumentParser(description=("Phase 3.7: Simulator diagnostics " "and Naive Bayes baseline"))

    parser.add_argument("--train", default="data/generated/moderate/train.csv")

    parser.add_argument("--val", default="data/generated/moderate/val.csv")

    parser.add_argument("--full-train", default=None, help="Optional fully observed training CSV")

    parser.add_argument("--full-val", default=None, help="Optional fully observed validation CSV")

    parser.add_argument("--output", default="artifacts/e1/simulator_diagnostics.json")

    parser.add_argument("--alpha", type=float, default=1.0)

    args = parser.parse_args()

    if args.alpha <= 0:
        parser.error("--alpha must be positive")

    if bool(args.full_train) != bool(args.full_val):
        parser.error("--full-train and --full-val must " "be supplied together")

    train_df = read_dataset(args.train)
    val_df = read_dataset(args.val)

    print("\nPHASE 3.7 — SIMULATOR DIAGNOSTICS")
    print("=" * 60)

    missingness = missingness_report(train_df)

    print("Training observations:", missingness["n_observations"])
    print("Overall missing rate:", f"{missingness['overall_missing_rate']:.2%}")
    print("Mean observed probes:", f"{missingness['mean_observed_probes']:.3f}")

    _, nb_metrics = train_evaluate_nb(train_df, val_df, alpha=args.alpha)

    print("\nNAIVE BAYES — SPARSE VALIDATION")
    print("-" * 60)
    print("Accuracy:", f"{nb_metrics['accuracy']:.4f}")
    print("Macro-F1:", f"{nb_metrics['macro_f1']:.4f}")

    diagnostics = probe_diagnostics(train_df)

    for name, values in diagnostics.items():
        print_conditional_table(name, values)

    results = {
        "experiment": "phase_3_7",
        "train_path": args.train,
        "val_path": args.val,
        "alpha": args.alpha,
        "missingness": missingness,
        "naive_bayes_sparse": nb_metrics,
        "probe_diagnostics": diagnostics,
    }

    if args.full_train is not None:
        full_train = read_dataset(args.full_train)
        full_val = read_dataset(args.full_val)

        full_train_missing = missingness_report(full_train)
        full_val_missing = missingness_report(full_val)

        if full_train_missing["overall_missing_rate"] > 0 or full_val_missing["overall_missing_rate"] > 0:
            raise ValueError("Full-evidence datasets still contain " "missing probe values")

        _, full_nb_metrics = train_evaluate_nb(full_train, full_val, alpha=args.alpha)

        full_signatures = full_evidence_report(full_val)

        print("\nNAIVE BAYES — FULL EVIDENCE")
        print("-" * 60)
        print("Accuracy:", f"{full_nb_metrics['accuracy']:.4f}")
        print("Macro-F1:", f"{full_nb_metrics['macro_f1']:.4f}")

        print("\nFULL-EVIDENCE SIGNATURES")
        print("-" * 60)
        print("Unique signatures:", full_signatures["n_unique_signatures"])
        print("Ambiguous signatures:", full_signatures["ambiguous_signatures"])
        print("Empirical upper bound:", full_signatures["empirical_signature_upper_bound"])

        results["naive_bayes_full"] = full_nb_metrics
        results["full_evidence_signatures"] = full_signatures

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
