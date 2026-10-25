"""Calculate grouped metrics from retained PointRAFT benchmark predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_FILES = {
    # "Linear regression": "results/model_comparison/predictions/linear_regression.csv",
    "PointNet": "results/model_comparison/predictions/pointnet.csv",
    "PointNet++": "results/model_comparison/predictions/pointnet2.csv",
    "DGCNN": "results/model_comparison/predictions/dgcnn.csv",
    "PointMLP": "results/model_comparison/predictions/pointmlp.csv",
    "RandLA-Net": "results/model_comparison/predictions/randlanet.csv",
    # "PTv3-S": "results/model_comparison/predictions/ptv3.csv",
    "PTv3-Compact": "results/model_comparison/predictions/ptv3_compact.csv",
    "PointRAFT": "results/pointraft/predictions.csv",
}

# Ground-truth weight classes; each is (lower, upper] in grams.
WEIGHT_CLASS_EDGES_G = (0, 100, 200, 300, 400, 500, 600)
WEIGHT_CLASS_LABELS = [
    f"{lower}-{upper}"
    for lower, upper in zip(WEIGHT_CLASS_EDGES_G[:-1], WEIGHT_CLASS_EDGES_G[1:])
]


def parse_args() -> argparse.Namespace:
    """Parse command-line paths.

    Returns
    -------
    argparse.Namespace
        Parsed project root and output path.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_metadata(project_root: Path) -> pd.DataFrame:
    """Load and validate test-tuber metadata.

    Parameters
    ----------
    project_root
        PointRAFT benchmark repository root.

    Returns
    -------
    pandas.DataFrame
        Test metadata indexed by the point-cloud parent identifier.
    """
    data_root = project_root / "data" / "3DPotatoTwin"
    ground_truth = pd.read_csv(data_root / "ground_truth.csv")
    splits = pd.read_csv(data_root / "splits.csv")
    metadata = ground_truth.merge(splits, on="label", validate="one_to_one")
    metadata = metadata.loc[metadata["split"] == "test"].copy()
    metadata = metadata.set_index("label")
    if not metadata.index.is_unique:
        raise ValueError("Ground-truth labels are not unique")
    if len(metadata) != 172:
        raise ValueError(f"Expected 172 test tubers, received {len(metadata)}")
    return metadata


def load_predictions(
    project_root: Path,
    relative_path: str,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Load one model's predictions and attach grouping metadata.

    Parameters
    ----------
    project_root
        PointRAFT benchmark repository root.
    relative_path
        Prediction CSV path relative to the repository root.
    metadata
        Validated test-tuber metadata indexed by label.

    Returns
    -------
    pandas.DataFrame
        Validated predictions with cultivar, season, and weight-class columns.
    """
    predictions = pd.read_csv(project_root / relative_path)
    required = {"file_name", "unique_id", "gt", "pred"}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if len(predictions) != 5_254:
        raise ValueError(
            f"Expected 5,254 predictions in {relative_path}, received "
            f"{len(predictions)}"
        )

    predictions["sample_id"] = predictions["file_name"].map(
        lambda value: Path(value).name
    )
    if predictions["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample identifiers in {relative_path}")
    unknown_ids = set(predictions["unique_id"]).difference(metadata.index)
    if unknown_ids:
        raise ValueError(f"Unknown test identifiers in {relative_path}: {unknown_ids}")

    expected_targets = predictions["unique_id"].map(metadata["weight_g_inctack"])
    if not np.allclose(predictions["gt"], expected_targets, atol=5e-4, rtol=0):
        maximum_error = np.max(np.abs(predictions["gt"] - expected_targets))
        raise ValueError(
            f"Ground-truth mismatch in {relative_path}: max error={maximum_error}"
        )

    predictions["cultivar"] = predictions["unique_id"].map(metadata["cultivar"])
    predictions["growing_season"] = predictions["unique_id"].map(
        metadata["growing_season"]
    )
    predictions["weight_class"] = pd.cut(
        predictions["gt"],
        bins=WEIGHT_CLASS_EDGES_G,
        labels=WEIGHT_CLASS_LABELS,
        right=True,
    )
    if predictions["weight_class"].isna().any():
        raise ValueError(
            f"Ground truth outside the weight classes in {relative_path}: "
            f"{WEIGHT_CLASS_EDGES_G[0]}-{WEIGHT_CLASS_EDGES_G[-1]} g"
        )
    return predictions


def calculate_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Calculate sample-level regression metrics for one subset.

    Parameters
    ----------
    predictions
        Prediction rows for one model and subset.

    Returns
    -------
    dict
        Tuber count, point-cloud count, MAE, RMSE, and R-squared.
    """
    residuals = predictions["pred"] - predictions["gt"]
    sum_squared_residuals = float(np.square(residuals).sum())
    centered_targets = predictions["gt"] - predictions["gt"].mean()
    sum_squared_targets = float(np.square(centered_targets).sum())
    return {
        "tubers": int(predictions["unique_id"].nunique()),
        "point_clouds": int(len(predictions)),
        "mae_g": float(np.abs(residuals).mean()),
        "rmse_g": float(np.sqrt(np.square(residuals).mean())),
        "r2": 1.0 - sum_squared_residuals / sum_squared_targets,
    }


def calculate_grouped_metrics(
    model: str,
    predictions: pd.DataFrame,
) -> list[dict[str, float | int | str]]:
    """Calculate cultivar, season, weight-class, and overall metrics for one model.

    Parameters
    ----------
    model
        Display name for the benchmark model.
    predictions
        Validated model predictions with grouping columns.

    Returns
    -------
    list of dict
        Long-form grouped metric records.
    """
    records: list[dict[str, float | int | str]] = []
    group_specs = (
        ("cultivar", "cultivar"),
        ("growing_season", "season"),
        ("weight_class", "weight_class"),
    )
    for column, group_type in group_specs:
        for group, subset in predictions.groupby(column, sort=True, observed=True):
            records.append(
                {
                    "model": model,
                    "group_type": group_type,
                    "group": str(group),
                    **calculate_metrics(subset),
                }
            )
    records.append(
        {
            "model": model,
            "group_type": "overall",
            "group": "Entire test set",
            **calculate_metrics(predictions),
        }
    )
    return records


def main() -> None:
    """Validate retained predictions and write grouped metrics."""
    args = parse_args()
    project_root = args.project_root.resolve()
    metadata = load_metadata(project_root)
    records: list[dict[str, float | int | str]] = []
    reference_samples: set[str] | None = None

    for model, relative_path in MODEL_FILES.items():
        predictions = load_predictions(project_root, relative_path, metadata)
        sample_ids = set(predictions["sample_id"])
        if reference_samples is None:
            reference_samples = sample_ids
        elif sample_ids != reference_samples:
            raise ValueError(f"Prediction sample set differs for {model}")
        records.extend(calculate_grouped_metrics(model, predictions))

    output = pd.DataFrame.from_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, float_format="%.8f")
    print(output.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote {len(output)} grouped records to {args.output}")


if __name__ == "__main__":
    main()
