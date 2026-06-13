import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import click
import pandas as pd

from .data.loader import prepare_training_data, prepare_sampled_training_data
from .models.model import (
    create_coord_model,
    evaluate_model,
    predictions_dataframe,
    save_model,
    split_dataset,
    train_coord_model,
)
from .predict.predict import predict_text
from .visualize_compare import plot_prediction_vs_actual_testing


def get_git_commit_hash() -> str | None:
    """Return the current git commit hash or None if git is unavailable."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return commit_hash
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def append_export_log(log_path: str, row: dict[str, object]) -> None:
    """Append a row to the export log CSV, creating the file if needed."""
    log_path_obj = Path(log_path)
    log_path_obj.parent.mkdir(parents=True, exist_ok=True)
    export_df = pd.DataFrame([row])
    if log_path_obj.exists():
        export_df.to_csv(log_path_obj, mode="a", header=False, index=False)
    else:
        export_df.to_csv(log_path_obj, index=False)


def resolve_unique_output_path(path: str, run_id: str) -> str:
    """Return a unique CSV file path if the requested path already exists."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        return str(output_path)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    unique_name = f"{output_path.stem}_{timestamp}_{run_id[:8]}{output_path.suffix}"
    return str(output_path.with_name(unique_name))


def annotate_export_metadata(
    df: pd.DataFrame,
    run_id: str,
    export_timestamp: str,
    git_commit: str | None,
    source_data_csv: str,
    model_path: str,
    alpha: float,
    test_size: float,
    random_state: int,
    export_type: str,
) -> pd.DataFrame:
    """Annotate prediction exports with run metadata."""
    annotated = df.copy()
    annotated["export_run_id"] = run_id
    annotated["export_timestamp"] = export_timestamp
    annotated["export_type"] = export_type
    annotated["source_data_csv"] = source_data_csv
    annotated["model_path"] = model_path
    annotated["git_commit"] = git_commit or ""
    annotated["alpha"] = alpha
    annotated["test_size"] = test_size
    annotated["random_state"] = random_state
    return annotated


@click.group()
def cli() -> None:
    """Command-line interface for the geo-guesser pipeline."""
    pass


@cli.command("prepare-data")
@click.option("--input-pattern", required=True, help="Glob pattern for raw spreadsheet files.")
@click.option("--coord-csv", required=True, help="CSV file containing coordinate lookup data.")
@click.option("--output-csv", required=True, help="Output path for the cleaned dataset CSV.")
@click.option("--coord-col", default="cd", help="Column name containing DMS coordinates.")
def prepare_data(input_pattern: str, coord_csv: str, output_csv: str, coord_col: str) -> None:
    """Prepare training data from raw spreadsheets and coordinate lookup."""
    prepare_training_data(
        input_pattern=input_pattern,
        coord_csv=coord_csv,
        output_csv=output_csv,
        coord_col=coord_col,
    )
    click.echo(f"Prepared training data and wrote {output_csv}")


@cli.command("prepare-sampled-data")
@click.option("--input-pattern", required=True, help="Glob pattern for raw spreadsheet files.")
@click.option("--coord-csv", required=True, help="CSV file containing coordinate lookup data.")
@click.option("--output-csv", required=True, help="Output path for the sampled dataset CSV.")
@click.option("--coord-col", default="cd", help="Column name containing DMS coordinates.")
@click.option("--n-samples", default=50, show_default=True, help="Samples per intersection.")
@click.option("--min-size", default=3, show_default=True, help="Minimum signs per sample.")
@click.option("--max-size", default=7, show_default=True, help="Maximum signs per sample.")
@click.option("--seed", default=42, show_default=True, help="Random seed for sampled generation.")
def prepare_sampled_data(
    input_pattern: str,
    coord_csv: str,
    output_csv: str,
    coord_col: str,
    n_samples: int,
    min_size: int,
    max_size: int,
    seed: int,
) -> None:
    """Prepare sampled training data by creating random text samples per intersection."""
    prepare_sampled_training_data(
        input_pattern=input_pattern,
        coord_csv=coord_csv,
        output_csv=output_csv,
        coord_col=coord_col,
        n_samples=n_samples,
        min_size=min_size,
        max_size=max_size,
        seed=seed,
    )
    click.echo(f"Prepared sampled training data and wrote {output_csv}")


@cli.command("train-model")
@click.option("--data-csv", required=True, help="Cleaned training dataset CSV path.")
@click.option("--model-path", required=True, help="File path to save the trained model.")
@click.option("--alpha", default=0.1, help="Ridge regularization strength.")
@click.option("--test-size", default=0.2, help="Proportion of data to reserve for testing.")
@click.option("--random-state", default=42, help="Random seed for train/test splitting.")
@click.option("--train-output-csv", default=None, help="Optional CSV path to save train actual vs predicted results.")
@click.option("--test-output-csv", default=None, help="Optional CSV path to save test actual vs predicted results.")
def train_model(
    data_csv: str,
    model_path: str,
    alpha: float,
    test_size: float,
    random_state: int,
    train_output_csv: str | None,
    test_output_csv: str | None,
) -> None:
    """Train the coordinate prediction model and evaluate it on a held-out test set."""
    df = pd.read_csv(data_csv)
    X = df["text_blob"]
    y = df[["latitude", "longitude"]]
    groups = df["intersection"] if "intersection" in df.columns else None

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        groups=groups,
    )

    model = create_coord_model(alpha=alpha)
    trained = train_coord_model(model, X_train, y_train)
    save_model(trained, model_path)

    train_metrics = evaluate_model(trained, X_train, y_train)
    test_metrics = evaluate_model(trained, X_test, y_test)

    click.echo(f"Trained model saved to {model_path}")
    click.echo(
        f"Train MAE latitude={train_metrics['mae_latitude']:.4f}, longitude={train_metrics['mae_longitude']:.4f}"
    )
    click.echo(
        f"Test MAE  latitude={test_metrics['mae_latitude']:.4f}, longitude={test_metrics['mae_longitude']:.4f}"
    )

    export_run_id = str(uuid.uuid4())
    export_timestamp = datetime.utcnow().isoformat()
    git_commit = get_git_commit_hash()
    export_log_path = "logs/export_runs.csv"

    if train_output_csv:
        train_output_csv_actual = resolve_unique_output_path(train_output_csv, export_run_id)
        train_df = predictions_dataframe(trained, X_train, y_train)
        train_df = annotate_export_metadata(
            train_df,
            run_id=export_run_id,
            export_timestamp=export_timestamp,
            git_commit=git_commit,
            source_data_csv=data_csv,
            model_path=model_path,
            alpha=alpha,
            test_size=test_size,
            random_state=random_state,
            export_type="train",
        )
        train_df.to_csv(train_output_csv_actual, index=False)
        append_export_log(
            export_log_path,
            {
                "run_id": export_run_id,
                "export_timestamp": export_timestamp,
                "export_type": "train",
                "requested_output_csv": train_output_csv,
                "output_csv": train_output_csv_actual,
                "source_data_csv": data_csv,
                "model_path": model_path,
                "alpha": alpha,
                "test_size": test_size,
                "random_state": random_state,
                "git_commit": git_commit or "",
                "mean_error_km": train_df["error_km"].mean(),
            },
        )
        click.echo(f"Saved train predictions to {train_output_csv_actual}")

    if test_output_csv:
        test_output_csv_actual = resolve_unique_output_path(test_output_csv, export_run_id)
        test_df = predictions_dataframe(trained, X_test, y_test)
        test_df = annotate_export_metadata(
            test_df,
            run_id=export_run_id,
            export_timestamp=export_timestamp,
            git_commit=git_commit,
            source_data_csv=data_csv,
            model_path=model_path,
            alpha=alpha,
            test_size=test_size,
            random_state=random_state,
            export_type="test",
        )
        test_df.to_csv(test_output_csv_actual, index=False)
        append_export_log(
            export_log_path,
            {
                "run_id": export_run_id,
                "export_timestamp": export_timestamp,
                "export_type": "test",
                "requested_output_csv": test_output_csv,
                "output_csv": test_output_csv_actual,
                "source_data_csv": data_csv,
                "model_path": model_path,
                "alpha": alpha,
                "test_size": test_size,
                "random_state": random_state,
                "git_commit": git_commit or "",
                "mean_error_km": test_df["error_km"].mean(),
            },
        )
        click.echo(f"Saved test predictions to {test_output_csv_actual}")


@cli.command("predict")
@click.option("--model-path", required=True, help="Path to a trained model file.")
@click.option("--text", required=True, help="Linguistic landscape text input.")
def predict(model_path: str, text: str) -> None:
    """Predict coordinates from text input."""
    lat, lon = predict_text(model_path, text)
    click.echo(f"predicted_latitude={lat:.6f} predicted_longitude={lon:.6f}")


@cli.command("plot-predictions")
@click.option("--prediction-csv", required=True, help="CSV path containing actual and predicted coordinates.")
@click.option("--output-dir", default="data/plots", help="Directory to save the prediction plot.")
@click.option("--title", default="Prediction vs Actual Coordinates", help="Title for the plot.")
def plot_predictions(prediction_csv: str, output_dir: str, title: str) -> None:
    """Create a prediction-vs-actual coordinate plot from a CSV."""
    df = pd.read_csv(prediction_csv)
    output_path = plot_prediction_vs_actual_testing(
        df,
        title=title,
        output_dir=output_dir,
    )
    click.echo(f"Saved prediction plot to {output_path}")


if __name__ == "__main__":
    cli()
