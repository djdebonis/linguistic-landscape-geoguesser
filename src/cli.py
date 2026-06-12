import click
import pandas as pd

from .data.loader import prepare_training_data
from .models.model import (
    create_coord_model,
    evaluate_model,
    predictions_dataframe,
    save_model,
    split_dataset,
    train_coord_model,
)
from .predict.predict import predict_text


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

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
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

    if train_output_csv:
        train_df = predictions_dataframe(trained, X_train, y_train)
        train_df.to_csv(train_output_csv, index=False)
        click.echo(f"Saved train predictions to {train_output_csv}")

    if test_output_csv:
        test_df = predictions_dataframe(trained, X_test, y_test)
        test_df.to_csv(test_output_csv, index=False)
        click.echo(f"Saved test predictions to {test_output_csv}")


@cli.command("predict")
@click.option("--model-path", required=True, help="Path to a trained model file.")
@click.option("--text", required=True, help="Linguistic landscape text input.")
def predict(model_path: str, text: str) -> None:
    """Predict coordinates from text input."""
    lat, lon = predict_text(model_path, text)
    click.echo(f"predicted_latitude={lat:.6f} predicted_longitude={lon:.6f}")


if __name__ == "__main__":
    cli()
