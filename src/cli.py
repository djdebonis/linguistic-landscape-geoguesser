import click
import pandas as pd

from .data.loader import prepare_training_data
from .models.model import create_coord_model, save_model, train_coord_model
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
def train_model(data_csv: str, model_path: str, alpha: float) -> None:
    """Train the coordinate prediction model."""
    df = pd.read_csv(data_csv)
    model = create_coord_model(alpha=alpha)
    X = df["text_blob"]
    y = df[["latitude", "longitude"]]
    trained = train_coord_model(model, X, y)
    save_model(trained, model_path)
    click.echo(f"Trained model saved to {model_path}")


@cli.command("predict")
@click.option("--model-path", required=True, help="Path to a trained model file.")
@click.option("--text", required=True, help="Linguistic landscape text input.")
def predict(model_path: str, text: str) -> None:
    """Predict coordinates from text input."""
    lat, lon = predict_text(model_path, text)
    click.echo(f"predicted_latitude={lat:.6f} predicted_longitude={lon:.6f}")


if __name__ == "__main__":
    cli()
