import glob
import random
from pathlib import Path

import pandas as pd

from ..utils.clean_utils import clean_string_columns
from ..utils.geo_utils import parse_dms_coordinate


def load_spreadsheet_files(path_pattern: str) -> pd.DataFrame:
    """Load all Excel files matching a glob pattern into one DataFrame."""
    file_paths = glob.glob(path_pattern, recursive=True)

    if not file_paths:
        pattern_path = Path(path_pattern)
        if pattern_path.parent.exists() and pattern_path.name.startswith("*"):
            suffix = pattern_path.suffix
            if suffix in {".xlsx", ".xls", ".csv"}:
                file_paths = [str(path) for path in pattern_path.parent.rglob(f"*{suffix}")]

    if not file_paths:
        raise FileNotFoundError(f"No files found for pattern: {path_pattern}")

    frames = [pd.read_excel(path) for path in file_paths]
    return pd.concat(frames, ignore_index=True)


def load_coordinate_lookup(coord_csv: str, coord_col: str = "cd") -> pd.DataFrame:
    """Load coordinate lookup data and parse DMS strings into latitude/longitude."""
    coords = pd.read_csv(coord_csv)
    if coord_col not in coords.columns:
        raise ValueError(f"Expected coordinate column '{coord_col}' in {coord_csv}")

    coords = coords.copy()
    coords = coords.dropna(subset=[coord_col])
    coords[["latitude", "longitude"]]
    coords[["latitude", "longitude"]] = (
        coords[coord_col]
        .apply(parse_dms_coordinate)
        .apply(pd.Series)
    )
    return coords


def merge_training_data(
    text_df: pd.DataFrame,
    coord_df: pd.DataFrame,
    on: str = "intersection",
) -> pd.DataFrame:
    """Join text rows to geographic coordinates for training."""
    coord_columns = [on, "latitude", "longitude", "zip", "city"]
    missing = [col for col in coord_columns if col not in coord_df.columns]
    if missing:
        raise ValueError(f"Coordinate lookup is missing columns: {missing}")

    return text_df.merge(
        coord_df[coord_columns],
        on=on,
        how="left"
    )


def generate_samples(
    signs: list[str],
    n_samples: int = 50,
    min_size: int = 3,
    max_size: int = 7,
    seed: int | None = None,
) -> list[str]:
    """Create random sample strings from a list of sign texts."""
    rng = random.Random(seed)
    samples: list[str] = []

    for _ in range(n_samples):
        lower = min(min_size, len(signs))
        upper = min(max_size, len(signs))
        size = rng.randint(lower, upper)
        subset = rng.sample(signs, size)
        samples.append(" ".join(subset))

    return samples


def build_intersection_dataset(
    df: pd.DataFrame,
    text_col: str = "text_on_sign_exact"
) -> pd.DataFrame:
    """Group rows by intersection and concatenate text into a single blob."""
    if text_col not in df.columns:
        raise ValueError(f"Expected text column '{text_col}' in input DataFrame")

    normalized = df.copy()
    normalized[text_col] = normalized[text_col].fillna("").astype(str)

    grouped = (
        normalized.groupby("intersection", dropna=False)
        .agg({
            text_col: " ".join,
            "latitude": "first",
            "longitude": "first",
            "zip": "first",
            "city": "first",
        })
        .reset_index()
        .rename(columns={text_col: "text_blob"})
    )

    grouped = grouped.dropna(subset=["latitude", "longitude"])
    grouped = grouped[grouped["text_blob"].str.strip().astype(bool)]
    return grouped


def build_intersection_sample_dataset(
    df: pd.DataFrame,
    text_col: str = "text_on_sign_exact",
    n_samples: int = 50,
    min_size: int = 3,
    max_size: int = 7,
    seed: int | None = None,
) -> pd.DataFrame:
    """Create a sampled training dataset where each intersection generates multiple rows."""
    if text_col not in df.columns:
        raise ValueError(f"Expected text column '{text_col}' in input DataFrame")

    normalized = df.copy()
    normalized[text_col] = normalized[text_col].fillna("").astype(str)

    sampled_rows: list[dict[str, object]] = []
    current_seed = seed

    for intersection, group in normalized.groupby("intersection", dropna=False):
        signs = group[text_col].str.strip().replace("", pd.NA).dropna().tolist()
        if not signs:
            continue

        lat = group["latitude"].iloc[0]
        lon = group["longitude"].iloc[0]
        zip_code = group["zip"].iloc[0] if "zip" in group.columns else None
        city = group["city"].iloc[0] if "city" in group.columns else None

        samples = generate_samples(
            signs,
            n_samples=n_samples,
            min_size=min_size,
            max_size=max_size,
            seed=current_seed,
        )

        if current_seed is not None:
            current_seed += 1

        for sample_text in samples:
            sampled_rows.append({
                "intersection": intersection,
                "text_blob": sample_text,
                "latitude": lat,
                "longitude": lon,
                "zip": zip_code,
                "city": city,
            })

    sampled_df = pd.DataFrame(sampled_rows)
    sampled_df = sampled_df.dropna(subset=["latitude", "longitude"])
    sampled_df = sampled_df[sampled_df["text_blob"].str.strip().astype(bool)]
    return sampled_df


def save_dataframe(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """Save a DataFrame to CSV, creating directories when needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)


def prepare_training_data(
    input_pattern: str,
    coord_csv: str,
    output_csv: str,
    coord_col: str = "cd",
    text_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Full preprocessing pipeline for raw spreadsheets and coordinate lookup."""
    text_columns = text_columns or ["text_on_sign_exact"]

    raw_text = load_spreadsheet_files(input_pattern)
    cleaned_text = clean_string_columns(raw_text, columns=text_columns)
    coord_lookup = load_coordinate_lookup(coord_csv, coord_col=coord_col)
    merged = merge_training_data(cleaned_text, coord_lookup, on="intersection")
    intersection_df = build_intersection_dataset(merged)
    save_dataframe(intersection_df, output_csv)
    return intersection_df


def prepare_sampled_training_data(
    input_pattern: str,
    coord_csv: str,
    output_csv: str,
    coord_col: str = "cd",
    text_columns: list[str] | None = None,
    n_samples: int = 50,
    min_size: int = 3,
    max_size: int = 7,
    seed: int | None = None,
) -> pd.DataFrame:
    """Full preprocessing pipeline that creates sampled training rows per intersection."""
    text_columns = text_columns or ["text_on_sign_exact"]

    raw_text = load_spreadsheet_files(input_pattern)
    cleaned_text = clean_string_columns(raw_text, columns=text_columns)
    coord_lookup = load_coordinate_lookup(coord_csv, coord_col=coord_col)
    merged = merge_training_data(cleaned_text, coord_lookup, on="intersection")
    sampled_df = build_intersection_sample_dataset(
        merged,
        text_col=text_columns[0],
        n_samples=n_samples,
        min_size=min_size,
        max_size=max_size,
        seed=seed,
    )
    save_dataframe(sampled_df, output_csv)
    return sampled_df
