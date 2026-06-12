import pandas as pd

def clean_string_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    collapse_whitespace: bool = True,
    strip: bool = True,
    keep_numbers: bool = True,
    keep_underscore: bool = True,
) -> pd.DataFrame:
    """
    Clean text columns in a DataFrame (optionally lowercasing, removing punctuation, etc.).

    Returns a NEW DataFrame (does not mutate the original).
    """
    out = df.copy()

    # Ensure string dtype (preserves missing values nicely)
    for col in columns:
        out[col] = out[col].astype("string")

    if lowercase:
        for col in columns:
            out[col] = out[col].str.lower()

    if remove_punctuation:
        # Build a regex for "allowed characters" and remove everything else.
        allowed = "a-z0-9" if keep_numbers else "a-z"
        underscore = "_" if keep_underscore else ""
        pattern = rf"[^{allowed}{underscore}\s]"  # remove anything NOT allowed
        for col in columns:
            out[col] = out[col].str.replace(pattern, "", regex=True)

    if collapse_whitespace:
        for col in columns:
            out[col] = out[col].str.replace(r"\s+", " ", regex=True)

    if strip:
        for col in columns:
            out[col] = out[col].str.strip()

    return out

def split_words(df, column, new_column):
    """
    Split `df[column]` into a list of words using .split(" "),
    and store the result in `df[new_column]`.
    """
    out = df.copy()
    out[new_column] = (
        out[column]
        .fillna("")
        .astype(str)
        .str.split(" ")
    )
    return out
