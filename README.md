# linguistic-landscape-geoguesser
A software and app seeking to predict geographic location or locale based on linguistic landscape data. NLP Project.

## Loading and Cleaning the Data

```bash
python -m src.cli prepare-data \
  --input-pattern "data/raw/test2026-06-13/*.xlsx" \
  --coord-csv "data/raw/coordinate_dict/coordinate_dict2026-05-29.csv" \
  --output-csv "data/processed/cleaned_concatenated.csv"
```


## Training the Model

The model can be trained with the data in `data/processed/cleaned_concatenated.csv`, which was extracted from the raw data in `data/raw/` in the loading/cleaning function above.
Next, the train model function gives the user the opportunity to create a pd.DataFrame that compares the predictions against the test data.

The CLI now preserves previous exports by generating a unique file name when the requested output path already exists. It also logs each export to `logs/export_runs.csv`, including run metadata and mean error.

```bash
python -m src.cli train-model \
  --data-csv data/processed/cleaned_concatenated.csv \
  --model-path models/coord_model.joblib \
  --test-output-csv data/processed/test_predictions.csv \
  --alpha 0.1 \
  --test-size 0.2 \
  --random-state 42
```

If `data/processed/test_predictions.csv` already exists, the CLI will write a new file with a timestamp and run ID suffix.

## Plotting the Training vs. Testing Data

```bash
python -m src.cli plot-predictions \
  --prediction-csv data/processed/filename.csv \
  --output-dir data/plots
```
