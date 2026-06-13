# linguistic-landscape-geoguesser
A software and app seeking to predict geographic location or locale based on linguistic landscape data. NLP Project.

## Loading and Cleaning the Data

### Concat Model
For a basic model where all of the strings of text of an intersection are combined into a single "row," use this:
```bash
python -m src.cli prepare-data \
  --input-pattern "data/raw/test2026-06-13/*.xlsx" \
  --coord-csv "data/raw/coordinate_dict/coordinate_dict2026-06-13.csv" \
  --output-csv "data/processed/cleaned_concatenated.csv"
```

### Random Samples Model
For a model that creates random samples by intersection, based on a count of strings (so that prediction data requires less input), use this functionality:
```bash
python -m src.cli prepare-sampled-data \
  --input-pattern "data/raw/test2026-06-13/*.xlsx" \
  --coord-csv "data/raw/coordinate_dict/coordinate_dict2026-06-13.csv" \
  --output-csv "data/processed/cleaned_sampled.csv" \
  --n-samples 50 \
  --min-size 5 \
  --max-size 10 \
  --seed 42
```
## Training the Model

The model can be trained with the data in `data/processed/cleaned_concatenated.csv`, which was extracted from the raw data in `data/raw/` in the loading/cleaning function above.
Next, the train model function gives the user the opportunity to create a pd.DataFrame that compares the predictions against the test data.

The CLI now preserves previous exports by generating a unique file name when the requested output path already exists. It also logs each export to `logs/export_runs.csv`, including run metadata and mean error.

### Concat Model
```bash
python -m src.cli train-model \
  --data-csv data/processed/cleaned_concatenated.csv \
  --model-path models/coord_model.joblib \
  --test-output-csv data/processed/test_predictions.csv \
  --alpha 0.01 \
  --test-size 0.2 \
  --random-state 42
```

### Random Samples Model
Save the Sampled Data in a Separate Model to not Overwrite
```bash
python -m src.cli train-model \
  --data-csv data/processed/cleaned_sampled.csv \
  --model-path models/coord_model_sample.joblib \
  --test-output-csv data/processed/test_predictions.csv \
  --alpha 0.1 \
  --test-size 0.2 \
  --random-state 42
```

If `data/processed/test_predictions.csv` already exists, the CLI will write a new file with a timestamp and run ID suffix.

## Plotting the Training vs. Testing Data

```bash
python -m src.cli plot-predictions \
  --prediction-csv filename.csv
```
