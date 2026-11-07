# streamflow-prediction

Predicting daily river streamflow from weather and basin characteristics, using the CAMELS dataset (`basin_dataset_public_v1p2`).
Three models (random forest, MLP, LSTM) are trained on one shared dataset and compared on the same test years, with extra attention to how each does on high-flow days.
Current scope: Pacific Northwest (region `17`); 82 basins after exclusions; train Oct 2006 – Sep 2011, test Oct 2011 – Sep 2014.

## Structure

| Folder | What it does |
|---|---|
| [`datacleaning-pipeline/`](datacleaning-pipeline/README.md) | Stage 1 cleans and joins streamflow, Daymet weather and basin metadata; stage 2 builds the model dataset (window, exclusions, target, split, scaling); stage 3 is the feature-selection notebook |
| `random_forest_model/` | JJ: scikit-learn random forest regressor |
| `neural_network_model/` | Franzi: Keras MLP |
| `lstm_model/` | Alex: Keras LSTM |

## Getting started

1. Download `basin_timeseries_v1p2_metForcing_obsFlow.zip` (3.4 GB) from the [CAMELS Zenodo record](https://zenodo.org/records/15529996) ([direct link](https://zenodo.org/records/15529996/files/basin_timeseries_v1p2_metForcing_obsFlow.zip?download=1)) and unzip it. Only this one file is needed.
2. Set `base_dir` in `datacleaning-pipeline/clean_data.ipynb` to your local `basin_dataset_public_v1p2` folder.
3. Run All. The last cell writes `datacleaning-pipeline/pnw_dataset.csv`.
4. `cd datacleaning-pipeline && python make_model_dataset.py` writes `pnw_model_dataset.csv`, the file every model loads. Needs pandas, numpy, scikit-learn.

## Team

Dartmouth Thayer School of Engineering

- Alex
- Franzi
- JJ
