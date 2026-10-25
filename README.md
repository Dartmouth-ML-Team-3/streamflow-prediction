# streamflow-prediction

Predicting daily river streamflow from weather and basin characteristics, using the CAMELS dataset (`basin_dataset_public_v1p2`).
Current scope: Pacific Northwest (region `17`, 91 basins, 1980–2014).

## Structure

| Folder | What it does |
|---|---|
| [`datacleaning-pipeline/`](datacleaning-pipeline/README.md) | Cleans and joins streamflow, Daymet weather, and basin metadata into one CSV (one row per gauge per day) |

## Getting started

1. Download `basin_timeseries_v1p2_metForcing_obsFlow.zip` (3.4 GB) from the [CAMELS Zenodo record](https://zenodo.org/records/15529996) ([direct link](https://zenodo.org/records/15529996/files/basin_timeseries_v1p2_metForcing_obsFlow.zip?download=1)) and unzip it. Only this one file is needed.
2. Set `base_dir` in `datacleaning-pipeline/clean_data.ipynb` to your local `basin_dataset_public_v1p2` folder.
3. Run All. The last cell runs the pipeline.

## Team

Dartmouth Thayer School of Engineering

- Alex
- Franzi
- JJ
