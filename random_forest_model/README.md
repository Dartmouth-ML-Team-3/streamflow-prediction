# Random forest

Owner: JJ. scikit-learn `RandomForestRegressor` predicting log(1 + runoff mm/day) from the shared `pnw_model_dataset.csv`.

## Run

1. Build the dataset: run `datacleaning-pipeline/clean_data.ipynb`, then `python datacleaning-pipeline/make_model_dataset.py`.
2. Open `random_forest.ipynb`, pick a Python kernel with pandas, scikit-learn and matplotlib, Run All. Training takes one to two minutes.

## Settings

200 trees, unlimited depth, `random_state=16`, all other defaults. Inputs are the 11 raw columns (no scaling needed for trees). Train = `split == "train"`, test = `split == "test"`; warm-up rows are not used.

## Results (test years Oct 2011 – Sep 2014, 82 basins, original scale)

| | Random forest | Climatology |
|---|---|---|
| NSE pooled | 0.569 | 0.168 |
| NSE per-basin median | 0.495 | 0.073 |
| RMSE (mm/day) | 4.31 | 5.99 |
| High-flow recall | 0.206 | 0.070 |
| High-flow precision | 0.585 | 0.160 |

Top feature importances: mean_precip_mm_day 0.28, tmax 0.22, dayl 0.15, prcp 0.08. The forest tracks ordinary days but clips flood peaks and drops too fast after storms, the expected weakness of a model with no memory of previous days' rain. Per basin, it does best on high snow-fed basins (above 1,500 m, NSE 0.6–0.75, a regular melt cycle) and worst on the 800–1,400 m band (0.2–0.4) where rain and snow mix; the one failure (NSE −2.1) is 13161500, the semi-arid Bruneau River, scored on a drought after its wettest training year.
