# Data pipeline

Turns raw CAMELS data (`basin_dataset_public_v1p2`) into the table every model trains on.
Scope: Pacific Northwest only (region `17`). Two stages: stage 1 cleans and joins the raw files;
stage 2 picks the study window, drops basins we should not model, adds the target, and scales the inputs.

## Data

All three sources are linked by the 8-digit gauge ID (e.g. `12043000`).

| Source | One row per | Contents |
|---|---|---|
| `usgs_streamflow/17/` | gauge, day | daily flow (the label) |
| `basin_mean_forcing/daymet/17/` | gauge, day | daily weather (gauge ID is only in the filename) |
| `basin_metadata/` | gauge | location, area, elevation, slope, forest, climate averages |

## Stage 1: clean and join (`clean_data.ipynb`)

| Function | Owner | Output |
|---|---|---|
| `clean_daymet` | Alex | `gauge_id`, `date`, weather columns |
| `clean_usgs_streamflow` | Franzi | `gauge_id`, `date`, `flow_cfs` (missing days kept as NaN) |
| `clean_basin_metadata` | JJ | one row per gauge |
| `join_data` | | streamflow + daymet on `gauge_id` + `date`, then metadata on `gauge_id`; adds `runoff_mm_day = flow_cfs × 2.446575 / drainage_area_km2` |
| `generate_csv` | | writes `pnw_dataset.csv` (91 gauges, 1980–2014, 1,135,850 rows) |

Rules: each function returns a DataFrame, `gauge_id` stays text (keeps leading zeros), and dates go in one `date` column. `hucs=("17",)` picks the regions.

Notes:
- Mean annual runoff is dropped from metadata: it comes from streamflow, so it would leak the label.
- Missing flow (`-999`) is kept as NaN here. Stage 2 decides which basins to drop, because only stage 2 knows the study window.

Run: set `base_dir` in the second cell to your local `basin_dataset_public_v1p2` folder, then Run All. Takes about 90 s.

## Stage 2: build the model dataset (`make_model_dataset.py`)

```bash
python make_model_dataset.py                                   # defaults below
python make_model_dataset.py --drop                            # keep all 13 inputs
python make_model_dataset.py --train-start 2008-10-01 --train-end 2011-09-30 \
                             --test-start 2011-10-01 --test-end 2014-09-30
```

or from Python: `from make_model_dataset import make_model_dataset`.

| Argument | Default | Meaning |
|---|---|---|
| `train_start`, `train_end` | 2006-10-01, 2011-09-30 | 5 water years (a water year runs Oct 1 – Sep 30) |
| `test_start`, `test_end` | 2011-10-01, 2014-09-30 | 3 water years; scored once, nothing tuned on it |
| `warmup_days` | 365 | weather rows kept before `train_start`, labelled `warmup`, so the LSTM can build its first sequences; never targets |
| `drop_inputs` | `("tmin", "mean_temp_c")` | inputs removed by feature selection (see stage 3) |

What it does, in order:

1. Reads `pnw_dataset.csv`; renames the weather columns to `prcp`, `tmax`, `tmin`, `srad`, `vp`, `dayl`.
2. Keeps rows from `train_start − warmup_days` through `test_end`.
3. Drops the 8 basins whose two CAMELS area estimates differ by more than 10 % (Addor et al. 2017; `basin_size_errors_10_percent.txt`): 10396000, 12095000, 12115500, 12147600, 12388400, 13083000, 14092750, 14158500.
4. Drops any basin with a NaN flow or a missing day inside train + test. With the defaults this removes one basin, 12383500, whose record ends in September 2010.
5. Adds the target `log_runoff = log(1 + runoff_mm_day)`. Undo with `np.expm1` before scoring.
6. Labels each row `split` = `warmup`, `train` or `test`.
7. Drops `huc`, `gauge_name`, `lat`, `lon`, `area_km2` (duplicate of `drainage_area_km2`) and the `drop_inputs`.
8. Standardizes the remaining inputs with a `StandardScaler` fit on **train rows only**, written as `<name>_z` next to the raw columns.
9. Writes `pnw_model_dataset.csv`, sorted by gauge and date.

Output with the defaults: **82 basins**, 149,732 train rows, 89,872 test rows, 29,930 warm-up rows, 28 columns:

```
gauge_id, date, split, flow_cfs, runoff_mm_day, log_runoff,
prcp, tmax, srad, vp, dayl, drainage_area_km2, elevation_m, slope_m_per_km, frac_forest, mean_precip_mm_day, mean_pet_mm_day,
prcp_z, tmax_z, srad_z, vp_z, dayl_z, drainage_area_km2_z, elevation_m_z, slope_m_per_km_z, frac_forest_z, mean_precip_mm_day_z, mean_pet_mm_day_z
```

Models: random forest uses the raw input columns; MLP and LSTM use the `_z` columns. Everyone filters to `split in ("train", "test")` except the LSTM, which may read `warmup` rows as sequence history. Read it back with `pd.read_csv(path, dtype={"gauge_id": str}, parse_dates=["date"])`.

## Stage 3: feature selection (`feature_selection.ipynb`)

Correlation heatmap and variance inflation factors (VIF) for the 13 candidate inputs, on training rows only. Rule: drop an input only if it is a near-copy of another (VIF > 10 or |corr| > 0.9).

Result: `tmin` (VIF 18.5; corr 0.90 with `vp`, 0.87 with `tmax`) and `mean_temp_c` (corr −0.90 with `elevation_m`, which has twice its target correlation) are dropped. The largest remaining VIF is 5.5. Final inputs: 11. The decision is the default in `make_model_dataset.py`; `--drop` with nothing after it keeps all 13.

## Files

| File | Tracked? |
|---|---|
| `clean_data.ipynb`, `make_model_dataset.py`, `feature_selection.ipynb`, `data_names.py` | yes |
| `pnw_dataset.csv` (195 MB), `pnw_model_dataset.csv` (110 MB) | no, over GitHub's limit; regenerate with the two steps above |
