# Data cleaning pipeline

Turns raw CAMELS data (`basin_dataset_public_v1p2`) into one table: **one row per gauge per day**.
Scope: Pacific Northwest only (region `17`, 91 basins, 1980–2014).

## Data

All three sources are linked by the 8-digit gauge ID (e.g. `12043000`).

| Source | One row per | Contents |
|---|---|---|
| `usgs_streamflow/17/` | gauge, day | daily flow (the label) |
| `basin_mean_forcing/daymet/17/` | gauge, day | daily weather (gauge ID is only in the filename) |
| `basin_metadata/` | gauge | location, area, elevation, slope, forest, climate averages |

## Pipeline

| Function | Owner | Output |
|---|---|---|
| `clean_daymet` | Alex | `gauge_id`, `date`, weather columns |
| `clean_usgs_streamflow` | Franzi | `gauge_id`, `date`, `flow` |
| `clean_basin_metadata` | JJ | one row per gauge |
| `join_data` | | streamflow + daymet on `gauge_id` + `date`, then metadata on `gauge_id` |
| `generate_csv` | | writes the final CSV |

Rules: each function returns a DataFrame, `gauge_id` stays text (keeps leading zeros), and dates go in one `date` column. `hucs=("17",)` picks the regions.

## Notes

- Runoff is dropped from metadata: it comes from streamflow, so it would leak the label.
- Missing flow is stored as `-999` and must be removed.

## Run

```python
metadata = clean_basin_metadata(basin_metadata_dir)
daymet = clean_daymet(daymet_mean_forcing_dir)
streamflow = clean_usgs_streamflow(usgs_streamflow_dir)

dataset = join_data(streamflow, daymet, metadata)
generate_csv(dataset, "pnw_dataset.csv")
```
