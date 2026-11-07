"""Stage 2: build the model-ready dataset from pnw_dataset.csv.

Reads the stage-1 table (one row per gauge per day, all 91 gauges, NaN where
flow is missing), keeps one train/test window, drops basins we should not
model, adds the target, labels the split, and scales the inputs.

Run from the terminal with the defaults:

    python make_model_dataset.py

or override any piece:

    python make_model_dataset.py --train-start 2008-10-01 --train-end 2011-09-30 \
        --test-start 2011-10-01 --test-end 2014-09-30 --drop tmin dayl

Output columns, in order:
    gauge_id, date, split, flow_cfs, runoff_mm_day, log_runoff,
    <kept inputs in raw units>, <the same inputs standardized, suffix _z>
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Basins whose two CAMELS area estimates differ by more than 10 %
# (basin_size_errors_10_percent.txt). Addor et al. 2017 advise against using
# them: the basin-mean weather and the runoff conversion both depend on area.
AREA_ERROR_BASINS = (
    "10396000", "12095000", "12115500", "12147600",
    "12388400", "13083000", "14092750", "14158500",
)

# Stage 1 keeps Daymet's unit suffixes; the model notebooks use bare names.
WEATHER_RENAME = {
    "prcp(mm/day)": "prcp",
    "tmax(C)": "tmax",
    "tmin(C)": "tmin",
    "srad(W/m2)": "srad",
    "vp(Pa)": "vp",
    "dayl(s)": "dayl",
}

# The 13 candidate inputs (6 daily weather + 7 basin descriptors).
INPUT_COLS = [
    "prcp", "tmax", "tmin", "srad", "vp", "dayl",
    "drainage_area_km2", "elevation_m", "slope_m_per_km", "frac_forest",
    "mean_precip_mm_day", "mean_pet_mm_day", "mean_temp_c",
]

# Never inputs: identifiers, location (would let a model memorise basins),
# and area_km2, which duplicates drainage_area_km2.
DROP_ALWAYS = ["huc", "gauge_name", "lat", "lon", "area_km2"]

# Inputs removed by feature selection (feature_selection.ipynb, training rows
# Oct 2006 - Sep 2011, 82 basins). Rule: drop only near-copies of another
# input, read as VIF > 10 or |corr| > 0.9.
#   tmin:        VIF 18.5; corr 0.90 with vp, 0.87 with tmax; 0.02 with target
#   mean_temp_c: VIF 8.1;  corr -0.90 with elevation_m, which has twice its
#                target correlation (-0.30 vs 0.16)
# After dropping both, the largest remaining VIF is 5.5 (tmax).
# Override with drop_inputs=... or --drop ... ; pass an empty tuple to keep all 13.
DEFAULT_DROP_INPUTS = ("tmin", "mean_temp_c")


def make_model_dataset(
    in_path,
    out_path,
    train_start="2006-10-01",
    train_end="2011-09-30",
    test_start="2011-10-01",
    test_end="2014-09-30",
    drop_inputs=DEFAULT_DROP_INPUTS,
    warmup_days=365,
):
    """Build and write the model dataset; returns the DataFrame.

    Dates are inclusive and are water-year boundaries by default
    (a water year runs Oct 1 - Sep 30). Rows in the `warmup_days` before
    train_start are kept with split = "warmup" so the LSTM can build its
    first sequences; they are never targets, and the forest and MLP should
    filter to split in ("train", "test").
    """
    train_start, train_end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    test_start, test_end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    if not (train_start <= train_end < test_start <= test_end):
        raise ValueError("need train_start <= train_end < test_start <= test_end")
    warmup_start = train_start - pd.Timedelta(days=warmup_days)

    # 1. read; gauge_id stays text so leading zeros survive
    df = pd.read_csv(in_path, dtype={"gauge_id": str}, parse_dates=["date"])
    df = df.rename(columns=WEATHER_RENAME)
    n_start = df["gauge_id"].nunique()

    # 2. keep the window (warm-up + train + test)
    df = df[(df["date"] >= warmup_start) & (df["date"] <= test_end)].copy()

    # 3. drop the area-error basins
    df = df[~df["gauge_id"].isin(AREA_ERROR_BASINS)]
    n_after_area = df["gauge_id"].nunique()

    # 4. drop any basin with a gap inside train + test (NaN flow or missing day)
    in_train = (df["date"] >= train_start) & (df["date"] <= train_end)
    in_test = (df["date"] >= test_start) & (df["date"] <= test_end)
    modelled = df[in_train | in_test]
    expected_days = (train_end - train_start).days + 1 + (test_end - test_start).days + 1
    per_basin = modelled.groupby("gauge_id")["flow_cfs"].agg(["size", "count"])
    complete = per_basin[(per_basin["size"] == expected_days) & (per_basin["count"] == expected_days)]
    dropped_for_gaps = sorted(set(per_basin.index) - set(complete.index))
    df = df[df["gauge_id"].isin(complete.index)].copy()

    # 5. target: log(1 + runoff in mm/day). Undo with np.expm1 before scoring.
    df["log_runoff"] = np.log1p(df["runoff_mm_day"])

    # 6. split label
    df["split"] = np.select(
        [df["date"] < train_start, df["date"] <= train_end, df["date"] >= test_start],
        ["warmup", "train", "test"],
        default="unused",
    )
    df = df[df["split"] != "unused"]

    # 7. drop non-inputs and whatever feature selection removed
    drop_inputs = tuple(drop_inputs)
    unknown = set(drop_inputs) - set(INPUT_COLS)
    if unknown:
        raise ValueError(f"drop_inputs not in INPUT_COLS: {sorted(unknown)}")
    inputs = [c for c in INPUT_COLS if c not in drop_inputs]
    df = df.drop(columns=DROP_ALWAYS + list(drop_inputs))

    # 8. standardize the inputs, fit on train rows only, applied to every row
    scaler = StandardScaler().fit(df.loc[df["split"] == "train", inputs])
    z_cols = [f"{c}_z" for c in inputs]
    df[z_cols] = scaler.transform(df[inputs])

    # 9. order columns and write
    front = ["gauge_id", "date", "split", "flow_cfs", "runoff_mm_day", "log_runoff"]
    df = df[front + inputs + z_cols].sort_values(["gauge_id", "date"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)

    print(f"basins: {n_start} in -> {n_after_area} after area-error drop "
          f"-> {df['gauge_id'].nunique()} after gap check")
    if dropped_for_gaps:
        print(f"  dropped for gaps in {train_start.date()}..{test_end.date()}: {dropped_for_gaps}")
    print("rows per split:", df["split"].value_counts().to_dict())
    print(f"inputs ({len(inputs)}): {inputs}")
    if drop_inputs:
        print(f"dropped by feature selection: {list(drop_inputs)}")
    print(f"wrote {out_path}")
    return df


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", default=os.path.join(here, "pnw_dataset.csv"))
    p.add_argument("--out", dest="out_path", default=os.path.join(here, "pnw_model_dataset.csv"))
    p.add_argument("--train-start", default="2006-10-01")
    p.add_argument("--train-end", default="2011-09-30")
    p.add_argument("--test-start", default="2011-10-01")
    p.add_argument("--test-end", default="2014-09-30")
    p.add_argument("--drop", nargs="*", default=list(DEFAULT_DROP_INPUTS),
                   help="inputs to leave out, e.g. --drop tmin dayl")
    p.add_argument("--warmup-days", type=int, default=365)
    a = p.parse_args()
    make_model_dataset(a.in_path, a.out_path, a.train_start, a.train_end,
                       a.test_start, a.test_end, a.drop, a.warmup_days)
