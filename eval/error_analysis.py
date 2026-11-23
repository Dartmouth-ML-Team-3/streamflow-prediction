"""Error analysis of the 1 x 128 LSTM (paper Section 4.3 and Figure 6).

Scores the saved model on the test years exactly as eval.ipynb does
(180-day windows per basin, standardisation fitted on the four LSTM training
years), then reports where the errors are:

  * NSE on the log(1 + runoff) target versus NSE on runoff in mm per day
  * median predicted / observed ratio on high-flow days
  * high-flow recall and precision
  * share of high-flow days missed in April to June versus October to March

and saves the miss-by-month bar chart to ../figures/fig6_miss_by_month.png.

Run from the eval folder with the project kernel:
    python error_analysis.py
"""
import os
import numpy as np
import pandas as pd
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "datacleaning-pipeline", "pnw_model_dataset.csv.gz")
MODEL = os.path.join(HERE, "..", "lstm_model", "lstm_128x1.keras")
FIG = os.path.join(HERE, "..", "figures", "fig6_miss_by_month")

input_columns = ['prcp_z', 'tmax_z', 'srad_z', 'vp_z', 'dayl_z', 'drainage_area_km2_z',
                 'elevation_m_z', 'slope_m_per_km_z', 'frac_forest_z',
                 'mean_precip_mm_day_z', 'mean_pet_mm_day_z']
target_column = "log_runoff"
sequence_length = 180
batch_size = 256


def nse(obs, sim):
    return 1.0 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)


# ---- data, same split as lstm.ipynb: last training water year is validation
df = pd.read_csv(DATA, dtype={"gauge_id": str}, parse_dates=["date"])
df = df.sort_values(["gauge_id", "date"]).reset_index(drop=True)
training_df = df[(df.split == "train") & (df.date < "2010-10-01")]
testing_df = df[df.split == "test"]
y_mu = float(training_df[target_column].mean())
y_sd = float(training_df[target_column].std(ddof=0))



def load_lstm(path):
    """load_model, or rebuild from config.json and load the weights when the file
    was saved by a newer Keras than the one installed."""
    try:
        return keras.models.load_model(path)
    except TypeError:
        import json, tempfile, zipfile
        from keras import layers
        z = zipfile.ZipFile(path)
        cfg = json.loads(z.read("config.json"))["config"]["layers"]
        m = keras.Sequential()
        for L in cfg:
            c, k = L["config"], L["class_name"]
            if k == "InputLayer":
                m.add(layers.Input(shape=tuple(c["batch_shape"][1:])))
            elif k == "LSTM":
                m.add(layers.LSTM(c["units"], return_sequences=c["return_sequences"]))
            elif k == "Dropout":
                m.add(layers.Dropout(c["rate"]))
            elif k == "Dense":
                m.add(layers.Dense(c["units"], activation=c["activation"]))
            else:
                raise ValueError(f"unexpected layer {k}")
        with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as f:
            f.write(z.read("model.weights.h5"))
        m.load_weights(f.name)
        os.unlink(f.name)
        return m


model = load_lstm(MODEL)

# ---- predict every test day that has a 180-day history
rows = []
for gauge_id, gauge in testing_df.groupby("gauge_id"):
    gauge = gauge.sort_values("date")
    x = gauge[input_columns].to_numpy(np.float32)
    windows = keras.utils.timeseries_dataset_from_array(
        x, None, sequence_length=sequence_length, batch_size=batch_size, shuffle=False)
    pred_log = model.predict(windows, verbose=0).ravel() * y_sd + y_mu
    obs_log = gauge[target_column].to_numpy(np.float32)[sequence_length - 1:]
    dates = gauge.date.to_numpy()[sequence_length - 1:]
    keep = np.isfinite(obs_log)
    rows.append(pd.DataFrame({
        "gauge_id": gauge_id, "date": dates[keep],
        "obs_log": obs_log[keep], "pred_log": pred_log[keep]}))
test = pd.concat(rows, ignore_index=True)
test["obs"] = np.expm1(test.obs_log)
test["pred"] = np.clip(np.expm1(test.pred_log), 0, None)
test["pred_log"] = np.log1p(test.pred)          # log of the clipped prediction

# ---- high-flow days: above the basin's 95th-percentile runoff in the training years
cutoff = df[df.split == "train"].groupby("gauge_id")["runoff_mm_day"].quantile(0.95)
test = test.join(cutoff.rename("cutoff"), on="gauge_id")
test["obs_high"] = test.obs > test.cutoff
test["pred_high"] = test.pred > test.cutoff
high = test[test.obs_high]
hits = (high.pred_high).sum()

print(f"scored days {len(test):,}  basins {test.gauge_id.nunique()}  high-flow days {len(high):,}")
print(f"NSE on log(1 + runoff)          {nse(test.obs_log.to_numpy(), test.pred_log.to_numpy()):.3f}")
print(f"NSE on runoff (mm per day)      {nse(test.obs.to_numpy(), test.pred.to_numpy()):.3f}")
print(f"median pred/obs on high-flow days {np.median(high.pred / high.obs):.3f}")
print(f"high-flow recall                {hits / len(high):.3f}")
print(f"high-flow precision             {hits / test.pred_high.sum():.3f}")

# ---- misses by month
high = high.assign(month=high.date.dt.month, missed=~high.pred_high)
by_month = high.groupby("month").agg(missed=("missed", "mean"), days=("missed", "size"))
amj = high[high.month.isin([4, 5, 6])].missed.mean()
ondjfm = high[high.month.isin([10, 11, 12, 1, 2, 3])].missed.mean()
print(f"missed, all months              {high.missed.mean():.3f}")
print(f"missed, April to June           {amj:.3f}")
print(f"missed, October to March        {ondjfm:.3f}")
print(by_month.assign(missed=lambda d: d.missed.round(3)).to_string())

# ---- Figure 6
order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
names = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
miss = by_month.missed.reindex(order)
n = by_month.days.reindex(order)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(6.4, 3.8))
cols = ["#D55E00" if m in (4, 5, 6) else "#0072B2" for m in order]
ax.bar(range(12), miss.values * 100, color=cols, width=0.72)
for i, (v, k) in enumerate(zip(miss.values, n.values)):
    ax.text(i, v * 100 + 1.5, f"{k}", ha="center", va="bottom", fontsize=7, color="#555555")
ax.set_xticks(range(12))
ax.set_xticklabels(names)
ax.set_ylabel("High-flow days missed (%)")
ax.set_ylim(0, 100)
overall = high.missed.mean() * 100
ax.axhline(overall, color="#333333", lw=0.8, ls="--")
ax.text(11.4, overall + 2, f"all months {overall:.0f}%", ha="right", fontsize=8, color="#333333")
ax.text(11.4, 93, f"April to June: {amj * 100:.0f}% missed (snowmelt)", color="#D55E00",
        fontsize=9, fontweight="bold", ha="right")
ax.text(-0.3, 93, f"October to March: {ondjfm * 100:.0f}% missed", color="#0072B2",
        fontsize=9, fontweight="bold", ha="left")
ax.text(-0.3, 78, "numbers above bars are\nhigh-flow days in that month", ha="left",
        fontsize=7, color="#555555")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("1 × 128 LSTM, test years 2012 to 2014, 82 basins", fontsize=9, loc="left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{FIG}.{ext}", dpi=300, bbox_inches="tight")
print(f"saved {FIG}.png and .pdf")
