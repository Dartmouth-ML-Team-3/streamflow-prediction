# Streamflow prediction (CAMELS, Pacific Northwest)

Predicting daily streamflow in Pacific Northwest rivers from weather and basin characteristics, using the CAMELS dataset, and judging the models on how well they handle high-flow days.

## Language

### Places

**Basin**:
All land that drains to one USGS gauge; the unit every prediction is made for. Identified by the 8-digit gauge ID.
_Avoid_: Catchment, watershed, site

**Gauge**:
The USGS station at a basin's outlet that measures streamflow. Shares its ID with the basin.
_Avoid_: Station, sensor

**Region**:
A USGS HUC-02 drainage region; this project uses region 17 (Pacific Northwest). Selects which basins are studied; nothing is predicted at this level.
_Avoid_: Area, zone

### Observations

**Basin-day**:
One basin on one date; one row of the dataset and one prediction.
_Avoid_: Sample, record, observation

**Streamflow**:
Daily mean volume of water passing a gauge. The target: the number the model predicts. Never an input.
_Avoid_: Discharge, flow rate, water level

**Runoff**:
Streamflow divided by basin area, in mm/day. The unit the target is expressed in, comparable across basins and with rainfall.
_Avoid_: Specific discharge, normalized flow

**Basin attributes**:
Fixed descriptions of a basin (climate, soil, geology, vegetation, topography) that are the same on every basin-day.
_Avoid_: Static features, metadata

### Evaluation

**High-flow day**:
A basin-day whose streamflow exceeds that basin's high-flow cutoff. Used to score models on the days that matter; not a training target.
_Avoid_: Flood, flood day, event

**High-flow cutoff**:
A basin's 95th-percentile runoff over the training years.
_Avoid_: Threshold, flood stage

**Flood**:
Water causing real impacts outside the channel. Used only as motivation; this project does not label floods.
