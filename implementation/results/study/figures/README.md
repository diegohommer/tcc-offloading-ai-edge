# Case-study figures

Written by `src/analyze/plot_study.py`; do not edit by hand. Each figure is drawn at
the thesis's text width (include it with `width=\textwidth`), as a vector PDF and a
300 dpi PNG, with the numbers behind it in `data/<name>.csv`.

Every data file has the same columns: `panel`, `scenario`, `households`, `series`, `x_name`, `x`, `y_name`, `y`, `y_low`, `y_high`. `y` is a mean over seeds 7, 8 and 9 and `y_low`, `y_high` their range, when the
point has seeds; a cell that does not apply is empty.

## traffic_week

`traffic_week.pdf` · `traffic_week.png` · `data/traffic_week.csv`

Messages sent per hour over one test week by 10,000 households, in each traffic scenario. With no drift every day follows WildChat's average day in local time. Mild drift adds BurstGPT's variation within each week, surges of about 25% lasting about 3 hours; strong drift adds all of its variation around the average day, swings of about 45 to 60% lasting about 12 hours, whole weeks included.

## traffic_profiles

`traffic_profiles.pdf` · `traffic_profiles.png` · `data/traffic_profiles.csv`

Share of the day's conversations opening in each hour, local time, on weekdays (a) and weekends (b): WildChat's users worldwide (the study's traffic), its Brazilian users, and BurstGPT's enterprise and campus users.

## beta_knob

`beta_knob.pdf` · `beta_knob.png` · `data/beta_knob.csv`

Accuracy (a) and energy per query (b) against RecServe's escalation quantile $\beta$, with no drift and 10,000 households. Dotted lines: each tier answering every query alone.

## equal_accuracy

`equal_accuracy.pdf` · `equal_accuracy.png` · `data/equal_accuracy.csv`

Energy per query against accuracy, one marker per $\beta$ from 0.1 to 0.9, with no drift and 10,000 households. Circles: each policy read at 0.80 accuracy. Crosses: each tier answering every query alone.

## tiers

`tiers.pdf` · `tiers.png` · `data/tiers.csv`

Share of queries answered at each tier against $\beta$, under RecServe (a) and the broadcast (b), with no drift and 10,000 households.

## batching_validation

`batching_validation.pdf` · `batching_validation.png` · `data/batching_validation.csv`

(a) Energy per generated token on one L4 GPU, from the static batch sweep the simulator uses and from continuous batching. (b) Energy the simulator predicts against what the GPU measured, over 15 Poisson runs.

## saving_over_recserve

`saving_over_recserve.pdf` · `saving_over_recserve.png` · `data/saving_over_recserve.csv`

Energy saved over RecServe at 0.80 accuracy, with no drift, by number of households. Bars: range over three seeds.

## broadcast_over_timetable

`broadcast_over_timetable.pdf` · `broadcast_over_timetable.png` · `data/broadcast_over_timetable.csv`

Energy the broadcast saves over the timetable at 0.80 accuracy, by number of households, with no, mild and strong drift. Bars: range over three seeds.

## drift

`drift.pdf` · `drift.png` · `data/drift.csv`

Energy per query at 0.80 accuracy with no, mild and strong drift, 10,000 households. Bars: range over three seeds.

## bandwidth

`bandwidth.pdf` · `bandwidth.png` · `data/bandwidth.csv`

(a) RecServe's communication burden, the bytes carried between tiers, against accuracy. (b) Change in energy and in communication burden against RecServe at 0.80 accuracy. No drift, 10,000 households; bars: range over three seeds.

## sensitivity

`sensitivity.pdf` · `sensitivity.png` · `data/sensitivity.csv`

Energy the broadcast saves over RecServe and over the timetable at 0.80 accuracy, 10,000 households, in every scenario of the study. Bars: range over three seeds.

## delta

`delta.pdf` · `delta.png` · `data/delta.csv`

(a) Energy the broadcast saves over RecServe at 0.80 accuracy against the margin $\delta$, from 0 (the policy as evaluated) to 1, where no query is moved and the policy is RecServe. (b) Its communication burden and mean latency against RecServe's. No drift, 10,000 households; bars: range over three seeds (one at $\delta = 1$).

## energy_split

`energy_split.pdf` · `energy_split.png` · `data/energy_split.csv`

Energy per query spent at each tier at 0.80 accuracy, under RecServe and under the broadcast with $\delta = 0.3$ and $\delta = 0$. No drift, 10,000 households, seed 7.
