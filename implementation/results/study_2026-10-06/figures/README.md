# Case-study figures

Written by `src/analyze/plot_study.py`; do not edit by hand. Each figure is drawn at
the thesis's text width (include it with `width=\textwidth`), as a vector PDF and a
300 dpi PNG, with the numbers behind it in `data/<name>.csv`.

Every data file has the same columns: `panel`, `scenario`, `households`, `series`, `x_name`, `x`, `y_name`, `y`, `y_low`, `y_high`. `y` is a mean over seeds 7, 8 and 9 and `y_low`, `y_high` their range, when the
point has seeds; a cell that does not apply is empty.

## traffic_week

`traffic_week.pdf` · `traffic_week.png` · `data/traffic_week.csv`

Messages sent per hour over one test week by 10,000 households, in each traffic scenario. With no drift every day follows BurstGPT's average day. Mild drift adds BurstGPT's variation within each week, surges of about 25% lasting about 3 hours; strong drift adds all of its variation around the average day, swings of about 45 to 60% lasting about 12 hours, whole weeks included.

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
