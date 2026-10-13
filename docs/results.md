# Results

This page reports how the system scores on the SceneDiff benchmark and on a
synthetic suite with exact ground truth. Every setting the selection changed
was chosen on the validation split, and the other defaults were set earlier and
only checked there. The test split was then run once with the settings frozen,
and its numbers are the ones to trust. For what the system does and why, see
the [design note](design.md).

Validation rows are copied from the `summary.json` of a run or from the sweep
log, which holds one line per sweep point. Test rows are copied from the
`summary.json` of the one test run and of its 87-pair pooling. The mean of
three, the change in mean, and the precision, recall, and F1 of the synthetic
suite are computed from the other columns. Run folders are local output and
are not committed. Metrics are rounded to four decimals. The scores are low in
absolute terms: at the chosen settings, test pixel IoU is 0.1174 and AP per
scene is 0.0321, against 0.2533 and 0.1083 on validation.

## What is measured

**Data.** [SceneDiff](https://github.com/yuqunw/scene_diff) SD-V is the
benchmark's set of video pairs of a room, recorded before and after objects were
added, moved, or removed, with a 2D mask of every changed object in the
annotated frames. Only the indoor pairs are used, at the dataset revision pinned in
`src/spatial_change/evaluation/scenediff_fetch.py`.

| Split | Pairs | Use |
|-------|-------|-----|
| validation | 28 (99 ground-truth changed objects) | Every choice of a setting. |
| test | 92 | One run with the settings frozen. |

Five test pairs are the demo pairs (`living_room_3_living_room_4`,
`living_room_37_living_room_38`, `living_room_9_living_room_10`,
`bedroom_28_bedroom_29`, `closet_1_closet_2`). They were used during
development, in the work on gravity alignment and in the first real runs. The
test results are therefore reported for all 92 pairs and for the 87 pairs that
are not demo pairs.

**Metrics.** They follow the SceneDiff reference implementation and score 2D
masks at a common resolution per scene. Each predicted 3D object is projected
into the annotated frames with a depth test and drawn as a mask. The IoU
threshold for a match is 0.5, strictly above.

| Metric | Definition |
|--------|------------|
| Pixel IoU | In each frame, all predicted masks are merged into one mask and all ground-truth masks into another. Overlapping, extra, and missed pixels are summed over every frame of every pair, and IoU is overlap over the sum of the three. |
| AP per image | Each predicted mask is a detection ranked by its score. It matches the best ground-truth mask of the same video and frame. Detections of all pairs are pooled before the average precision is taken. |
| AP per scene | Each predicted object is one detection, ranked by the mean score of its masks. It matches the ground-truth object with the best IoU, where the IoU stacks the masks of both videos. |
| Typed AP per scene | As AP per scene, but a prediction only matches ground truth of the same type: moved (masks in both videos) or static (added or removed). Reported, never used to choose a setting. |
| tp, fp, fn | Objects. Matched predictions, unmatched predictions, and unmatched ground-truth objects, with the type ignored. |
| Mean of three | The mean of pixel IoU, AP per image, and AP per scene. The sweep rule below uses it. |

**How a run is scored.** `spatial-change evaluate` runs the frames of the 30 fps
annotation timeline whose index is a multiple of 30, with up to 64 keyframes
per video and the template describer (descriptions are not scored). Uncertain
changes are left out of the scores (`--uncertain-as drop`). A pair that fails
counts as zero predictions, so its ground truth is missed. The validation and
test runs used an RTX 4070 Laptop GPU with 8 GB.

## How settings were chosen

The selection rules were written before any validation metric was read, apart
from the failure message of one pair. The record of when they were written is an
internal note and is not in the repository.

1. **Uncertain changes.** Compare `drop` and `moved` on the validation pairs
   using the three headline metrics (pixel IoU, AP per image, AP per scene).
   Take the setting that is higher on at least two of the three. On a split or a
   tie, keep `drop`.
2. **Thresholds.** Sweep one parameter at a time from the current defaults,
   for at most two rounds. A value replaces a default only if it raises the mean
   of the three headline metrics by at least 0.01 and lowers none of the three
   by more than 0.01. If several values pass, the largest mean wins. Every sweep
   point is reported on this page.
3. **Ablations.** They are reported and never used to choose a setting.
4. **Test split.** It is run once with the frozen settings, and reported for all
   92 pairs and for the 87 non-demo pairs.

Two facts limit what validation can show. The split is small (28 pairs, 99
objects), so differences of a few thousandths between rows are not evidence. And
the chosen settings are the best of the points tried on the same split that
reports them, so the validation numbers are optimistic. The test run is the
number to trust. Some code changes between the first baseline and the second run
were made after looking at validation failures, for example the fix for the
up-direction failure on `bed_3_bed_4`.

## Validation results

### Progression

Rows 1 and 2 differ in code. Rows 2, 3, and 4 differ only in settings.

| Run | Pairs ok | Pixel IoU | AP per image | AP per scene | Mean of three | Typed AP per scene | tp | fp | fn |
|-----|----------|-----------|--------------|--------------|---------------|--------------------|----|----|----|
| 1. First baseline | 27/28 | 0.1646 | 0.0577 | 0.0343 | 0.0855 | 0.0269 | 15 | 185 | 84 |
| 2. Later code, gates off | 28/28 | 0.1838 | 0.0697 | 0.0453 | 0.0996 | 0.0375 | 16 | 194 | 83 |
| 3. Both gates on | 28/28 | 0.2270 | 0.1149 | 0.1026 | 0.1482 | 0.0848 | 15 | 77 | 84 |
| 4. Gates and `min_free_votes` 1 (chosen) | 28/28 | 0.2533 | 0.1206 | 0.1083 | 0.1607 | 0.0916 | 16 | 80 | 83 |

1. First baseline at commit `b4b70ae`, with the defaults of that commit: gates
   did not exist yet and `visibility.min_free_votes` was 2. `bed_3_bed_4` failed
   at gravity alignment (the up direction was wrong), so it counts as missed.
2. Commit `c7289e6`, which holds all the code changes made after the baseline
   (a 30 fps timeline, frame timing from the container's packet durations, a
   keyframe cap of 64 per video in `evaluate`, the up-sign fix, and the
   optional gates, which are off). All 28 pairs run.
3. The two match gates on, `match.max_occluded_fraction` 0.001 and
   `match.max_box_volume_m3` 1.0, set through a config file.
4. `visibility.min_free_votes` 1 as well. These are the settings that ship as
   defaults.

### Uncertain changes: `drop` or `moved`

This comparison was made on the first baseline, as rule 1 says, before any gate
existed.

| `uncertain_as` | Pixel IoU | AP per image | AP per scene | Typed AP per scene | tp | fp | fn |
|----------------|-----------|--------------|--------------|--------------------|----|----|----|
| drop | 0.1646 | 0.0577 | 0.0343 | 0.0269 | 15 | 185 | 84 |
| moved | 0.1636 | 0.0577 | 0.0342 | 0.0269 | 15 | 191 | 84 |

`drop` is higher on pixel IoU and AP per scene and equal to four decimals on AP
per image, so the rule keeps `drop`. The choice matters more once gates are on:
the first baseline had 6 uncertain changes, and the gated run has 124 (next
table), all of which `drop` leaves out of the scores.

### Gate values

Each gate demotes a change to `uncertain` when it is past the value:

- `match.max_occluded_fraction`: the occluded votes of the change over its
  consistent, free, and occluded votes exceed the value. Unobserved votes are
  not in that share, and a change with none of those votes has the share 0.
- `match.max_box_volume_m3`: the volume of the larger of its before and after
  boxes exceeds the value, in cubic meters.
- `match.min_change_confidence`: its confidence is below the value.

The selection is offline. It does not run the pipeline for each candidate.
It reads the stored per-change results of run 2 (the votes, boxes, and
confidence of every change), demotes each change that is past a candidate
value, and scores what is left with `uncertain_as drop`, a failed pair counting
as missed, as the batch does. It follows the rule above: one gate at a time, in
the order occluded share, volume, confidence, for at most two rounds. Each
value is compared with the settings in effect when its scan started: no gates
for the occluded share in round 1, the adopted occluded share for the volume,
the adopted occluded share and volume for the confidence, and the final
settings for every scan of round 2. Every logged candidate is below. Round 0 is
run 2. The adopted values are in bold.

| Round | Occluded share | Volume (m3) | Confidence | Pixel IoU | AP per image | AP per scene | Mean of three | Change in mean | tp | fp | fn |
|-------|----------------|-------------|------------|-----------|--------------|--------------|---------------|----------------|----|----|----|
| 0 | off | off | off | 0.1838 | 0.0697 | 0.0453 | 0.0996 | base | 16 | 194 | 83 |
| 1 | 0.0 | off | off | 0.2085 | 0.1001 | 0.0862 | 0.1316 | +0.0320 | 15 | 79 | 84 |
| 1 | **0.001** | off | off | 0.2244 | 0.0999 | 0.0853 | 0.1365 | +0.0369 | 15 | 80 | 84 |
| 1 | 0.002 | off | off | 0.2239 | 0.0899 | 0.0741 | 0.1293 | +0.0297 | 15 | 83 | 84 |
| 1 | 0.005 | off | off | 0.2264 | 0.0822 | 0.0691 | 0.1259 | +0.0263 | 15 | 91 | 84 |
| 1 | 0.01 | off | off | 0.2295 | 0.0815 | 0.0686 | 0.1265 | +0.0269 | 16 | 95 | 83 |
| 1 | 0.02 | off | off | 0.2470 | 0.0828 | 0.0642 | 0.1313 | +0.0317 | 16 | 107 | 83 |
| 1 | 0.05 | off | off | 0.2486 | 0.0773 | 0.0554 | 0.1271 | +0.0275 | 16 | 132 | 83 |
| 1 | 0.1 | off | off | 0.2407 | 0.0759 | 0.0541 | 0.1236 | +0.0240 | 16 | 148 | 83 |
| 1 | 0.001 | 0.05 | off | 0.0814 | 0.0741 | 0.0815 | 0.0790 | -0.0576 | 11 | 53 | 88 |
| 1 | 0.001 | 0.1 | off | 0.1902 | 0.1181 | 0.1071 | 0.1385 | +0.0019 | 14 | 61 | 85 |
| 1 | 0.001 | 0.2 | off | 0.1961 | 0.1145 | 0.1103 | 0.1403 | +0.0038 | 15 | 69 | 84 |
| 1 | 0.001 | 0.3 | off | 0.2141 | 0.1132 | 0.1065 | 0.1446 | +0.0081 | 15 | 71 | 84 |
| 1 | 0.001 | 0.5 | off | 0.2109 | 0.1113 | 0.1039 | 0.1420 | +0.0055 | 15 | 74 | 84 |
| 1 | **0.001** | **1.0** | off | 0.2270 | 0.1149 | 0.1026 | 0.1482 | +0.0116 | 15 | 77 | 84 |
| 1 | 0.001 | 1.0 | 0.05 | 0.2407 | 0.1146 | 0.1026 | 0.1526 | +0.0045 | 15 | 62 | 84 |
| 1 | 0.001 | 1.0 | 0.1 | 0.2425 | 0.1146 | 0.1026 | 0.1533 | +0.0051 | 15 | 45 | 84 |
| 1 | 0.001 | 1.0 | 0.2 | 0.2424 | 0.1119 | 0.0997 | 0.1514 | +0.0032 | 14 | 34 | 85 |
| 1 | 0.001 | 1.0 | 0.3 | 0.2425 | 0.1105 | 0.0966 | 0.1499 | +0.0017 | 13 | 26 | 86 |
| 2 | 0.0 | 1.0 | off | 0.2109 | 0.1156 | 0.1037 | 0.1434 | -0.0048 | 15 | 76 | 84 |
| 2 | 0.001 | 1.0 | off | 0.2270 | 0.1149 | 0.1026 | 0.1482 | 0.0000 | 15 | 77 | 84 |
| 2 | 0.002 | 1.0 | off | 0.2267 | 0.1112 | 0.0993 | 0.1457 | -0.0024 | 15 | 79 | 84 |
| 2 | 0.005 | 1.0 | off | 0.2221 | 0.1089 | 0.0963 | 0.1424 | -0.0058 | 15 | 86 | 84 |
| 2 | 0.01 | 1.0 | off | 0.2253 | 0.1042 | 0.0921 | 0.1405 | -0.0077 | 16 | 90 | 83 |
| 2 | 0.02 | 1.0 | off | 0.2163 | 0.0972 | 0.0868 | 0.1335 | -0.0147 | 16 | 100 | 83 |
| 2 | 0.05 | 1.0 | off | 0.2373 | 0.0955 | 0.0758 | 0.1362 | -0.0120 | 16 | 120 | 83 |
| 2 | 0.1 | 1.0 | off | 0.2320 | 0.0936 | 0.0741 | 0.1332 | -0.0149 | 16 | 135 | 83 |
| 2 | 0.001 | 0.05 | off | 0.0814 | 0.0741 | 0.0815 | 0.0790 | -0.0692 | 11 | 53 | 88 |
| 2 | 0.001 | 0.1 | off | 0.1902 | 0.1181 | 0.1071 | 0.1385 | -0.0097 | 14 | 61 | 85 |
| 2 | 0.001 | 0.2 | off | 0.1961 | 0.1145 | 0.1103 | 0.1403 | -0.0079 | 15 | 69 | 84 |
| 2 | 0.001 | 0.3 | off | 0.2141 | 0.1132 | 0.1065 | 0.1446 | -0.0036 | 15 | 71 | 84 |
| 2 | 0.001 | 0.5 | off | 0.2109 | 0.1113 | 0.1039 | 0.1420 | -0.0061 | 15 | 74 | 84 |
| 2 | 0.001 | 1.0 | off | 0.2270 | 0.1149 | 0.1026 | 0.1482 | 0.0000 | 15 | 77 | 84 |
| 2 | 0.001 | 1.0 | 0.05 | 0.2407 | 0.1146 | 0.1026 | 0.1526 | +0.0045 | 15 | 62 | 84 |
| 2 | 0.001 | 1.0 | 0.1 | 0.2425 | 0.1146 | 0.1026 | 0.1533 | +0.0051 | 15 | 45 | 84 |
| 2 | 0.001 | 1.0 | 0.2 | 0.2424 | 0.1119 | 0.0997 | 0.1514 | +0.0032 | 14 | 34 | 85 |
| 2 | 0.001 | 1.0 | 0.3 | 0.2425 | 0.1105 | 0.0966 | 0.1499 | +0.0017 | 13 | 26 | 86 |

In round 1, an occluded share of 0.001 is the largest mean among the values that
pass (+0.0369 on the mean), and then a volume of 1.0 passes
(+0.0116). No confidence value reaches +0.01 (the best, 0.1, gives
+0.0051), so that gate stays off. In round 2 no value passes, and
the selection ends. The round 2 rows for the volume and the confidence repeat
the metrics of round 1, because their settings are the same; only the base of
the change in mean differs. The grid stops at 1.0 for the volume and 0.3 for the
confidence, so larger values were not tried. The same values were chosen on the
first baseline.

Run 3 is the real run that confirmed the selection. The pipeline with the two
chosen gates gave the same metrics as the offline score, to every digit of the
stored result: 0.2270, 0.1149, and 0.1026, with 15 tp, 77 fp, and 84 fn.

The gates demote changes to `uncertain`. They do not remove them from the
report. The number of changes is the same with and without them:

| Run | added | moved | removed | uncertain | All |
|-----|-------|-------|---------|-----------|-----|
| 2. Gates off | 82 | 38 | 90 | 6 | 216 |
| 3. Gates on | 32 | 20 | 40 | 124 | 216 |

### Threshold sweep

The sweep covers seven parameters. The table gives the default of each one. In
round 1 the base is the gated run (run 3 above, mean 0.1482), and each parameter
takes two other values. In round 2 the base is the single winner of round 1
(`visibility.min_free_votes` 1, mean 0.1607), and the six other parameters take
two other values each. That makes 27 points: the round 1 base, 14 round 1
points, and 12 round 2 points. The round of a point follows from its settings: a
point that sets `visibility.min_free_votes` to 1 together with another
parameter is a round 2 point. The change in mean is relative to the base of the
round. The sweep log records 305 s to 842 s per point (the round 1 base has no
time).

| Parameter | Default |
|-----------|---------|
| `candidates.link_radius` | 2 |
| `candidates.min_changed_fraction` | 0.3 |
| `lift.min_agreement` | 0.5 |
| `lift.merge_overlap` | 0.5 |
| `appearance.changed_below` | 0.4 |
| `match.min_fitness` | 0.85 |
| `visibility.min_free_votes` | 2 |

**Round 1** (base: both gates on, mean 0.1482)

| Parameter | Value | Pixel IoU | AP per image | AP per scene | Mean of three | Change in mean | tp | fp | fn |
|-----------|-------|-----------|--------------|--------------|---------------|----------------|----|----|----|
| (base: gates only) | - | 0.2270 | 0.1149 | 0.1026 | 0.1482 | base | 15 | 77 | 84 |
| `candidates.link_radius` | 1 | 0.2184 | 0.1053 | 0.0777 | 0.1338 | -0.0143 | 17 | 125 | 82 |
| `candidates.link_radius` | 3 | 0.2137 | 0.0944 | 0.0685 | 0.1255 | -0.0227 | 10 | 64 | 89 |
| `candidates.min_changed_fraction` | 0.5 | 0.2004 | 0.1065 | 0.0838 | 0.1302 | -0.0179 | 13 | 59 | 86 |
| `candidates.min_changed_fraction` | 0.7 | 0.1545 | 0.0958 | 0.0745 | 0.1082 | -0.0399 | 10 | 30 | 89 |
| `lift.min_agreement` | 0.7 | 0.2289 | 0.1105 | 0.0923 | 0.1439 | -0.0043 | 14 | 79 | 85 |
| `lift.min_agreement` | 0.85 | 0.2285 | 0.1105 | 0.0923 | 0.1438 | -0.0044 | 14 | 79 | 85 |
| `lift.merge_overlap` | 0.7 | 0.2267 | 0.1143 | 0.1013 | 0.1475 | -0.0007 | 15 | 79 | 84 |
| `lift.merge_overlap` | 0.9 | 0.2267 | 0.1143 | 0.1013 | 0.1475 | -0.0007 | 15 | 79 | 84 |
| `appearance.changed_below` | 0.3 | 0.2272 | 0.1149 | 0.1026 | 0.1482 | +0.0001 | 15 | 77 | 84 |
| `appearance.changed_below` | 0.5 | 0.2262 | 0.1137 | 0.0973 | 0.1458 | -0.0024 | 15 | 77 | 84 |
| `match.min_fitness` | 0.55 | 0.2175 | 0.1063 | 0.1074 | 0.1437 | -0.0045 | 14 | 55 | 85 |
| `match.min_fitness` | 0.7 | 0.2175 | 0.1001 | 0.0953 | 0.1376 | -0.0106 | 15 | 58 | 84 |
| **`visibility.min_free_votes`** | **1** | 0.2533 | 0.1206 | 0.1083 | 0.1607 | +0.0125 | 16 | 80 | 83 |
| `visibility.min_free_votes` | 3 | 0.2304 | 0.1030 | 0.0892 | 0.1408 | -0.0073 | 12 | 81 | 87 |

Only `visibility.min_free_votes` 1 raises the mean by 0.01 or more (+0.0125),
and it lowers none of the three metrics. It is adopted.

**Round 2** (base: both gates on and `visibility.min_free_votes` 1, mean 0.1607)

| Parameter | Value | Pixel IoU | AP per image | AP per scene | Mean of three | Change in mean | tp | fp | fn |
|-----------|-------|-----------|--------------|--------------|---------------|----------------|----|----|----|
| (base: `min_free_votes` 1) | - | 0.2533 | 0.1206 | 0.1083 | 0.1607 | base | 16 | 80 | 83 |
| `candidates.link_radius` | 1 | 0.2819 | 0.1300 | 0.0855 | 0.1658 | +0.0051 | 19 | 126 | 80 |
| `candidates.link_radius` | 3 | 0.2057 | 0.0964 | 0.0830 | 0.1283 | -0.0324 | 11 | 60 | 88 |
| `candidates.min_changed_fraction` | 0.5 | 0.2571 | 0.1216 | 0.0985 | 0.1591 | -0.0016 | 15 | 61 | 84 |
| `candidates.min_changed_fraction` | 0.7 | 0.2120 | 0.1083 | 0.0725 | 0.1309 | -0.0298 | 10 | 31 | 89 |
| `lift.min_agreement` | 0.7 | 0.2571 | 0.1246 | 0.1038 | 0.1618 | +0.0011 | 16 | 80 | 83 |
| `lift.min_agreement` | 0.85 | 0.2569 | 0.1246 | 0.1037 | 0.1617 | +0.0010 | 16 | 81 | 83 |
| `lift.merge_overlap` | 0.7 | 0.2530 | 0.1208 | 0.1083 | 0.1607 | 0.0000 | 16 | 81 | 83 |
| `lift.merge_overlap` | 0.9 | 0.2530 | 0.1208 | 0.1083 | 0.1607 | 0.0000 | 16 | 82 | 83 |
| `appearance.changed_below` | 0.3 | 0.2533 | 0.1206 | 0.1083 | 0.1607 | 0.0000 | 16 | 80 | 83 |
| `appearance.changed_below` | 0.5 | 0.2531 | 0.1204 | 0.1116 | 0.1617 | +0.0010 | 16 | 81 | 83 |
| `match.min_fitness` | 0.55 | 0.2472 | 0.1123 | 0.0873 | 0.1489 | -0.0118 | 13 | 64 | 86 |
| `match.min_fitness` | 0.7 | 0.2807 | 0.1168 | 0.0819 | 0.1598 | -0.0009 | 14 | 67 | 85 |

No point raises the mean by 0.01. The largest gain is `candidates.link_radius`
1 (+0.0051), which also lowers AP per scene by 0.0227. The second condition of
the rule never decided a case: in both rounds, no point other than the adopted
one reached +0.01 on the mean. The sweep stops after round 2.

**Chosen settings.** Both gates on (`match.max_occluded_fraction` 0.001,
`match.max_box_volume_m3` 1.0) and `visibility.min_free_votes` 1. On the
validation split they give pixel IoU 0.2533, AP per image 0.1206, AP per scene
0.1083, and typed AP per scene 0.0916, with 16 tp, 80 fp, and 83 fn, and all 28
pairs run. These are the code defaults.

## Ablations

The ablations run on top of the chosen settings, with the same validation pairs
and the same stage cache. They are reported and were not used to choose
anything. Every row runs all 28 pairs.

| Row | Pixel IoU | AP per image | AP per scene | Mean of three | tp | fp | fn |
|-----|-----------|--------------|--------------|---------------|----|----|----|
| full | 0.2533 | 0.1206 | 0.1083 | 0.1607 | 16 | 80 | 83 |
| geometry_veto | 0.2533 | 0.1206 | 0.1083 | 0.1607 | 16 | 80 | 83 |
| geometry_only | 0.0527 | 0.0429 | 0.0450 | 0.0469 | 5 | 48 | 94 |
| no_veto_no_occluded_gate | 0.1427 | 0.0921 | 0.0601 | 0.0983 | 9 | 149 | 90 |
| no_gates | 0.1909 | 0.0814 | 0.0493 | 0.1072 | 17 | 210 | 82 |
| frames_8 | 0.1497 | 0.0516 | 0.0413 | 0.0808 | 12 | 112 | 87 |
| long_side_392 | 0.2362 | 0.1067 | 0.0689 | 0.1373 | 14 | 88 | 85 |

What each row changes from the chosen settings:

- **full**: nothing. This is the chosen sweep point, not a second run.
- **geometry_veto**: `appearance.changed_below` -1.0, the lowest value allowed,
  which turns the appearance cue off. The visibility veto stays on.
- **geometry_only**: the appearance cue off and `visibility.veto` false. The
  veto is the only difference from geometry_veto. Both gates stay on.
- **no_veto_no_occluded_gate**: `visibility.veto` false and
  `match.max_occluded_fraction` null. The volume gate stays on and the
  appearance cue stays at its default.
- **no_gates**: both match gates null. The veto and the appearance cue stay on.
- **frames_8**: top-level `max_keyframes` 8 instead of 64 per video. The
  reconstruction is computed again.
- **long_side_392**: top-level `backbone_long_side` 392 instead of 518. The
  reconstruction is computed again.

What the rows show:

- Turning the appearance cue off changes nothing at four decimals. At the
  default threshold of 0.4 the cue almost never fires: in the cached similarity
  data, 109 of 2,753,191 points are below 0.4, and the lowest cosine is 0.369.
- Turning the veto off (geometry_only) leaves 5 tp and a mean of 0.0469, against
  16 tp and 0.1607. Both gates stay on in that row, so it does not show what
  the veto does without the gates.
- Turning both gates off (no_gates) gives 17 tp and 210 fp, against 16 tp and 80
  fp with them on. The gates cost one true positive on the validation split and
  remove 130 false positives.
- Fewer keyframes per video (frames_8) lower all three headline metrics: pixel
  IoU from 0.2533 to 0.1497, AP per image from 0.1206 to 0.0516, and AP per
  scene from 0.1083 to 0.0413. Matched objects fall from 16 to 12 and false
  positives rise from 80 to 112.
- A smaller backbone input (long_side_392) also lowers all three, and AP per
  scene the most: 0.1083 to 0.0689, against 0.2533 to 0.2362 for pixel IoU and
  0.1206 to 0.1067 for AP per image.

What could not be ablated, and why:

- **Appearance on or off.** There is no switch. A threshold of -1.0 is the
  closest thing, and it is what geometry_veto uses.
- **Higher resolution.** Pi3X resizes its input to at most 153,000 pixels
  (about 518 x 294), and that limit is not a run setting, so a larger
  `backbone_long_side` than 518 would change nothing. It needs a code change.
- **Higher frame budget.** `evaluate` already allows 64 keyframes per video,
  and the validation videos give 4 to 15 frames per video at stride 30. Only
  a lower budget changes the input.
- **Labels and relations.** The scorer ignores labels and relations, so they
  cannot move the metrics.
- **ICP refinement.** It cannot be switched off (`max_rounds` is at least 1).

## Synthetic suite

The synthetic suite builds room pairs with exact ground truth
(`src/spatial_change/synthetic/`), runs the pipeline on them with fake models
on a CPU, and scores the changes of a run against the known ones
(`src/spatial_change/evaluation/synthetic.py`). It places unchanged objects
behind other objects and in zones the other video cannot see, on purpose. A
change finds a changed object when both have the same type and every box center
is within 0.35 m of the true one. An uncertain change finds nothing and counts
as a false positive. Eleven seeds give 33 changed objects. The runs below are on
commit `1ec2429`.

| Setting | Seeds | tp | fp | fn | Precision | Recall | F1 |
|---------|-------|----|----|----|-----------|--------|----|
| Gates off, veto on | 11 | 32 | 2 | 1 | 0.941 | 0.970 | 0.955 |
| Gates off, veto off | 3 | 6 | 20 | 3 | 0.231 | 0.667 | 0.343 |
| Both shipped gates on | 11 | 10 | 24 | 23 | 0.294 | 0.303 | 0.299 |
| Occluded-share gate alone | 11 | 10 | 24 | 23 | 0.294 | 0.303 | 0.299 |
| Volume gate alone | 11 | 32 | 2 | 1 | 0.941 | 0.970 | 0.955 |

The first row is the setting of the shipped synthetic test, which turns the
default gates off (`UNGATED_GATES`). The no-veto row uses the first 3 seeds
only, so it does not compare directly with the others. False positives at an
occluded or an unobserved unchanged object (`fp_occluded` and `fp_unobserved`)
are 0 in every row.

**The tradeoff.** With both shipped gates on, 23 changes are `uncertain` and
the misses rise from 1 to 23 of the 33 changed objects, so F1 falls from 0.955
to 0.299. The occluded-share gate alone gives the same
numbers, and the volume gate alone changes nothing. The occluded-share gate
demotes a change when its occluded votes over its consistent, free, and
occluded votes exceed 0.001. Unobserved votes are not in that share. That limit
is low, and these scenes put objects behind other objects on purpose, so most of
their changes are demoted. On the real validation
captures the two gates together cut false positives from 210 to 80 for one true
positive (ablation table). The defaults follow the validation choice, because
the rules take their evidence from validation only, and the synthetic suite was
not used to tune anything. This is a known limitation. A capture with much
occlusion may report many changes as `uncertain`, and the report keeps their
boxes.

## Test results

The test split was run once, on commit `1ec2429` with no config file, so with
the shipped defaults: the chosen settings above. Every pair of the test run
records the same settings digest as the confirming validation run. Nothing was
changed after the test numbers were read. All 92 pairs ran and none failed, so
the metrics of the ok pairs alone are the same as the headline ones. The run
took 3,222 s on the same GPU, 18 s to 150 s per pair (median 30 s).

| Pairs | Pairs ok | Pixel IoU | AP per image | AP per scene | Mean of three | Typed AP per scene | tp | fp | fn |
|-------|----------|-----------|--------------|--------------|---------------|--------------------|----|----|----|
| Test, all 92 | 92/92 | 0.1174 | 0.0602 | 0.0321 | 0.0699 | 0.0239 | 22 | 283 | 232 |
| Test, 87 non-demo | 87/87 | 0.1173 | 0.0571 | 0.0316 | 0.0686 | 0.0229 | 21 | 263 | 217 |
| Validation, 28 (chosen settings) | 28/28 | 0.2533 | 0.1206 | 0.1083 | 0.1607 | 0.0916 | 16 | 80 | 83 |

The 92 test pairs hold 254 ground-truth changed objects, and the 87 non-demo
pairs hold 238. Leaving out the demo pairs changes little: the five of them
hold 16 objects, the system matched 1, and they add 20 false positives.

**Compared with validation.** Every headline metric on test is at most about
half of its validation value: pixel IoU 0.1174 against 0.2533, AP per image
0.0602 against 0.1206, and AP per scene 0.0321 against 0.1083. The system
matches 22 of 254 changed objects on test (about 1 in 12) against 16 of 99 on
validation (about 1 in 6). Of the changes it scores, 22 of 305 match an object
on test against 16 of 96 on validation.

Part of the gap is expected, because the settings are the best of the points
tried on validation, so the validation numbers are optimistic. That does not
explain all of it: the test mean of three (0.0699) is below the validation mean
of both runs made before any setting was chosen (0.0855 for the first baseline
and 0.0996 for run 2). Run 2 already holds the code changes made after looking
at validation failures, but the first baseline does not, and it is higher too.
The test pairs may be harder for this system, the gates and the vote count may
help less there, or the later code changes may help validation more than test.
One test run cannot tell these apart, and the test split will not be used to
find out.

What the run records about the two splits (descriptive, not tested as causes):

| | Test, 92 pairs | Validation, 28 pairs |
|-|----------------|----------------------|
| Pairs with at least one matched object | 20 | 12 |
| Pairs with no changed object in the ground truth | 7 (29 of the 283 fp) | 2 (8 of the 80 fp) |
| Changes reported | 839 | 234 |
| Of those, `uncertain` and left out of the scores | 534 (64 %) | 138 (59 %) |
| Pairs without an accepted floor plane (`floor_observed` false) | 20 | 3 |
| Frames per video at stride 30 (median) | 3 to 25 (10) | 4 to 15 (11) |
| Camera height above the floor, m (median) | 0.56 to 4.38 (1.39) | 0.56 to 2.31 (1.44) |

No video reached the cap of 64 keyframes, and one test pair
(`study_5_study_6`) had its AFTER frames cropped to the aspect of the BEFORE
video. `floor_observed` is false when no fitted plane lies at least half way
from the cameras down to the low end of the points, so a level floor is assumed
there; a floor can be in view and still not be accepted. The pooled pixel IoU of
the 20 test pairs without an accepted floor (0.1578, against 0.1071 on the
other 72) is carried by one pair, `laundry_room_1_laundry_room_2` (0.7525).
Without it, the other 19 pool to 0.0592. These counts do not show whether the
floor matters. The two test camera heights above 2 m (4.38 m on
`kitchen_30_kitchen_31` and 2.24 m on `bed_1_bed_2`) are in pairs without an
accepted floor, so they are measured to the assumed floor.

**What this means.** At the shipped settings, the system matches about one
changed object in twelve on the 92 test pairs (about one in eleven on the 87
non-demo pairs), and most of the changes it scores do not match one. The 534
`uncertain` changes are left out of these scores, so they count neither for nor
against it. The 3D reports keep every change with its evidence and mark the
uncertain ones, but under the benchmark's scorer the numbers above are the
measure of detection quality today.

## How to reproduce

Install as in the [README](../README.md). The commands below download the
indoor pairs of a split (`--fetch val` or `--fetch test`) into the data folder
and run them. Use one `--out` folder per set of settings, because a pair run
again with different settings replaces its earlier result. A shared
`--cache-dir` lets the settings that do not touch the reconstruction reuse it.

```bash
# All 28 validation pairs with the shipped defaults.
uv run spatial-change evaluate data/scenediff-val \
    --out runs/val --cache-dir runs/cache-val --fetch val

# One sweep point: --config takes a JSON file with the settings to change.
uv run spatial-change evaluate data/scenediff-val \
    --out runs/val-point --cache-dir runs/cache-val --config point.json
```

A round 2 point is the shipped defaults plus one change. This `point.json` is
the `candidates.link_radius` 1 point of round 2:

```json
{"stages": {"candidates": {"link_radius": 1}}}
```

The defaults changed after the sweep, so a round 1 point needs the old vote
count spelled out. This is the round 1 point for the same parameter:

```json
{"stages": {"visibility": {"min_free_votes": 2},
            "candidates": {"link_radius": 1}}}
```

The gates-off settings of runs 1 and 2 are:

```json
{"stages": {"match": {"max_occluded_fraction": null,
                      "max_box_volume_m3": null},
            "visibility": {"min_free_votes": 2}}}
```

To repeat the rows of the progression table exactly, check out the commit named
there. The ablation rows set the values listed under "What each row changes" in
the same way. `max_keyframes` and `backbone_long_side` go at the top level of
the file, next to `stages`.

The test split is run with the shipped defaults and no config file:

```bash
# All 92 test pairs, once.
uv run spatial-change evaluate data/scenediff-test \
    --out runs/test --cache-dir runs/cache-test --fetch test
```

The 87-pair figure comes from the same data folder and `--out` with
`--summary-only` and `--pairs` naming the 87 non-demo pairs, which pools the
same per-pair results again without running anything. It rewrites
`summary.json` in `--out`, so copy the 92-pair one first.

The synthetic rows with the gates off come from
`uv run pytest tests/evaluation/test_synthetic_suite.py -s`, which prints one
summary line per test. The rows with gates on run the same scenes and seeds with
the match gates set through `stages`; the script for them is not in the
repository.
