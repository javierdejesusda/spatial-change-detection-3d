# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-14

First release.

### Added

#### Pipeline and outputs

- A 12-stage pipeline that takes a BEFORE and an AFTER phone video of the same
  room and reports `added`, `removed`, `moved`, and `uncertain` changes. Each
  change has a 3D box, a label, evidence counts, and a one-sentence
  description. The stages are listed in the README.
- Regions that only one video saw are reported as unobserved, never as change.
- Descriptions from a local vision-language model (Qwen3.5-2B), checked against
  the 3D facts, or from a deterministic template. The template is the default
  on CPU.
- Gates that demote a change to `uncertain`: box volume and occluded share are
  on by default, and a minimum confidence is off. One free vote per changed
  point is required by default.
- Open fallbacks for the two gated models (DINOv3 and SAM 3), used when access
  is missing.
- A stage cache for the reconstruction and appearance stages, and run
  settings from a JSON config file (unknown keys are rejected).
- Four outputs per run: `changes.json`, `report.html` (one offline page),
  `scene.rrd` (a Rerun recording), and `scene.glb`.

#### Command line and web app

- `spatial-change run` detects the changes between two videos.
- `spatial-change evaluate` runs and scores SceneDiff pairs, and resumes after
  a crash.
- `spatial-change fetch-demo` downloads the five demo pairs and can run them.
- `spatial-change app` opens a local web app, from the `app` extra, to upload
  two videos and read the report. It listens on `127.0.0.1` only.

#### Evaluation

- A reimplementation of the SceneDiff metrics (pixel IoU and three kinds of
  average precision), and a resumable batch runner.
- `evaluate --fetch val` and `--fetch test` download only the indoor pairs of a
  split.
- [docs/results.md](docs/results.md): settings chosen on the 28 validation
  pairs, then one run on the 92 test pairs with the settings frozen. Pixel IoU
  is 0.1174 and AP per scene is 0.0321. The page also has the ablations and a
  synthetic suite with exact ground truth.

#### Input handling

- Only ISO base media files, which is what phones record as MP4 or MOV, are
  read, judged by the start of the file and not by its name. Other containers
  fail with `error: not an MP4 or MOV video`.
- A fixed-stride run refuses a video whose timeline is longer than one hour.
- A run that fails leaves no outputs that look like its own, and `--overwrite`
  with a missing video keeps the earlier results.
- `--port` values outside 1 to 65535 are refused.

### Known limitations

- Accuracy is low. On the 92 test pairs the system matches 22 of 254 changed
  objects (about 1 in 12) and reports 283 unmatched changes among the scored
  ones. Test scores are at most about half of the validation scores.
- The five demo pairs are test pairs that were used during development. The 87
  other test pairs score pixel IoU 0.1173 and AP per scene 0.0316.
- 534 of the 839 changes reported on the test pairs (64 %) are `uncertain`, and
  the scores leave them out. A capture with much occlusion may report many
  changes as `uncertain`.
- A moved object can be split in two and reported as a `removed` and an `added`
  change, or as a `moved` change that boxes only part of it.
- Two adjacent changed objects can merge into one change with one oversized box.
- Close-up footage can give false changes and boxes of the wrong size, because
  no stage corrects the depth scale.
- A capture that looks up for most of its length can give an upside-down room
  frame or fail with `the up direction is wrong`. A capture of the ceiling
  alone fails at alignment.
- Labels are weak for plain shapes: the label is `object` or a wrong noun.
- Objects hidden in both videos are invisible, and lighting changes between the
  recordings can create false changes. A run keeps 48 keyframes per video by
  default and at most 64.

[0.1.0]: https://github.com/javierdejesusda/spatial-change-detection-3d/releases/tag/v0.1.0
