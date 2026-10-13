# Spatial Change Detection 3D

Given two phone videos of the same room, recorded at different times, this
project builds one shared 3D representation of both and reports which objects
were added, removed, or moved, and which changes it cannot settle. Each change
comes with a 3D box, a label, evidence counts, and a one-sentence description.
Regions that only one video saw are reported as unobserved, never as change,
so occlusion and unseen areas do not turn into false alarms.

Results and demo media are not published yet.

## How it works

A run goes through 12 stages, in this order. Each stage prints one progress
line when it ends, such as `[4/12] voting: 4.2 s`.
[docs/design.md](docs/design.md) explains the choices and the known failure
modes.

| # | Stage | What it does |
|---|-------|--------------|
| 1 | keyframes | Picks sharp, well-spread frames from each video. |
| 2 | reconstruction | Runs both videos through one Pi3X pass, so every camera shares one world frame. |
| 3 | alignment | Levels the scene on the floor (z-up, meters), then registers the AFTER epoch onto the BEFORE epoch using only structure that did not change. |
| 4 | voting | Projects each point into the other epoch's depth maps. A point counts as changed only if the other epoch saw through it (free space). Occluded or out-of-view points stay unknown. |
| 5 | appearance | Compares DINO patch features, to catch changes that keep their shape. |
| 6 | candidates | Groups changed points into regions, and finds the regions that only one video saw (unobserved). |
| 7 | lifting | Turns each region into a 3D object, using SAM masks from several views. |
| 8 | descriptors | Gives each object an appearance feature. |
| 9 | labeling | Labels each object with SigLIP, and labels furniture to use as landmarks. |
| 10 | matching | Pairs objects that disappeared with objects that appeared. A pair that moved is `moved`; the rest are `removed`, `added`, or `uncertain`. |
| 11 | describing | Writes one sentence per change from its 3D facts, with a local vision-language model or a deterministic template. |
| 12 | outputs | Writes the four result files. |

## Requirements

| Item | Requirement |
|------|-------------|
| OS | Windows or Linux |
| GPU | NVIDIA GPU with 8 GB of VRAM is the target; the defaults are sized for it |
| Python | 3.12 (`requires-python` is `>=3.12,<3.13`) |
| Tools | [uv](https://docs.astral.sh/uv/) and git (the Pi3 code is installed from GitHub) |
| Disk | about 15 GB for model weights, downloaded on first use |
| Optional | `ffprobe` on `PATH`, for exact frame timing with a fixed frame stride (used by `evaluate`) |

Without a GPU, use the `cpu` extra. The describer then uses the template
instead of the vision-language model. Memory and timing measurements are in
[docs/benchmarks/backbones.md](docs/benchmarks/backbones.md); they come from an
RTX 4070 Laptop GPU with 8 GB.

## Install

With an NVIDIA GPU (CUDA 13.0 build of PyTorch):

```bash
uv sync --extra cu130
```

CPU only:

```bash
uv sync --extra cpu
```

On Linux, open3d also needs the EGL, OpenGL, and OpenMP runtime libraries (on
Debian or Ubuntu: `sudo apt-get install libegl1 libgl1 libgomp1`).

Check the install:

```bash
uv run spatial-change --version
uv run pytest -m "not gpu"
```

## Hugging Face access

Two default models are gated on Hugging Face. The other three are open.

| Role | Default (gated) | Open fallback |
|------|-----------------|---------------|
| Patch features | [`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | `facebook/dinov2-base` |
| Masks | [`facebook/sam3`](https://huggingface.co/facebook/sam3) | `facebook/sam2.1-hiera-large` |

To use the defaults:

1. Sign in at huggingface.co and open each model page.
2. Accept the license form. Access is approved manually by the model owner.
3. Authenticate on your machine with `uv run hf auth login`, or set the
   `HF_TOKEN` environment variable to a read token.

Without access, the run still works: when a gated repository cannot be loaded,
the adapter loads the open fallback and continues, and the run prints nothing
about it. The one exception is a run with a stage cache (`--cache-dir`, which
`evaluate` turns on by default): the appearance result computed with the
fallback is not stored, and a `RuntimeWarning` says so. The `model_features`
and `model_segmentation` entries in the `changes.json` metadata name the
first-choice repository either way, so they do not show which one loaded.

The weights are about 15 GB with the defaults (5.4 GB Pi3X, 4.6 GB Qwen,
3.4 GB SAM 3, 1.5 GB SigLIP 2, 0.1 GB DINOv3) and about 13 GB with the
fallbacks. Sizes are the file sizes listed on the Hub. Qwen is downloaded only
when the run uses the vision-language describer.

## Quickstart

```bash
uv run spatial-change run before.mp4 after.mp4 --out runs/demo
```

The output folder must be empty, or pass `--overwrite`. Progress goes to
stderr, one line per finished stage, such as `[4/12] voting: 4.2 s`. At the
end the command prints a summary and the files it wrote, for example:

```text
2 changes (1 added, 1 moved), 3 unobserved regions
wrote runs/demo/changes.json
wrote runs/demo/report.html
wrote runs/demo/scene.rrd
wrote runs/demo/scene.glb
```

Record the two videos as described in the [capture guide](docs/capture-guide.md).

## Outputs

| File | What it is |
|------|------------|
| `changes.json` | The machine-readable report (schema version `"1"`). |
| `report.html` | One self-contained page: summary, a 3D view of both point clouds with the change boxes, and before/after crops with the evidence of each change. It opens offline. |
| `scene.rrd` | A [Rerun](https://rerun.io) recording of both epochs, the change boxes, ghosts of moved objects, the unobserved regions, and keyframe panels. Open it with `uv run rerun runs/demo/scene.rrd`. |
| `scene.glb` | A glTF 2.0 binary with one colored point cloud per epoch, a node per change (its fields are in the node `extras`), and a box per unobserved region. |

All coordinates are in a z-up room frame with the floor at `z = 0`, in meters
(Pi3X predicts approximately metric scale). The origin and horizontal axes are
arbitrary: `+y` is the heading of the first BEFORE camera.

Fields of `changes.json`:

| Field | Meaning |
|-------|---------|
| `changes[].change_id` | Unique id, such as `change-0`. |
| `changes[].type` | `added`, `removed`, `moved`, or `uncertain`. |
| `changes[].label` | Object class from a fixed indoor vocabulary, or `object` when no label scored at least 0.3. |
| `changes[].description` | One sentence about the change. |
| `changes[].confidence` | Number in [0, 1]. |
| `changes[].before_box`, `after_box` | Oriented box (`center`, `extent`, row-major `rotation`), or `null` on the side where the object is absent. |
| `changes[].displacement_m`, `rotation_deg` | Distance between box centers and rotation. Both are set for `moved`. An `uncertain` pair has `displacement_m` only, except that a gate that demotes a `moved` change leaves both. `added` and `removed` changes have neither. |
| `changes[].evidence` | Vote counts `consistent`, `free`, `occluded`, `unobserved` behind the change. |
| `unobserved_regions[]` | `region_id`, `box`, and `epoch_seen`, the epoch that saw the region. |
| `metadata` | Run information: device, seed, frame counts, timings per stage, model repositories, floor and registration quality, peak VRAM on CUDA. |

`uncertain` means the system found a pair but cannot call it a move: the object
is in place with a different look, the registration could not confirm one rigid
object, or an optional gate demoted the change (see the next section). A gate
can also demote an `added` or `removed` change, which then has a single box.

## Configuration

`run` also takes `--out DIR` (required), `--config FILE` (a JSON file with the
settings of the run), `--overwrite`, `--cache-dir DIR` (keeps the reconstruction
and appearance stages, so a rerun on the same videos skips them), and `--debug`
(shows the traceback of a failed run). The settings below are keys of the
config file. Three of them also have a flag, which wins over the file. Unknown
keys are rejected.

| Key | Flag | Default | Meaning |
|-----|------|---------|---------|
| `sample_fps` | | 5.0 | Frames per second sampled before keyframe selection. |
| `max_keyframes` | `--max-keyframes` | 48 | Keyframes kept per video, 2 to 64. |
| `backbone_long_side` | | 518 | Longest frame side given to the backbone, a multiple of 14. |
| `voxel_size` | | 0.02 | Meters; thins each epoch's fused cloud. `null` keeps every point. |
| `describer` | `--describer` | `auto` | `vlm`, `template`, or `auto`, which uses the vision-language model on CUDA and the template otherwise. |
| `device` | `--device` | `auto` | Torch device, such as `cuda` or `cpu`. `auto` is CUDA if available. |
| `frame_stride` | | `null` | Use every Nth slot of a 30 fps timeline instead of keyframes (`evaluate` sets it). |
| `seed` | | 0 | Seed of every random search and subsample. |
| `stages` | | `{}` | Overrides of any field of any stage config. |

`stages` maps a stage name to the fields to change. The stage names are
`floor`, `refine`, `visibility`, `appearance`, `candidates`, `unobserved`,
`lift`, `descriptors`, `label`, `landmarks`, `match`, `relation`, and `rerun`.
An object on a field that holds a config changes fields of that nested config,
and `null` sets an optional field back to none. `visibility.device` cannot be
set; the run resolves it from `device`. Every override is checked before the
run starts. For example:

```json
{
  "describer": "template",
  "max_keyframes": 32,
  "stages": {
    "appearance": { "changed_below": 0.3 },
    "lift": { "views": { "max_views": 7 } },
    "match": {
      "max_box_volume_m3": 2.0,
      "max_occluded_fraction": 0.6,
      "min_change_confidence": 0.2
    }
  }
}
```

The three `match` gates are off by default. Each demotes a change to
`uncertain` (it keeps its boxes, label, and motion, and its confidence is capped at 0.5)
when its largest box is too big, too many of its votes were occluded, or its
confidence is too low. Field names and defaults of every stage are in the
`*Config` dataclasses under `src/spatial_change/`.

## Evaluation

`evaluate` runs the system on [SceneDiff](https://github.com/yuqunw/scene_diff)
pairs, scores each against its annotations, and pools the scores into
`summary.json` in the results folder.

```bash
# Download the indoor pairs of the validation split, run them, and score them.
uv run spatial-change evaluate data/scenediff --out runs/eval --fetch val

# Run named pairs that are already in data/scenediff.
uv run spatial-change evaluate data/scenediff --out runs/eval \
    --pairs living_room_3_living_room_4

# Rebuild summary.json from the results on disk without running anything.
uv run spatial-change evaluate data/scenediff --out runs/eval --summary-only
```

`data_dir` holds one folder per pair, each with `segments.pkl` and the two
videos. `--fetch val` or `--fetch test` downloads only the indoor pairs of that
split, reading the members it needs from the benchmark archive with HTTP Range
requests. Running the same command again resumes after a crash: finished pairs
with the same settings are reused. A pair that fails is recorded and the batch
goes on, and the next run tries that pair again. The command prints
pixel IoU and the three average-precision scores, for all pairs (failed ones
scored as missed) and for the pairs that finished. Other flags:

| Flag | Meaning |
|------|---------|
| `--pairs NAME...` | Pairs to run. Default: every subfolder with `segments.pkl`. Cannot be combined with `--fetch`. |
| `--describer` | Descriptions are not scored, so `evaluate` uses `template` unless this flag or the config file sets `vlm` or `template` (`auto` becomes `template`). |
| `--stride N` | Run and score the frames of the 30 fps annotation timeline whose index is a multiple of `N`. Default: 30. |
| `--uncertain-as {drop,moved}` | Leave uncertain changes out of the scores (default), or score them as drawn. |
| `--summary-only` | Rebuild `summary.json` and run nothing. Cannot be combined with `--fetch`. |

`--config`, `--device`, `--cache-dir`, and `--debug` work as for `run`; the
cache defaults to `cache` inside `--out`.

## Web app

`app` serves a page on this computer where you upload the two videos and read
the report, a table of the changes, the 3D scene, and the four files described
under Outputs. It needs the `app` extra, which you add to the install command
of your PyTorch build:

```bash
uv sync --extra cu130 --extra app   # CPU only: --extra cpu --extra app
uv run spatial-change app
```

Open the address it prints. It listens on `127.0.0.1` only, runs one job at a
time, takes MP4, MOV, or M4V videos up to 1 GB each, and keeps them on this
computer. To reach it from another machine, tunnel the port over SSH, for
example `ssh -L 7860:127.0.0.1:7860 user@host`. Each run writes into its own
folder under `.cache/spatial-change/app` in your home folder, next to the stage
`cache`, and a run that fails leaves no folder behind; pass `--runs-dir` to
keep the runs elsewhere. `--config`, `--describer`, `--device`, and
`--max-keyframes` work as for `run`, and `--port` picks the port (default
7860, or the next free one).

## Third-party licenses

This repository's code is MIT, but the model weights it downloads are not.

- The default reconstruction weights, Pi3X, are **CC BY-NC 4.0: non-commercial
  use only**.
- The default DINOv3 and SAM 3 weights are under **custom Meta licenses** (the
  DINOv3 License and the SAM License), not a standard open-source license. Read
  their terms before redistributing or using the project commercially.
- The remaining weights and the code dependencies are under the licenses listed
  below.

### Model weights

| Role | Repository | License | Access |
|------|------------|---------|--------|
| Reconstruction | `yyfz233/Pi3X` | CC BY-NC 4.0, non-commercial | Open |
| Patch features | `facebook/dinov3-vits16-pretrain-lvd1689m` | DINOv3 License (custom Meta license) | Gated, manual approval |
| Patch features, fallback | `facebook/dinov2-base` | Apache-2.0 | Open |
| Masks | `facebook/sam3` | SAM License (custom Meta license) | Gated, manual approval |
| Masks, fallback | `facebook/sam2.1-hiera-large` | Apache-2.0 | Open |
| Labels | `google/siglip2-base-patch16-224` | Apache-2.0 | Open |
| Labels, fallback | `google/siglip-base-patch16-224` | Apache-2.0 | Open |
| Descriptions | `Qwen/Qwen3.5-2B` | Apache-2.0 | Open |
| Evaluation data | `yuqun/SceneDiff` (dataset) | MIT | Open |

Checked on 2026-10-13 against: the license and gating fields of
`https://huggingface.co/api/models/<repository>` (and of
`https://huggingface.co/api/datasets/yuqun/SceneDiff`); the `LICENSE.md` of the
DINOv3 repository; the `LICENSE` of `github.com/facebookresearch/sam3` (the SAM
License text); and the Pi3 README at the pinned commit, which says the Pi3 and
Pi3X weights are CC BY-NC 4.0 and the code is BSD 3-Clause. The two Meta
licenses are custom texts: read them before redistributing.

### Code dependencies

Runtime dependencies at the versions in `uv.lock`: the direct dependencies of
the project, `torch` and `torchvision` (installed by the `cpu` or `cu130`
extra), and `safetensors`, which `pi3` and `transformers` pull in and which
reads the model weights. Licenses are from the PyPI metadata of that version,
except `pi3`.

| Package | License |
|---------|---------|
| huggingface-hub 2.2.0 | Apache-2.0 |
| numpy 2.5.3 | BSD-3-Clause (the PyPI expression adds 0BSD, MIT, Zlib, CC0-1.0 for bundled parts) |
| open3d 0.20.0 | MIT |
| opencv-python-headless 5.0.0.93 | Apache-2.0 (OpenCV); the wheels ship FFmpeg under LGPL-2.1 |
| pi3 (git, pinned commit) | BSD-3-Clause (code only; the weights are in the table above) |
| pillow 12.3.0 | MIT-CMU |
| pydantic 2.14.0 | MIT |
| rerun-sdk 0.38.1 | MIT OR Apache-2.0 |
| transformers 5.19.0 | Apache-2.0 |
| torch 2.14.1 | BSD-3-Clause (the PyPI expression adds Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSL-1.0, MIT for bundled parts) |
| torchvision 0.29.1 | BSD (as listed on PyPI) |
| safetensors 0.8.0 (via pi3 and transformers) | Apache-2.0 |

The SceneDiff metrics in `src/spatial_change/evaluation` follow the reference
implementation; its MIT notice is in `src/spatial_change/evaluation/NOTICE`.

## Troubleshooting

Failures print `error: <message>` and exit with code 1. Invalid arguments or
settings exit with code 2. Add `--debug` for the traceback.

| Message | Cause and fix |
|---------|---------------|
| `error: <dir> is not empty; pass --overwrite to write into it` | The output folder holds files. Use a new folder or add `--overwrite`. |
| `error: video not found: <path>` | A video path is wrong. |
| `error: OpenCV cannot open video: <path>` or `error: no decodable frames in video: <path>` | OpenCV could not open or decode the file. Check that it plays, and re-export it in a common format such as H.264 MP4 if it does. |
| `error: no sharp frame in video: <path>` | Sampling the video gave no frame to choose from. Check that the file plays to the end, or re-export it. |
| `invalid configuration: ...` | The config file is missing, is not a JSON object, or a key or value is not valid. The message names the field, for example `invalid configuration: stages: Value error, appearance.changed_below: expected float, got str`. |
| `error: could not load facebook/sam3 or the fallback facebook/sam2.1-hiera-large` (the DINO and SigLIP models give the same message with their own repositories) | Neither repository could be loaded, usually because the machine is offline and the weights are not cached. Run once with network access. Pi3X and Qwen have no fallback: when one fails to load, the error is the Hugging Face message itself, and the remedy is the same. |
| `error: CUDA out of memory. ...` | The reconstruction attends over all frames at once. Lower `--max-keyframes`, and close other programs that use the GPU: the run limits itself to the VRAM that is free when the model loads. |
| `error: median camera height above the low end of the points must be positive; the up direction is wrong`, or `error: median camera height is far below a fitted floor; the up direction is wrong` | The up direction estimate failed, for example when the cameras look up for most of the capture or only the ceiling is in view. Keep the camera level or pitched down. |
| `error: no confident points to align` | No pixel of either video passed the confidence threshold of the reconstruction. Try other footage. |
| `error: fewer than 3 usable pairs (...)` | The registration of the AFTER epoch onto the BEFORE epoch found almost no matching points (by default within 0.25 m). Record both videos over the same area. |
| `error: the web app needs the app extra; add --extra app to your uv sync command` | `app` was started without the `app` extra. Run `uv sync --extra cu130 --extra app`, or `--extra cpu --extra app`. |
| `error: port <n> is in use; pick another with --port` or `error: no free port was found; pick one with --port` | The port given with `--port` is in use, or none of the ports that `app` tries was free. Pick a free port with `--port`, or leave `--port` out to take the next free port from 7860. |
| `RuntimeWarning: stage cache: could not <read, decode, or write> the <stage> entry (...); continuing without it` | The message names one of the three. `read` or `decode`: a cache entry was unreadable or damaged, and the run recomputes the stage. `write`: the stage ran, but its result could not be stored, so the next run computes it again. |
| `RuntimeWarning: stage cache: the appearance came from '<fallback>', not '<model>', so it was not stored` | The gated DINOv3 weights could not be loaded, so the open fallback encoded the frames. The run is fine. The result is not cached, so a later run with access does not get it. See Hugging Face access. |

For `evaluate`, `error: data folder not found: <dir>` means `data_dir` does not
exist, `error: no pairs found in <dir> (a pair is a folder with segments.pkl)`
means it holds no pairs (use `--fetch` or `--pairs`), and
`interrupted: rerun the same command to resume` follows Ctrl+C (exit code 130).

## License

MIT. See [LICENSE](LICENSE).
