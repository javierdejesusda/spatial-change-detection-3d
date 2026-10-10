# Backbone and describer benchmarks (8 GB laptop GPU)

This page records the measurements behind three pipeline defaults: which
feed-forward 3D backbone reconstructs the two videos, how many frames go into
one reconstruction pass, and whether the reconstruction models and the
vision-language describer share one Python environment. All numbers come from
one SceneDiff pair (`living_room_3_living_room_4`) on a single laptop, so read
them as a feasibility and ranking check, not as a benchmark of the models.

## Decisions

| Question | Decision | Main evidence |
|----------|----------|---------------|
| Default 3D backbone | **Pi3X** with the unused multimodal prior branch removed, trunk and decoders in bf16, conv heads run 8 frames at a time | Best pose against COLMAP (11.9 to 12.7 % ATE, 4.4 to 5.2 deg mean rotation error), the only variant that fits 128 frames on 8 GB, and the weights are not gated |
| Frames per pass | Both videos go through **one joint pass** so they share a coordinate frame. Default **64 frames** (32 per video) at 518 x 294 (height x width); ceiling **128 frames** (64 per video). 672 x 378 is a quality option up to 64 frames when VRAM allows | 64 frames take 13.2 s at 5,080 MiB reserved, about 2 GB below the 7,054 MiB that was free; 128 take 39.2 s at 6,576 MiB. 672 x 378 improves ATE by under 1 point but needs 6,628 MiB at 64 frames |
| Environments | **One venv** for reconstruction and the describer | The default backbone package declares only `safetensors`, and `transformers` 5.19 installs next to it without changing any version |
| Describer | **Qwen3.5-2B** in bf16 on the GPU, loaded after the 3D models are unloaded | About 1.7 s per change for a before/after crop pair, 4,466 MiB peak reserved |

Fallbacks:

- **VGGT-Omega** has pose accuracy similar to Pi3X but fits at most 96 frames,
  and its weights are gated and non-commercial. It stays an optional adapter.
- **Depth Anything 3 (DA3-BASE)** is the fastest model and its weights are
  Apache-2.0, but its poses were poor here (see the open item below). It stays
  the permissive fallback until that is resolved.
- On a machine without a GPU, Qwen3-VL-4B (Q4_K_M) through llama.cpp describes
  one crop in about 8 s on 8 CPU threads.

## Setup

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4070 Laptop, 8,188 MiB, 7,054 MiB free after the desktop and CUDA context; driver 581.57 (WDDM) |
| Software | Python 3.12, torch 2.14.1+cu130, numpy 2.5.2, transformers 5.19.0 (describer only) |
| Data | SceneDiff SD-V pair `living_room_3_living_room_4`: video 1 has 608 frames, video 2 has 459, both 720 x 1280 portrait |
| Frame choice | Half the frames from each video, evenly spaced |

Every run caps the torch allocator at the VRAM that was free at start
(`torch.cuda.set_per_process_memory_fraction`). Without the cap, the Windows
WDDM driver silently spills to system RAM instead of raising an out-of-memory
error: reserved memory reached 22 GB on the 8 GB card, and 128 Pi3X frames took
113.8 s instead of 39.2 s. With the cap, "fits" means the run finished without
`torch.OutOfMemoryError`.

Timing wraps the forward pass with `torch.cuda.synchronize()` on both sides,
after a 2-frame warm-up. Frame decoding, resizing, and the copy to the GPU are
outside the timer, except for DA3, where the timed `inference()` call includes
its CPU preprocessing. Memory is `torch.cuda.max_memory_allocated` and
`max_memory_reserved` in MiB.

## Memory and time against frame count

Largest frame count N that fits under the cap (tested N: 8, 16, 24, 32, 48, 64,
96, 128). Resolutions are height x width.

| Variant | Resolution | Weights (MiB) | Max N | Peak allocated (MiB) | Peak reserved (MiB) | Seconds at max N | First OOM |
|---------|------------|---------------|-------|----------------------|---------------------|------------------|-----------|
| DA3-BASE, bf16 autocast | 504 x 280 | 518 | 128 | 6,302 | 7,006 | 7.43 | none |
| VGGT-Omega, fp32 weights | 512 x 288 | 4,365 | 48 | 6,479 | 6,970 | 5.49 | 64 |
| VGGT-Omega, bf16 aggregator | 512 x 288 | 2,633 | 96 | 6,837 | 7,016 | 14.69 | 128 |
| Pi3X as published | 518 x 294 | 5,192 | 8 | 6,158 | 6,928 | 1.05 | 16 |
| Pi3X, multimodal branch removed | 518 x 294 | 3,790 | 16 | 5,715 | 6,870 | 2.32 | 24 |
| ... plus heads in chunks of 8 frames | 518 x 294 | 3,790 | 64 | 6,006 | 6,710 | 13.91 | 96 |
| ... plus bf16 trunk | 518 x 294 | 1,933 | 32 | 5,774 | 6,594 | 5.20 | 48 |
| **... plus bf16 trunk and chunked heads (default)** | 518 x 294 | 1,933 | **128** | 5,930 | 6,576 | 39.16 | none |
| Same, repository default resolution | 672 x 378 | 1,933 | 64 | 5,615 | 6,628 | 29.90 | 96 |

The default Pi3X variant across N:

| N | 8 | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|----|----|----|----|----|----|-----|
| Seconds | 0.98 | 2.19 | 3.57 | 5.12 | 8.62 | 13.20 | 24.28 | 39.16 |
| Peak reserved (MiB) | 4,838 | 4,880 | 4,964 | 4,652 | 5,128 | 5,080 | 6,020 | 6,576 |

Notes:

- Pi3X ships a multimodal prior branch that the image-only pipeline never
  uses. Removing it frees 1,402 MiB of weights. Casting the trunk and decoders to
  bf16 halves the rest. Running the convolutional heads on 8 frames at a time
  caps the activation peak. Together these move Pi3X from 8 to 128 frames.
- The Windows torch build has no flash-attention kernel. Pi3X needs the
  memory-efficient SDPA kernel to reach these numbers.
- Time grows faster than linearly with N (attention across all frames).
  64 frames cost a third of the time of 128.

## Pose accuracy against COLMAP

**Reference.** COLMAP (pycolmap 4.2.1, CPU) ran on 86 frames per video: about
80 evenly spaced frames plus the 8 benchmark frames. With 12,000 SIFT features,
a 0.9 ratio test, and at least 8 inliers, it registered all 172 images in one
model: 70,974 points, 0.398 px mean reprojection error, mean track length 6.69.
An 8,192-feature run split into two models, which is why the larger setting is
used. COLMAP has no metric scale, so it is a pseudo ground truth for shape, not
for scale.

**Method.** Each backbone predicts camera poses for the same 16 frames (8 per
video) in one pass. Its camera centres are aligned to COLMAP's with a
similarity transform (Umeyama). ATE is the RMS error of the aligned centres as
a percentage of the RMS extent of the COLMAP trajectory (3.44 COLMAP units).
Rotation error is the geodesic angle per frame after the alignment rotation.

| Backbone | ATE (%) | ATE video 1 / 2 (%) | Rotation mean / median / max (deg) |
|----------|---------|---------------------|------------------------------------|
| Pi3X default variant at 672 x 378 | **11.87** | 12.60 / 11.09 | **4.43** / 4.43 / 5.27 |
| **Pi3X default variant at 518 x 294** | 12.66 | 13.18 / 12.12 | 5.15 / 5.08 / 6.89 |
| Pi3X, multimodal branch removed, fp32 (518 x 294) | 12.54 | 12.79 / 12.28 | 5.20 / 5.15 / 6.92 |
| VGGT-Omega, fp32 weights | 12.26 | 13.41 / 10.99 | 5.81 / 5.67 / 8.84 |
| VGGT-Omega, bf16 aggregator | 12.81 | 14.08 / 11.41 | 5.60 / 5.44 / 8.40 |
| DA3-BASE | 64.88 | 69.85 / 59.49 | 22.20 / 20.39 / 72.50 |

Readings:

- Pi3X and VGGT-Omega agree within about 1 point of ATE. bf16 costs Pi3X
  nothing measurable. The higher 672 x 378 resolution helps a little.
- Chunking the heads does not change the poses; it changes only memory. The
  default variant at 518 x 294 therefore has the same poses as the unchunked
  bf16 run.
- DA3's poses are far off. Inverting its extrinsics or flipping to the OpenGL
  convention makes them worse (rotation errors of 166 and 174 deg), so the
  convention was read correctly. A preprocessing cause (resolution or aspect
  handling) is not ruled out yet.

## Describer: local vision-language model

Both runs use the same crops, frames, and prompts:

- 3 crops of 400 x 400 from a video 1 frame, each with the prompt "Describe the
  main object in this image in one short sentence: its category, color, and
  material."
- Full 720 x 1280 frames (2 on the CPU, 4 on the GPU), each with the prompt "List up to five distinct
  objects visible in this room, as a comma-separated list."
- Greedy decoding, at most 96 new tokens. Every input runs twice: a cold call,
  then a warm call. The warm call is reported.

| Model | Runtime | Hardware | Crop (s) | Full frame (s) | Memory |
|-------|---------|----------|----------|----------------|--------|
| Qwen3.5-2B, bf16 | transformers 5.19, SDPA | RTX 4070 Laptop | 0.38 to 0.42 | 0.77 to 0.99 | 4,466 MiB peak reserved (4,222 MiB weights) |
| Qwen3-VL-4B, Q4_K_M | llama.cpp b11500, 8 threads | Ryzen 7 7730U, 8 cores | 7.7 to 8.1 | about 42 | 5.6 to 6.3 GB RSS |

Two extra GPU measurements, closer to how the describer is used:

- **Before/after pair.** Two crops in one message, with the prompt asking which
  object changed: 1.66 s warm, 342 prompt tokens, 39 generated tokens. This is
  the planned cost of one change description.
- **Batch of 3 crops.** One `generate` call for 3 single-crop conversations:
  0.48 s in total, 0.16 s per crop.

Notes:

- The model loads in about 3 s from a warm disk cache. At 4.4 GiB it cannot
  share the card with a reconstruction pass, so the pipeline loads it only
  after the 3D models are unloaded.
- The Gated DeltaNet layers ran on the reference PyTorch path, because
  `flash-linear-attention` and `causal-conv1d` were not installed. The
  timings above are without those kernels.
- **Greedy decoding can loop.** On one full frame the model repeated
  "magazine," until it hit the 96-token limit (4.2 s). The describer needs a
  repetition penalty and a short token limit. The model card recommends
  sampling with `presence_penalty=1.5`; greedy plus a repetition penalty keeps
  the output reproducible.
- With the "one short sentence" prompt the 2B model answers with terse lists
  ("orange envelope, paper, plastic"). The before/after prompt gives a full
  sentence. Output quality is judged later on the demo pairs, not here.

## Environments

| Package | Declared dependencies that matter |
|---------|-----------------------------------|
| `pi3` (Pi3X) | `safetensors` only |
| `transformers` 5.19 | Installs next to the reconstruction stack without changing any version (dry run: additions only) |
| `vggt-omega` | `numpy<2`, `opencv-python` |
| `depth-anything-3` | `numpy<2`, `opencv-python`, `xformers`, `open3d`, `fastapi`, `uvicorn`, and more |

The conflicts come only from the two optional backbones, and both ran under
numpy 2 with `opencv-python-headless` in the benchmark. One venv with
dependency overrides for those two is simpler than a separate reconstruction
environment, which would add a process boundary and serialize every result for
no current benefit. The models run one after another and are unloaded between
stages, so one process also keeps the GPU handoff simple.

## Licenses of the weights

| Model | License | Gated |
|-------|---------|-------|
| Pi3X | CC BY-NC 4.0 (non-commercial) | no |
| Pi3 | BSD-2-Clause | no |
| VGGT-Omega | FAIR Noncommercial Research License | yes, manual approval |
| DA3-BASE | Apache-2.0 | no |
| Qwen3.5-2B | Apache-2.0 | no |

The repository code is MIT. The default backbone's weights are non-commercial,
so the README license table flags them. A fully permissive setup needs DA3-BASE
(once its pose issue is resolved) or Pi3, which has not been measured here.

## Open items

- [ ] Check DA3 preprocessing (input resolution, aspect handling, autocast)
      before ruling out its poses.
- [ ] Measure whether 128 frames per pass improve change detection over 64.
      That question needs the SceneDiff evaluation, not pose error on 16 frames.
- [ ] Repeat the pose comparison on more pairs. One scene and 16 frames give a
      ranking, not a confidence interval.
