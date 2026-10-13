# Design

This page explains why the system is built the way it is, what each choice
costs, and where it is known to fail. For install and usage see the
[README](../README.md). The evidence behind the model choices is in the
[research review](research/sota-review-2026-10.md) and the
[backbone benchmarks](benchmarks/backbones.md).

## The idea

Finding a change and explaining it are separate jobs, and each goes to the tool
that is reliable at it. The research review found vision-language models weak
at before/after comparison (a bias toward "no difference", weak reasoning
about viewpoint), so they never decide what changed.

| Role | Cue | How |
|-----|-----|-----|
| Propose | Geometry | Both videos go through one 3D reconstruction, so they share a coordinate frame. |
| Veto | Visibility | Each surface point is tested against the other video's depth maps. A surface that the other video could not see is never called changed: only free-space evidence, or the appearance cue below at a surface both videos saw, marks a point changed. |
| Confirm | Appearance | Feature similarity at surfaces both videos saw catches changes that keep their shape. |
| Explain | Language | A label (from image crops) and a sentence (from the 3D facts of each change), and the sentence is checked against the facts. |

Everything runs locally and uses no paid service. Models are open weights,
loaded one at a time on a single 8 GB GPU.

## Choices

### One joint reconstruction pass

Keyframes of both videos go through a single Pi3X forward pass (up to 64 per
video, 128 in all; 48 per video by default). Every camera lands in one world
frame, so there is no registration of two separate reconstructions to get
wrong.

- **Why Pi3X.** It is the backbone of the closest published system (SceneDiff),
  whose ablation drops 8.7 points of per-scene object AP when Pi3 is swapped
  for VGGT. Against a COLMAP reconstruction, its pose error is within about a
  point of VGGT-Omega (12.7 % against 12.3 to 12.8 % ATE) and far below
  Depth Anything 3 Base (64.9 %). Its weights are not gated.
- **How it fits 8 GB.** The adapter removes the multimodal branch that an
  image-only pipeline never uses (1,402 MiB), runs the trunk and decoders in
  bf16, and runs the convolutional heads 8 frames at a time. 128 frames then
  take 39.2 s and 6,576 MiB reserved on an RTX 4070 Laptop GPU. These numbers
  come from one SceneDiff pair, so read them as a feasibility check.
- **Allocator cap.** On Windows the driver spills to system RAM instead of
  raising an out-of-memory error: in the benchmark, 128 frames took 113.8 s
  instead of 39.2 s. The adapter caps the torch allocator at the VRAM that is
  free when the model loads, so a too-large input fails fast.
- **Cost.** The weights are CC BY-NC 4.0. The frame budget is the whole
  budget: quality depends on which keyframes are chosen. The scale is whatever
  Pi3X predicts (the scale policy is `model`), and no later stage corrects it.

### Keyframes

Each video is sampled at about 5 fps. Frames in the blurriest quarter, or below half
the median sharpness, are dropped. A greedy pass keeps a frame when its overlap
with the last keyframe (phase correlation on 160 px thumbnails) falls below
0.7, and the result is thinned evenly to the cap. If the AFTER video has a
different aspect ratio, its frames are center-cropped to the BEFORE ratio,
because the backbone needs one frame size. The crop is recorded in the metadata
(`after_cropped_to_before_aspect`).

### Gravity-aligned room frame and registration

- **Up** comes from the camera x axes, not the mean camera up. A camera pitched
  down tilts the mean up, but its x axis stays horizontal, so up is the
  direction the x axes are most nearly perpendicular to. The sign is chosen by
  a vote over the cameras (see the failure mode for looking up).
- **Floor** is the lowest well-supported horizontal plane from a seeded RANSAC
  (at least 5 % of the points), so a table top or a ceiling never beats a
  smaller visible floor. If no plane lies at least half way from the cameras
  down to the low end of the points, a level floor is assumed there and
  `floor_observed` is false in the metadata.
- **One alignment for both epochs.** It is fitted once on the union of both
  clouds and cameras. Separate fits would give the epochs different headings,
  and the registration that follows fails silently once they differ by about
  10 degrees.
- **Change-aware registration.** The AFTER cloud is registered onto the BEFORE
  cloud with trimmed ICP. After each round, every point more than 5 cm from the
  other cloud is excluded as a possible change, for up to 5 rounds, so a moved
  sofa does not pull the registration toward itself. The transform is rigid.

### Visibility vote

Every point is projected into every frame of the other epoch and compared with
that frame's depth map. The tolerance is `(0.05 m + 0.02 * depth)`, widened for
low-confidence pixels.

| Vote | Meaning |
|------|---------|
| consistent | The frame's depth matches the point's depth. |
| free | The frame sees through the point to farther surfaces. All pixels of the 3 x 3 window around it must be valid and farther, so a point on a depth edge does not see through its own silhouette. |
| occluded | A nearer surface blocks the view. |
| unobserved | The point is outside the frame, or its depth is unusable. |

A point is `changed` with at least one free vote (`visibility.min_free_votes`)
and more free than consistent ones. Otherwise it is `present` with one
consistent vote, and else `unknown`.
This is the veto: occluded and unobserved votes can never make a point changed,
so an object that was hidden, or never seen, is not reported as removed or
added. A switch (`visibility.veto`) turns the veto off for ablations; it floods
the report with hidden objects and is not meant for runs.

### Appearance cue

Geometry cannot see a surface that kept its shape and changed its look. Points
are binned into 5 cm voxels, and every frame is encoded once into DINO patch
features. A voxel gathers the feature at its projection in each frame that sees
it, in its own epoch and in the other one, and the cosine between the two means
is its similarity. `present` points below 0.4 become `changed`. A cached stage
lets you sweep that threshold without redoing the encoding (`--cache-dir`).

### Change-first lifting

Instead of segmenting the whole scene in 3D, the system lifts only the changed
regions. The review found that 3D instance segmentation methods assume sensor
depth and clean scans, and none was tested on reconstructed geometry.

1. The `changed` points of each epoch are binned into 5 cm voxels and linked
   across gaps of one empty voxel. A region needs at least 10 points, 3 voxels,
   and 30 % changed points among the changed and present points in its voxels,
   which rejects speckles on a confirmed surface.
2. Each region is projected into the up to 5 frames of its epoch that see it
   best. A box and a point become the prompt of SAM, one mask per view.
3. Masks are lifted through the frame depth. A voxel stays only if at least half
   of the views that see it include it, and only if it links to the region's own
   points. Lifted points within 2 cm of the floor are dropped unless they touch
   the region.
4. Fragments whose masks cover each other's points are merged into one object.

### Association

An object that moved is lifted twice, once where it vanished (BEFORE) and once
where it appeared (AFTER). The cost of a pair is the cosine distance of the DINO
features, plus the misfit of registering the smaller cloud onto the larger one
(FPFH features with RANSAC, then trimmed ICP from several yaw starts), plus a
weak distance term. A minimum-cost assignment pairs each object with at most one
partner. A pair is refused if its extents differ by more than 3 times, its
registration fits worse than 0.85, or its cost reaches 0.9.

- A pair whose disappeared box travels 5 cm or turns 10 degrees under the
  registration is `moved`. Any other pair is `uncertain`: it is in place with a
  new look, or the registration cannot vouch for one rigid object.
- Objects left without a partner are `removed` or `added`.
- Confidence combines the registration fit or the changed fraction with a
  support factor, `1 - exp(-n / 500)`, for the `n` observations that saw the
  location, so a change seen by few frames scores low. An abstention is capped
  at 0.5.
- Three gates (`match.max_occluded_fraction`, `match.max_box_volume_m3`,
  `match.min_change_confidence`) demote a change to `uncertain` after the pairs
  are made. The first two are on by default, with the values chosen on the
  validation split, and the confidence gate is off. A gate set to `null` is off.

### Labels, landmarks, and language

- **Labels.** SigLIP 2 scores up to 5 crops of each object against a fixed
  vocabulary of 48 indoor nouns. The label is the best mean softmax share. Below
  0.3 there is no label, and the report says `object`.
- **Landmarks.** Static furniture gives the description its context. The points
  both epochs agree on lose the floor, the ceiling, and vertical walls, and a
  cluster whose label is a furniture name becomes a landmark.
- **Facts.** For each change, geometry computes the supporting surface, the
  nearest landmarks, the direction relative to the first BEFORE camera, the
  distance, and the rotation.
- **Sentence.** With the vision-language model (Qwen3.5-2B, bf16), the prompt is
  the facts as text plus one before and one after crop, and the reply must be a
  JSON object. A rule-based validator rejects a sentence whose change type,
  direction, or numbers contradict the facts (tolerances 0.10 m, 15 degrees, or
  50 % relative). A reply that cannot be parsed is retried once with a
  correction added to the prompt. A reply that parses but fails the validator
  is not retried: the deterministic template is used at once, as it is after a
  second unusable reply. Uncertain changes always use the template. The model
  verbalizes evidence and never discovers it.

### Models, memory, and caching

The core logic talks to models through small ports (reconstructor, feature
encoder, mask segmenter, labeler, vision-language model), so tests use fakes and
adapters are swappable. Each model is built when its stage starts and unloaded
when it ends, so at most one is on the GPU. Qwen, at 4.4 GiB, loads only after
the others are gone. The reconstruction and the appearance comparison, the two
long stages, can be cached on disk (`--cache-dir`). A key holds a fingerprint of
the input frames or aligned reconstructions, the model id and settings, and the
package version.

### Outputs

`changes.json` is validated by a strict schema (unknown fields are rejected, and
box and type fields must agree), so a parsed report is exactly what the schema
says. `report.html` is one offline file for people who just want to look.
`scene.rrd` is for stepping through both epochs in Rerun. `scene.glb` is the
annotated point cloud for any glTF viewer.

## Alternatives not used

| Option | Reason |
|--------|--------|
| VGGT-Omega | Similar pose accuracy, but gated, non-commercial weights and at most 96 frames on 8 GB. |
| Depth Anything 3 Base | Apache-2.0 weights and fast, but its poses were far off in the benchmark. The cause is not resolved. |
| COLMAP | Slow. It served as pseudo ground truth for the pose benchmark. |
| 3D Gaussian Splatting change detectors | They need a 3DGS reconstruction per epoch, most are unreleased, and they have no added, removed, or moved types. |
| A vision-language model as detector | Biased toward "no difference" on before/after pairs. |
| YOLOE, YOLO-World | AGPL or GPL licenses. |
| Full-scene 3D instance segmentation | Built for clean RGB-D scans; not tested on reconstructed geometry. |

## Accepted tradeoffs

| Tradeoff | Why accepted |
|----------|--------------|
| Non-commercial default weights (Pi3X) | Fine for research and a demo. The README license table lists every weight. |
| Gated defaults (DINOv3, SAM 3) | Each has an open fallback that loads on its own when access is missing. The run prints nothing about it, except that the stage cache warns when it declines to store the fallback's appearance result. |
| Small local describer (Qwen3.5-2B) | Zero cost and private. The model only verbalizes facts, so less fluency than a hosted model costs little accuracy. |
| 8 GB frame ceiling | Keeps the target hardware cheap. Quality depends on keyframe choice. |

## Known failure modes

These have been seen in evaluation runs. Each row gives the effect to expect
and the mechanism in the code behind it.

| Failure | Expected effect | Mechanism |
|---------|--------------|-----------|
| Oversized or merged instances | One change whose box covers a neighbor, a shelf, or a wall. | Changed regions link across gaps of one empty 5 cm voxel, so two adjacent changed objects become one region and get one box prompt. Fragments whose masks overlap are merged by design, and the view consensus only drops voxels that most views disagree on. The `match.max_box_volume_m3` gate demotes the largest boxes to `uncertain`. |
| Moved objects split in two | A moved object appears as a `removed` and an `added` change, or as a `moved` change that boxes only part of it. This is a common miss on the validation pairs. | One object can lift as two fragments in one epoch. The assignment is one-to-one, so at most one fragment pairs with the object in the other epoch, and the other is reported alone as `removed` or `added`. A fragment registers well onto a whole object, because the fit is measured from the smaller cloud, so the pair can be made with the wrong half. |
| Scale drift on close-ups | False changes and boxes of the wrong size or position in close-up footage. | No stage corrects the depth scale of a frame: the scale policy is `model`, and the registration is rigid. The vote tolerance is `0.05 m + 2 %` of depth, so a depth error larger than that turns consistent surfaces into free or occluded votes. |
| Wrong gravity for upward-looking captures | An upside-down room frame, or the error `the up direction is wrong`. | The sign of up is a vote of `cos(t) + sin(t)` for a camera pitched `t` below the horizon. It turns negative past 45 degrees of looking up. With a floor and a ceiling in view, the flipped up makes the ceiling the lowest plane along it, so the ceiling lands at `z = 0`. |
| Weak labels on plain primitive shapes | The label is `object`, or a wrong noun. | A crop is scored by its softmax share over 48 nouns, so a crop that matches no noun well has a low best share, and below 0.3 there is no label. The vocabulary is fixed in code. |
| Near-plane clipping | In `report.html`, a box or points vanish when you zoom in. | The 3D view drops points and clips box edges closer than 5 % of the scene radius (half the diagonal of the scene bounds, at least 1 m) to the orbit camera. This is a viewer limit, not a detection error. |
| Ambiguous rotations of near-symmetric objects | A rotated box or cylinder reports `rotation_deg` near 0, or is `uncertain`. | Registration cannot tell symmetric poses apart by fit. Among fits within 0.02 of the best, it prefers an upright pose with the smallest rotation. A turn below 10 degrees with under 5 cm of travel is not `moved`. |
| Ceiling-only scenes | The run fails at alignment. | Before it searches for a floor, the alignment compares the median camera height with the 2nd percentile of the point heights along up. If the cameras are not above that low end, as when every point is above them, it raises `median camera height above the low end of the points must be positive; the up direction is wrong`. |

What helps: keep the camera level or pitched down, avoid extreme close-ups, and
follow the [capture guide](capture-guide.md).

## Limits by design

- A region only one video saw is reported as unobserved and never as a change.
  Objects hidden in both videos are invisible to the system.
- The reconstruction takes at most 128 frames, and a run keeps at most 64
  keyframes per video (`max_keyframes` cannot be set higher).
- Very small or low-texture objects may be missed (see the capture guide).
- Lighting changes between recordings lower the appearance similarity, which
  can create false changes.

## Evaluation notes

`spatial-change evaluate` scores against SceneDiff, whose metrics are 2D masks.
Each predicted 3D object is projected into the frames, with a depth test, and
splatted into a mask. Uncertain changes are left out of the scores unless asked
(`--uncertain-as moved`). An object visible in only one video gets no mask in
the other, so the metrics score a moved object seen in one video as static. A
synthetic scene suite with exact ground truth exists in
`src/spatial_change/evaluation/synthetic.py`; it has no command-line entry
point.
