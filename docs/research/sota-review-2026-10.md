# State of the Art Review: 3D Scene Change Detection from Two Casual Videos

**Date of research**: 2026-10-10
**Recency window applied**: 2025-04 to 2026-10 (older work only when it is still the reference baseline)
**Method**: six parallel research vectors (change-detection papers, reconstruction and 3D segmentation papers, industry write-ups, open-source repo health, benchmarks, current library docs), then cross-verification.

## Research target

Given two handheld phone videos (plain RGB, no LiDAR or AR poses) of the same small indoor area, captured at different times from different viewpoints with partial overlap:

1. Build a shared, co-registered 3D representation.
2. Detect and localize in 3D the objects that were moved, added, or removed.
3. Separate real change from viewpoint differences, occlusion, lighting, and unobserved regions.
4. Describe every change with semantic labels and natural language.

Constraints: Python and PyTorch, a local RTX 4070 Laptop GPU with 8 GB VRAM on Windows 11, optional cloud GPU, no real-time requirement, a simple and polished user experience.

## SOTA landscape

- **Geometry has moved to feed-forward foundation models.** VGGT (CVPR 2025), Pi3, Depth Anything 3 (ICLR 2026), MapAnything (3DV 2026), and VGGT-Omega (CVPR 2026 oral) estimate cameras and dense geometry for many unposed frames in one forward pass.
  - Running both videos jointly through one of these models puts them in a single coordinate frame directly, which removes most of the classic SfM and cross-session registration pain.
  - Rankings across papers are stable (VGGT-Omega > DA3 >= Pi3 > VGGT > MapAnything on pose), but absolute numbers differ by up to 20 points between papers, and two papers flag benchmark contamination.
- **The closest published system is SceneDiff (ECCV 2026).** It is training-free, takes two casual videos, and labels objects as added, removed, or moved. It ships MIT-licensed code and a 350-pair benchmark.
  - It outputs 2D masks only: no 3D objects, no semantic labels, no language.
  - It has no explicit free-space reasoning, so "removed" and "not observed" can be confused.
- **3D Gaussian Splatting change detectors** (MV-3DCD, O-SCD, GS-Diff, GS-Pool) lead the PASLCD pixel-level benchmark.
  - They need a 3DGS reconstruction per epoch, most are unreleased, and they have no added/removed/moved taxonomy.
  - 3DGS-CD scored about 1.0 object AP on casual video in SceneDiff's comparison.
- **Learned change heads are emerging.** Argos (Oct 2026) is a frozen VGGT-Omega plus cross-session attention, but its code release is unconfirmed and it outputs 2D masks.
- **VLMs are poor change detectors.**
  - Benchmarks show a "no difference" bias on before/after pairs (VDiff-Bench) and weak viewpoint reasoning (best 68.5% on the VLM-SubtleBench viewpoint split, against 95.5% for humans).
  - Production systems (for example robot inspection with Gemini Robotics-ER 2) use 2D VLM prompting, not 3D change reasoning.
  - Consensus: localize change with geometry, then use a VLM to label and describe it.
- **No published system covers the full target end to end.** None combines casual RGB video, 3D localization, an added/removed/moved taxonomy, explicit visibility reasoning, and grounded natural language. That gap is the opportunity.

## Top candidates per pipeline stage

### Stage 1: joint reconstruction and co-registration

#### 1. Pi3 / Pi3X (default candidate)

- **What it is:** a permutation-equivariant feed-forward reconstruction model. Pi3X (Dec 2025) adds approximate metric scale.
- **Source:** https://github.com/yyfz/Pi3
- **Evidence:**
  - Backbone of SceneDiff. Its ablation shows that swapping Pi3 for VGGT drops per-scene object AP by 8.7, and FASt3R drops it by 22.4 (https://arxiv.org/html/2512.16908v2).
  - Third-party AUC@30 is 77.0 on 7-Scenes and 79.6 on ETH3D (VGGT-Omega paper table).
- **Strengths:**
  - Proven for change detection specifically.
  - Ungated weights.
  - Handles scenes where many objects moved.
- **Weaknesses:**
  - Weights are CC BY-NC; the HF card and README conflict on license.
  - No published VRAM numbers.
  - Pins torch 2.5.1 and numpy < 2.

#### 2. VGGT-Omega 1B (accuracy ceiling)

- **What it is:** the successor to VGGT, from Meta and Oxford (arXiv 2605.15195, CVPR 2026 oral).
- **Source:** https://github.com/facebookresearch/vggt-omega
- **Evidence:**
  - Best pose AUC in every table found: 7-Scenes AUC@30 83.1, ETH3D 88.5.
  - The authors disclose possible contamination. Their reproduction checkpoint scores 27.4 instead of 29.6 AUC@3 on 7-Scenes.
- **Memory (official, A100, 624x416, including weights):** 10 frames 6.7 GB, 25 frames 7.8 GB, 50 frames 9.7 GB, 100 frames 13.4 GB. On 8 GB that means roughly 20 to 25 frames per pass.
- **Weaknesses:**
  - Gated weights under the FAIR Noncommercial Research License.
  - Small frame budget locally.

#### 3. Depth Anything 3 (permissive fallback)

- **Source:** https://github.com/ByteDance-Seed/Depth-Anything-3, arXiv 2511.10647.
- **Evidence:** DA3-Giant leads Pi3 on HiRoom, ScanNet++, and 7-Scenes AUC@3 (self-reported).
- **Strengths:**
  - Base and Small weights are Apache 2.0.
  - A streaming mode runs in under 12 GB.
  - A metric variant exists.
- **Weaknesses:**
  - Base is much weaker than Giant (7-Scenes AUC@3 20.1 vs 28.5), and Giant is CC BY-NC.
  - The Large license conflicts between the HF card and the README.

#### 4. MapAnything (metric conditioning only)

- **Source:** https://github.com/facebookresearch/map-anything
- **Evidence:** the healthiest repo, with Apache weights, and it can condition on known intrinsics.
- **Weaknesses:**
  - Ranks last on pose in the DA3 and VGGT-Omega tables, which conflicts with its own paper.
  - Pins old rerun and pycolmap versions.

#### 5. Cross-session registration recipe: VGGT-CD

- **Source:** arXiv 2605.16859.
- **Recipe:** a joint pass on farthest-point keyframes from both epochs, then Umeyama Sim(3), then translation refinement on static correspondences. Indoor ATE is 0.14 m, against 0.33 m for the best baseline.
- **Use:** adopt the idea, not the code (18 GB peak). An ATE of 14 cm is large next to small-object moves, so a final robust ICP restricted to static structure is required.

**Classical baseline:** COLMAP 4.2 global mapper (GLOMAP merged March 2026, BSD). It is slow but serves as pseudo ground truth and a fallback. Independent evidence is mixed: it beats VGGT on handheld objects but fails on 56% of 360-degree clips (ORBIT++).

### Stage 2: masks, features, labels

| Component | Choice | Evidence | License / access |
|---|---|---|---|
| Promptable masks | SAM 3 / 3.1 via `transformers` `Sam3Model`; fallback SAM 2.1 | SAM 3: LVIS mask AP 48.5, SA-Co/Gold cgF1 54.1 vs OWLv2 24.6 (Meta, self-reported) | SAM 3 gated (SAM License); SAM 2.1 Apache and open |
| Object association features | DINOv3 ViT-B; fallback DINOv2 | SceneDiff matches moved objects with DINOv3 cosine > 0.7 | DINOv3 gated; DINOv2 open |
| Zero-shot labels (offline) | SigLIP 2 | standard open-vocabulary classifier, in `transformers` 5.19 | open |
| Avoid | YOLOE, YOLO-World | | AGPL / GPL license trap |
| Avoid as dependencies | OpenMask3D, Open3DIS, MaskClustering, ConceptGraphs, OpenYOLO3D | built for clean RGB-D scans, repos stale, untested on reconstructed geometry | mixed |

Zero-shot 3D lifting methods report class-agnostic AP on ScanNet200 of 35.5 for MV3DIS, 32.5 for Any3DIS, and about 19.7 for MaskClustering. Open-vocabulary AP is 26.0 for OpenTrack3D. All of these assume sensor depth and ground-truth poses. Lifting only the change candidates (change-first) avoids depending on full-scene 3D instance segmentation.

### Stage 3: change reasoning

| Approach | Venue / date | Result | Code | Fit |
|---|---|---|---|---|
| SceneDiff | ECCV 2026 | SD-V: px IoU 39.1, obj/image AP 43.8, obj/scene AP 22.8; SD-K: 20.8 / 20.9 / 10.6; 159 s per pair on A40 | MIT, plus MIT benchmark | Closest; 2D output, no free-space logic |
| Argos | arXiv 2026-10 | IoU 62.5 / F1 46.0 on the SceneDiff benchmark (different protocol) | unconfirmed | Watch; learned, 2D |
| GS-Diff / GS-Pool / O-SCD | CVPR 2026 and arXiv 2026 | PASLCD mIoU 0.644 / 0.751 / 0.552 | mostly unreleased | Pixel-level, heavy 3DGS |
| Free-space ray voting (room-diff, Khronos and POCD lineage) | 2026 / classical | Unvalidated on real data | single-commit repo | The only clean separation of "removed" from "unobserved" |
| FOCUS, OASIS-Map, ReScene4D | 2026 | Robot RGB-D and odometry assumptions | none | Concepts only |

### Stage 4: language

- **The VLM's role:** describe the evidence, do not detect the change. Ground truth for language should come from geometry.
  - On MMSI multi-image spatial reasoning, Gemini-3-pro scores 49.2, GPT-5 41.9, Claude 3.7 Sonnet 30.2, and humans 97.2.
- **Hosted options:** Claude, Gemini, and OpenAI SDKs all support multi-image input with structured (schema-validated) output.
- **Local option:** Qwen3.5 (Apache, March 2026). At 2B it fits in 8 GB; at 4B it needs quantization.

### Stage 5: presentation

- **Rerun 0.38.1:** Windows wheels, `.rrd` recordings, web viewer, and a Gradio embed component.
- **viser 1.1.1:** a lightweight alternative.
- **Gradio 6.30:** for the upload-two-videos web UI.
- **Skip gsplat on Windows:** the MSVC build is broken on main. Defer any splat rendering to WSL2 as a stretch goal.

## Comparison matrix (reconstruction backbone)

| Backbone | Date | Pose evidence (AUC@30, 7-Scenes / ETH3D) | 8 GB local | Weights license | Change-detection evidence | Verdict |
|---|---|---|---|---|---|---|
| VGGT-Omega 1B | 2026-05 | 83.1 / 88.5 (contamination caveat) | about 20-25 frames | FAIR NC, gated | Argos backbone | Optional high-accuracy adapter |
| Pi3 / Pi3X | 2025-07 / 2025-12 | 77.0 / 79.6 | unmeasured | CC BY-NC | SceneDiff backbone, best in its ablation | Default candidate, confirm in spike |
| DA3 Base / Giant | 2025-11 | 78.2 / 87.0 (Giant) | Base likely | Base Apache, Giant NC | none | Permissive fallback |
| VGGT 1B | 2025-03 | 74.4 / 62.1 | about 6 frames reported on 8 GB in fp16 | CC BY-NC | worse than Pi3 in SceneDiff | Superseded |
| MapAnything (apache) | 2025-09 | 61.4 / 51.0 | memory-efficient mode | Apache | none | Metric conditioning only |
| COLMAP 4.2 global | 2026-09 | classical | CPU/GPU | BSD | used as pseudo-GT | Baseline and fallback |

## Conflicts and uncertainties

- **Benchmark tables disagree.** VGGT on ETH3D is AUC@30 80.8 in the DA3 paper and 62.1 in the VGGT-Omega paper. MapAnything claims to beat VGGT in its own paper and ranks last in competitors' tables. Only rankings are trusted, not absolute values.
- **Contamination.** VGGT-Omega and AMB3R both flag possible training overlap with 7-Scenes and NRGBD. These are not used as headline numbers.
- **SceneDiff numbers changed between arXiv versions** (v1 per-scene AP 46.3 vs v2 22.8, after a metric redefinition). Only v2 tables are cited. Argos reports SceneDiff at 30.31 IoU under its own protocol, so cross-paper comparisons are invalid.
- **License conflicts.** Pi3 (HF card BSD-2 vs README CC BY-NC) and DA3-Large (HF Apache vs README NC). Both are treated as non-commercial.
- **Dependency conflicts.** rerun-sdk needs numpy >= 2. VGGT, VGGT-Omega, Pi3, and the `sam3` package pin numpy < 2. This must be validated in a spike, with an isolated reconstruction environment as the fallback.
- **Windows.** The native `sam3` package has open Windows issues (forced FlashAttention, NCCL). The `transformers` port should avoid them, but this is unverified on this machine.
- **VRAM.** Only VGGT-Omega publishes memory numbers. Pi3 and DA3-Base must be measured locally.

## Recommendation

Build a **visibility-aware, object-level 3D change reasoner on top of a feed-forward reconstruction**, using four cues:

1. **Geometry proposes.** Run both videos jointly through Pi3X, with VGGT-Omega or DA3 as swappable adapters chosen by a measured spike. Use keyframe anchors plus chunked densification and Sim(3) alignment.
2. **Visibility vetoes.** Each surface point is tested against the other session's depth maps and classified as consistent, free-space violated, occluded, or unobserved. Only free-space evidence can declare a removal or an addition. Unobserved regions are reported as unknown, never as change.
3. **Appearance confirms.** Dense features (DINOv3 or DINOv2) at co-visible surface points catch changes that geometry misses, such as flat objects or same-shape swaps.
4. **Language explains.** A local VLM (Qwen3.5-2B, zero cost) receives before/after crops plus structured 3D facts (type, displacement, rotation, supporting surface, nearest landmarks) and returns schema-validated descriptions, which are checked against the geometry. A deterministic template path guarantees a result even without the VLM.

Changes are excluded from registration iteratively (change-aware alignment). Moved objects are recovered by matching disappeared and appeared object instances on appearance and shape, then estimating a rigid transform.

Evaluate quantitatively on the public SceneDiff benchmark (MIT), using its own metrics, by projecting 3D change objects back into frames. Also evaluate on a procedural synthetic suite with exact ground truth.

**Tradeoffs accepted:**

- Non-commercial model weights are acceptable for a research demo and are documented in a license table. Every default has an open fallback.
- 8 GB VRAM caps frames per pass, so quality depends on keyframe selection and chunked alignment.
- Zero cost: language comes from a small local VLM (Qwen3.5-2B, Apache) plus a deterministic template fallback. That gives up some description fluency compared with frontier hosted models. Grounding the model in structured 3D facts limits the loss, because the model only has to verbalize evidence, not discover it.

## Sources

**Change detection**
- SceneDiff: https://arxiv.org/abs/2512.16908, https://arxiv.org/html/2512.16908v2, https://github.com/yuqunw/scene_diff, https://huggingface.co/datasets/yuqun/SceneDiff
- Argos: https://arxiv.org/abs/2610.10181, https://www.multyxu.com/argos/
- VGGT-CD: https://arxiv.org/abs/2605.16859, https://github.com/WZ-CS/VGGT-CD
- MV-3DCD: https://arxiv.org/abs/2412.03911
- O-SCD: https://arxiv.org/abs/2511.12370, https://github.com/Chumsy0725/O-SCD
- GS-Diff: https://arxiv.org/abs/2605.07203
- GS-Pool: https://arxiv.org/abs/2610.06688
- 3DGS-CD: https://arxiv.org/abs/2411.03706, https://github.com/520xyxyzq/3DGS-CD
- SCaR-3D: https://arxiv.org/abs/2512.22830
- room-diff: https://github.com/viv344/room-diff
- VSCD: https://arxiv.org/abs/2605.20821
- ReScene4D: https://arxiv.org/abs/2601.11508
- Situat3DChange: https://arxiv.org/abs/2510.11509
- FOCUS: https://arxiv.org/abs/2610.05639
- OASIS-Map: https://arxiv.org/abs/2607.14899
- Khronos: https://github.com/MIT-SPARK/Khronos
- Awesome list: https://github.com/DoongLi/Awesome-Scene-Change-Detection

**Reconstruction**
- VGGT-Omega: https://arxiv.org/html/2605.15195, https://github.com/facebookresearch/vggt-omega
- DA3: https://arxiv.org/html/2511.10647, https://github.com/ByteDance-Seed/Depth-Anything-3
- MapAnything: https://arxiv.org/html/2509.13414v3, https://github.com/facebookresearch/map-anything
- Pi3: https://github.com/yyfz/Pi3
- VGGT: https://github.com/facebookresearch/vggt
- VGGT-Long: https://github.com/DengKaiCQ/VGGT-Long
- VGGT-SLAM 2.0: https://arxiv.org/abs/2601.19887
- AMB3R: https://arxiv.org/html/2511.20343
- E3D-Bench: https://arxiv.org/html/2506.01933v3
- ORBIT++: https://arxiv.org/html/2608.22039
- COLMAP: https://github.com/colmap/colmap

**Segmentation and features**
- SAM 3: https://arxiv.org/html/2511.16719v2, https://github.com/facebookresearch/sam3
- MV3DIS: https://arxiv.org/html/2604.08916
- Any3DIS: https://arxiv.org/html/2411.16183v1
- OpenTrack3D: https://arxiv.org/html/2512.03532v2
- SpaCeFormer compilation: https://arxiv.org/html/2604.20395v1
- OVSeg3R: https://arxiv.org/html/2509.23541
- DINOv3: https://github.com/facebookresearch/dinov3
- YOLOE: https://arxiv.org/html/2503.07465v1

**VLMs and industry**
- Gemini Robotics-ER 2: https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/, https://ai.google.dev/gemini-api/docs/robotics-spatial
- Robot inspection with Gemini: https://bostondynamics.com/blog/aivi-learning-now-powered-google-gemini-robotics/
- SAM 3D: https://ai.meta.com/blog/sam-3d/
- MMSI-Bench: https://github.com/InternRobotics/MMSI-Bench
- SpatialScore: https://arxiv.org/html/2505.17012v3
- VLM-SubtleBench: https://arxiv.org/html/2603.07888
- EASI: https://arxiv.org/html/2508.13142
- VDiff-Bench: https://arxiv.org/abs/2609.06245
- ReVSI: https://arxiv.org/abs/2604.24300

**Libraries**
- PyPI JSON for torch 2.14.1, transformers 5.19.0, open3d 0.20.0, rerun-sdk 0.38.1, gradio 6.30.0, viser 1.1.1, anthropic 1.13.0, google-genai 2.29.0, openai 3.28.0
- PyTorch wheel index: https://download.pytorch.org/whl/cu130
- uv with PyTorch: https://docs.astral.sh/uv/guides/integration/pytorch

## What this research did not cover

- **No hands-on measurement yet.** VRAM, speed, and numpy compatibility of Pi3 and DA3 on this machine are unverified.
- **Self-reported numbers.** Most benchmark numbers are self-reported. Independent indoor phone-video comparisons of feed-forward models against COLMAP were not found.
- **Robustness gaps.** Robustness of change detection to strong lighting changes is documented only as a failure mode (SceneDiff, PlenoCI). No method addresses it well.
- **Unloaded claims.** CUT3R, Fast3R, StreamVGGT, HOV-SG, Florence-2, and Grounding DINO 1.5/1.6 claims were not loaded and are not relied on.
- **No published approach found** for SAM 3 masks lifted to 3D in a change-detection setting. This pipeline's combination is new and unproven.
