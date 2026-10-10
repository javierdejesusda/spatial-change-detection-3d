# Spatial Change Detection 3D

Robots and spatial agents need to know how their surroundings changed between
visits. This project takes two phone videos of the same indoor area, recorded
at different times, and reports which objects were added, removed, or moved,
in 3D and in plain language.

Status: under active development.

## What it will do

- Build a shared 3D reconstruction from two casual phone videos.
- Detect added, removed, and moved objects in 3D.
- Stay visibility-aware: occlusion and regions seen in only one video are
  reported as unobserved, not as change.
- Attach labels and grounded natural-language descriptions to each change.

## Quickstart (development)

Install with an NVIDIA GPU (CUDA 13.0 build of PyTorch):

```bash
uv sync --extra cu130
```

or on CPU only:

```bash
uv sync --extra cpu
```

Then check the install and run the tests:

```bash
uv run spatial-change --version
uv run pytest -m "not gpu"
```

## Recording your own videos

See the [capture guide](docs/capture-guide.md) for how to record a
before/after pair that works well.

## License

MIT. See [LICENSE](LICENSE).
