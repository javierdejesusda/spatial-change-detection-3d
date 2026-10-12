"""A hand-built scene pair with one change of each type, for the tests."""

from pathlib import Path

import numpy as np
from matching_pipeline import oracle_segmenter, run_pipeline
from numpy.typing import NDArray
from rotations import perturbation
from rrd_reading import Rrd
from visibility_scenes import scene_evidence

from spatial_change.domain import (
    SE3,
    BoxModel,
    Change,
    ChangeReport,
    ChangeType,
    Epoch,
    EpochReconstruction,
    EvidenceCounts,
    Frame,
    Intrinsics,
    OrientedBox,
    UnobservedRegion,
)
from spatial_change.perception import FakeFeatureEncoder
from spatial_change.visibility import (
    CrossEvidence,
    PointEvidence,
    PointState,
    VisibilityConfig,
    Vote,
)
from spatial_change.visualization import (
    RerunConfig,
    SceneVisualization,
    save_recording,
)

PATCH = 8
POINTS = 6000
FRAMES = 6
UNOBSERVED_POINTS = 40
SIZE = (48, 64)
REMOVED_BOX = (-1.0, 0.0, 3.0)
ADDED_BOX = (1.0, 0.0, 3.0)
MOVED_FROM = (0.0, 0.5, 3.0)
MOVED_TO = (0.5, 0.5, 3.0)
UNCERTAIN_BOX = (0.0, -0.5, 3.0)
REGION_BOX = (2.0, 0.0, 3.0)
OTHER_REGION_BOX = (-2.0, 0.0, 3.0)


def _tilted() -> NDArray[np.float64]:
    yaw = perturbation([0.0, 1.0, 0.0], 10.0, np.zeros(3))
    tilt = perturbation([1.0, 0.0, 0.0], 8.0, np.zeros(3))
    return yaw.rotation @ tilt.rotation


TILTED = _tilted()
TILTED_FRAME = 2


def box(center: tuple[float, float, float], size: float = 0.4) -> OrientedBox:
    """Builds an axis-aligned cube.

    Args:
        center: Box center.
        size: Side length.

    Returns:
        The box.
    """
    return OrientedBox(np.array(center), np.full(3, size), np.eye(3))


def _change(
    change_id: str,
    kind: ChangeType,
    before: OrientedBox | None,
    after: OrientedBox | None,
) -> Change:
    moved = kind is ChangeType.MOVED
    return Change(
        change_id=change_id,
        type=kind,
        label=f"{kind.value}-object",
        description=f"{kind.value} object",
        confidence=0.9,
        before_box=None if before is None else BoxModel.from_box(before),
        after_box=None if after is None else BoxModel.from_box(after),
        displacement_m=0.5 if moved else None,
        rotation_deg=0.0 if moved else None,
        evidence=EvidenceCounts(consistent=1, free=2, occluded=0, unobserved=0),
    )


def report() -> ChangeReport:
    """Builds a report with one change of each type and one unknown region.

    Returns:
        The report. The uncertain change has only a before box.
    """
    return ChangeReport(
        changes=[
            _change("added-0", ChangeType.ADDED, None, box(ADDED_BOX)),
            _change("removed-1", ChangeType.REMOVED, box(REMOVED_BOX), None),
            _change(
                "moved-2",
                ChangeType.MOVED,
                box(MOVED_FROM),
                box(MOVED_TO),
            ),
            _change(
                "uncertain-3", ChangeType.UNCERTAIN, box(UNCERTAIN_BOX), None
            ),
        ],
        unobserved_regions=[
            UnobservedRegion(
                region_id="region-0",
                box=BoxModel.from_box(box(REGION_BOX, 0.8)),
                epoch_seen=Epoch.BEFORE,
            ),
            UnobservedRegion(
                region_id="region-1",
                box=BoxModel.from_box(box(OTHER_REGION_BOX, 0.8)),
                epoch_seen=Epoch.AFTER,
            ),
        ],
        metadata={},
    )


def _frames(epoch: Epoch, rng: np.random.Generator) -> list[Frame]:
    height, width = SIZE
    intrinsics = Intrinsics(50.0, 50.0, 31.5, 23.5, width, height)
    frames = []
    for i in range(FRAMES):
        x = -1.5 + 3.0 * i / (FRAMES - 1)
        rotation = TILTED if i == TILTED_FRAME else np.eye(3)
        pose = SE3(rotation, np.array([x, 0.0, 0.0]))
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        depth = np.full((height, width), 3.0, np.float32)
        frames.append(Frame(epoch, i, i / 30.0, intrinsics, pose, image, depth))
    return frames


def _reconstruction(
    epoch: Epoch, rng: np.random.Generator
) -> EpochReconstruction:
    points = rng.uniform((-2.0, -1.0, 2.5), (2.0, 1.0, 3.5), (POINTS, 3))
    return EpochReconstruction(
        epoch=epoch,
        frames=_frames(epoch, rng),
        points=points,
        colors=rng.integers(0, 256, (POINTS, 3), dtype=np.uint8),
        confidence=np.ones(POINTS, np.float32),
    )


def _evidence(unobserved: int) -> PointEvidence:
    counts = np.zeros((POINTS, 4), np.int32)
    counts[:, Vote.CONSISTENT] = 2
    counts[:unobserved] = 0
    counts[:unobserved, Vote.UNOBSERVED] = 3
    states = np.full(POINTS, PointState.PRESENT, np.int8)
    states[:unobserved] = PointState.UNKNOWN
    return PointEvidence(counts, states)


def hand_scene(with_evidence: bool = True) -> SceneVisualization:
    """Builds the scene pair of the visualization tests.

    Args:
        with_evidence: Whether to include the cross evidence, whose first
            ``UNOBSERVED_POINTS`` points of each epoch were never seen.

    Returns:
        The scene.
    """
    rng = np.random.default_rng(0)
    evidence: CrossEvidence | None = None
    if with_evidence:
        evidence = CrossEvidence(
            before=_evidence(UNOBSERVED_POINTS),
            after=_evidence(UNOBSERVED_POINTS),
        )
    return SceneVisualization(
        before=_reconstruction(Epoch.BEFORE, rng),
        after=_reconstruction(Epoch.AFTER, rng),
        report=report(),
        evidence=evidence,
    )


def synthetic_scene(seed: int) -> SceneVisualization:
    """Builds the scene of a rendered synthetic pair after association.

    Uses the oracle segmenter and the fake encoder, so it runs on the CPU.

    Args:
        seed: Scene seed.

    Returns:
        The clouds colored from the rendered frames, the cross evidence, and
        the report of the changes with the scene's unobserved zone.
    """
    votes = scene_evidence(seed, VisibilityConfig())
    run = run_pipeline(
        votes, oracle_segmenter(votes.pair), FakeFeatureEncoder(PATCH)
    )
    recons = {}
    for epoch, cloud in votes.clouds.items():
        frames = votes.pair.epoch_data(epoch).frames
        colors = []
        for frame in frames:
            assert frame.image is not None and frame.depth is not None
            valid = np.isfinite(frame.depth) & (frame.depth > 0)
            colors.append(frame.image[valid])
        recons[epoch] = EpochReconstruction(
            epoch=epoch,
            frames=list(frames),
            points=cloud.points,
            colors=np.concatenate(colors),
            confidence=np.ones(len(cloud.points), np.float32),
        )
    report = ChangeReport(
        changes=list(run.changes),
        unobserved_regions=[votes.pair.unobserved_region()],
        metadata={},
    )
    return SceneVisualization(
        before=recons[Epoch.BEFORE],
        after=recons[Epoch.AFTER],
        report=report,
        evidence=CrossEvidence(
            before=votes.evidence[Epoch.BEFORE],
            after=votes.evidence[Epoch.AFTER],
        ),
    )


def record(
    tmp_path: Path,
    scene: SceneVisualization | None = None,
    config: RerunConfig | None = None,
) -> Rrd:
    """Saves a scene and reads the recording back.

    Args:
        tmp_path: Directory to write ``scene.rrd`` in.
        scene: The scene; the hand-built one if None.
        config: The recording config; the default if None.

    Returns:
        The saved recording.
    """
    path = tmp_path / "scene.rrd"
    save_recording(
        hand_scene() if scene is None else scene,
        path,
        RerunConfig() if config is None else config,
    )
    return Rrd(path)
