"""Hand-built scenes and segmenters for the mask-lifting tests.

A scene is a ground plane at ``z = 0`` with axis-aligned boxes on it, seen by
pinhole cameras. Depth and the object index of every pixel come from exact
ray casting, so the tests know which pixel belongs to which object.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from spatial_change.change import Candidate
from spatial_change.domain import (
    SE3,
    Epoch,
    EpochReconstruction,
    EvidenceCounts,
    Frame,
    Intrinsics,
)
from spatial_change.geometry import fit_upright_box, unproject_depth
from spatial_change.synthetic.trajectory import look_at

INTRINSICS = Intrinsics(fx=80.0, fy=80.0, cx=47.5, cy=35.5, width=96, height=72)
BACKGROUND = -1
_NO_EVIDENCE = EvidenceCounts(consistent=0, free=0, occluded=0, unobserved=0)


@dataclass(frozen=True)
class Cuboid:
    """An axis-aligned box.

    Attributes:
        low: Minimum corner.
        high: Maximum corner.
    """

    low: tuple[float, float, float]
    high: tuple[float, float, float]


def ring_poses(
    count: int,
    radius: float = 4.0,
    height: float = 3.0,
    target: Sequence[float] = (0.0, 0.0, 0.3),
    start: float = 0.0,
) -> list[SE3]:
    """Places cameras evenly on a circle around the scene.

    Args:
        count: Number of cameras.
        radius: Circle radius.
        height: Camera height.
        target: Point every camera looks at.
        start: Angle of the first camera in radians.

    Returns:
        The camera poses.
    """
    angles = start + 2.0 * np.pi * np.arange(count) / count
    return [
        look_at([radius * np.cos(a), radius * np.sin(a), height], target)
        for a in angles
    ]


def _render(
    boxes: Sequence[Cuboid], pose: SE3
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Casts one ray per pixel against the plane and the boxes.

    Args:
        boxes: The boxes.
        pose: Camera-to-world pose.

    Returns:
        Depth along the camera axis and the index of the object hit.
    """
    k = INTRINSICS
    cols, rows = np.meshgrid(np.arange(k.width), np.arange(k.height))
    rays_camera = np.stack(
        [(cols - k.cx) / k.fx, (rows - k.cy) / k.fy, np.ones(cols.shape)],
        axis=-1,
    )
    rays = rays_camera @ pose.rotation.T
    origin = pose.translation
    depth = np.full(cols.shape, np.inf)
    instance = np.full(cols.shape, BACKGROUND, np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        down = rays[..., 2] < 0.0
        plane = np.where(down, -origin[2] / rays[..., 2], np.inf)
        depth = np.where(plane > 0.0, plane, np.inf)
        for index, box in enumerate(boxes):
            low = (np.asarray(box.low) - origin) / rays
            high = (np.asarray(box.high) - origin) / rays
            near = np.minimum(low, high).max(axis=-1)
            far = np.maximum(low, high).min(axis=-1)
            hit = (near <= far) & (far > 0.0) & (near < depth)
            depth = np.where(hit, near, depth)
            instance = np.where(hit, index, instance)
    return depth.astype(np.float32), instance.astype(np.int32)


@dataclass(frozen=True, eq=False)
class HandScene:
    """Frames of a hand-built scene with their object index maps.

    Attributes:
        frames: Frames with image and depth.
        instance_maps: Per-frame ``(H, W)`` object index maps.
        epoch: The epoch of the frames.
    """

    frames: list[Frame]
    instance_maps: list[NDArray[np.int32]]
    epoch: Epoch = Epoch.BEFORE

    def reconstruction(self) -> EpochReconstruction:
        """Wraps the frames and the points of all objects.

        Returns:
            A reconstruction whose points are every valid depth pixel.
        """
        points = np.concatenate(
            [
                unproject_depth(f.depth, f.intrinsics, f.world_from_camera)
                for f in self.frames
                if f.depth is not None
            ]
        )
        count = len(points)
        return EpochReconstruction(
            epoch=self.epoch,
            frames=self.frames,
            points=points,
            colors=np.zeros((count, 3), np.uint8),
            confidence=np.ones(count, np.float32),
        )

    def object_points(
        self, index: int, frames: Sequence[int] | None = None
    ) -> NDArray[np.float64]:
        """Lists the surface points of one object seen by some frames.

        Args:
            index: Object index.
            frames: Frame positions to use; all frames if None.

        Returns:
            World points of shape ``(N, 3)``.
        """
        chosen = range(len(self.frames)) if frames is None else frames
        parts = []
        for position in chosen:
            frame = self.frames[position]
            assert frame.depth is not None
            parts.append(
                unproject_depth(
                    frame.depth,
                    frame.intrinsics,
                    frame.world_from_camera,
                    self.instance_maps[position] == index,
                )
            )
        return np.concatenate(parts)

    def segmenter(self) -> "OracleSegmenter":
        """Builds the oracle segmenter of these frames.

        Returns:
            A segmenter that answers with the exact object masks.
        """
        return OracleSegmenter(
            {
                id(f.image): m
                for f, m in zip(self.frames, self.instance_maps, strict=True)
            }
        )


def hand_scene(
    boxes: Sequence[Cuboid],
    poses: Sequence[SE3],
    epoch: Epoch = Epoch.BEFORE,
) -> HandScene:
    """Renders a scene from several cameras.

    Args:
        boxes: The boxes on the plane.
        poses: Camera-to-world poses.
        epoch: Epoch of the frames.

    Returns:
        The frames and the object index maps.
    """
    frames, maps = [], []
    for index, pose in enumerate(poses):
        depth, instance = _render(boxes, pose)
        image = np.full((INTRINSICS.height, INTRINSICS.width, 3), 90, np.uint8)
        image[instance >= 0] = 200
        frames.append(
            Frame(
                epoch=epoch,
                index=index,
                timestamp_s=float(index),
                intrinsics=INTRINSICS,
                world_from_camera=pose,
                image=image,
                depth=depth,
            )
        )
        maps.append(instance)
    return HandScene(frames, maps, epoch)


def candidate_of(
    points: NDArray[np.float64],
    candidate_id: str = "before-0",
    epoch: Epoch = Epoch.BEFORE,
) -> Candidate:
    """Wraps points in a candidate with a box and empty evidence.

    Args:
        points: The candidate's CHANGED points.
        candidate_id: Identifier.
        epoch: Epoch of the candidate.

    Returns:
        The candidate.
    """
    return Candidate(
        candidate_id=candidate_id,
        epoch=epoch,
        point_indices=np.arange(len(points), dtype=np.int64),
        points=points,
        box=fit_upright_box(points, 0.05),
        evidence=_NO_EVIDENCE,
        voxel_count=1,
        changed_fraction=1.0,
    )


@dataclass
class OracleSegmenter:
    """Returns the exact object under each point prompt.

    Attributes:
        maps: Object index map of each image, by ``id`` of the image array.
        calls: The ``(image, boxes, points)`` of each ``segment`` call.
    """

    maps: dict[int, NDArray[np.int32]]
    calls: list[
        tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64]]
    ] = field(default_factory=list)

    def segment(
        self,
        image: NDArray[np.uint8],
        boxes_xyxy: NDArray[np.float64],
        points_xy: NDArray[np.float64] | None,
    ) -> NDArray[np.bool_]:
        """Returns the instance under each point, or an empty mask.

        Args:
            image: A frame image of the scene.
            boxes_xyxy: ``(K, 4)`` boxes, recorded.
            points_xy: ``(K, 2)`` points; required.

        Returns:
            ``(K, H, W)`` masks of the objects under the points; a point on
            the background gives an empty mask.
        """
        assert points_xy is not None
        self.calls.append((image, boxes_xyxy, points_xy))
        instances = self.maps[id(image)]
        masks = np.zeros((len(points_xy), *instances.shape), np.bool_)
        for k, (x, y) in enumerate(points_xy):
            label = instances[int(round(y)), int(round(x))]
            if label != BACKGROUND:
                masks[k] = instances == label
        return masks

    def unload(self) -> None:
        """Releases nothing."""
