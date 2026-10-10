"""Procedural two-epoch rooms with exact ground truth.

The room spans ``[0, X] x [0, Y] x [0, Z]`` in meters with z up. Each seed
produces one layout in which, between the before and after epochs:

* some objects are unchanged, one is moved, one is added, one is removed;
* a tall unchanged cabinet (the occluder) hides an unchanged stool from
  every after view while the stool lies inside the after frusta;
* a box-shaped zone near the x = 0 wall holds an unchanged crate, is seen
  by the before cameras, and lies outside every after frustum.

Placement uses rejection sampling from ``numpy.random.default_rng(seed)``.
A candidate layout is rendered and discarded unless the renders confirm all
of the properties above, so the ground truth always agrees with the images.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import (
    SE3,
    BoxModel,
    ChangeType,
    Epoch,
    Frame,
    Intrinsics,
    OrientedBox,
    UnobservedRegion,
)
from spatial_change.geometry.projection import (
    in_image,
    project,
    unproject_depth,
)
from spatial_change.synthetic.layout import (
    Disk,
    Rect,
    disk_clear_of_rect,
    disk_in_room,
    disks_clear,
    footprint,
)
from spatial_change.synthetic.objects import (
    FACTORIES,
    ObjectModel,
    place,
    world_box,
    world_vertices,
)
from spatial_change.synthetic.render import PlacedMesh, RenderScene
from spatial_change.synthetic.trajectory import (
    Waypoint,
    sees_points,
    sweep,
)

_MIN_ROOM = (7.0, 5.5, 2.4)
_RIG_X = 2.4
_OCCLUDER_GAP = 1.8
_HIDDEN_GAP = 1.0
_X_OCCLUDER = _RIG_X + _OCCLUDER_GAP
_X_HIDDEN = _X_OCCLUDER + _HIDDEN_GAP
_SIDE_OFFSET = 1.7
_ZONE_X = (0.15, 1.35)
_ZONE_SIZE_Y = 1.2
_ZONE_SIZE_Z = 1.4
_FOOTPRINT_GAP = 0.10
_WALL_MARGIN = 0.15
_CAMERA_MARGIN = 0.2
_CAMERA_CLEARANCE = 0.15
_MIN_PIXELS = 6
_SLOT_TRIES = 150
_VIEW_MARGIN_PX = 3
_NEAR_M = 0.5
_SEEN_THROUGH_MARGIN_M = 0.05
_BOX_INSET = 0.8
_MIN_SEEN_CORNERS = 0.5
_POSITION_JITTER_M = 0.01
_TARGET_JITTER_M = 0.02


class ObjectStatus(StrEnum):
    """How an object differs between the two epochs."""

    ADDED = "added"
    REMOVED = "removed"
    MOVED = "moved"
    UNCHANGED = "unchanged"

    @property
    def change_type(self) -> ChangeType | None:
        """The matching domain change type, or None for unchanged."""
        return (
            None if self is ObjectStatus.UNCHANGED else ChangeType(self.value)
        )


@dataclass(frozen=True)
class SceneConfig:
    """Parameters of the synthetic scene generator.

    Attributes:
        room_size: Room extent ``(x, y, z)`` in meters.
        width: Image width in pixels.
        height: Image height in pixels.
        hfov_deg: Horizontal field of view in degrees.
        fps: Video frame rate used for timestamps.
        frames_per_epoch: Number of rendered frames per epoch, at least 4.
        min_move_m: Minimum box-center displacement of the moved object.
        max_attempts: Candidate layouts tried before giving up.
    """

    room_size: tuple[float, float, float] = (8.0, 6.0, 2.8)
    width: int = 96
    height: int = 72
    hfov_deg: float = 70.0
    fps: float = 10.0
    frames_per_epoch: int = 8
    min_move_m: float = 0.3
    max_attempts: int = 200

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If the room is smaller than the layout needs, the
                image or field of view is degenerate, there are fewer than
                4 frames per epoch, or the minimum move is not positive.
        """
        if any(s < m for s, m in zip(self.room_size, _MIN_ROOM, strict=True)):
            raise ValueError(f"room_size must be at least {_MIN_ROOM}")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("width, height, and fps must be positive")
        if not 20.0 <= self.hfov_deg <= 120.0:
            raise ValueError("hfov_deg must be within [20, 120]")
        if self.frames_per_epoch < 4:
            raise ValueError("frames_per_epoch must be at least 4")
        if self.min_move_m <= 0 or self.max_attempts < 1:
            raise ValueError("min_move_m and max_attempts must be positive")

    @property
    def intrinsics(self) -> Intrinsics:
        """The pinhole intrinsics shared by all frames."""
        focal = (self.width / 2.0) / np.tan(np.radians(self.hfov_deg) / 2.0)
        return Intrinsics(
            fx=float(focal),
            fy=float(focal),
            cx=(self.width - 1) / 2.0,
            cy=(self.height - 1) / 2.0,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True, eq=False)
class SyntheticObject:
    """One object with its ground truth in both epochs.

    Attributes:
        index: Position in ``SyntheticScenePair.objects``; the value that
            marks this object in instance maps.
        object_id: Stable identifier, unique within the pair.
        label: Class name of the object.
        status: Added, removed, moved, or unchanged.
        model: Local mesh and color.
        pose_before: ``world_from_object`` in the before epoch, or None if
            the object does not exist there.
        pose_after: ``world_from_object`` in the after epoch, or None.
        box_before: Tight world box in the before epoch, or None.
        box_after: Tight world box in the after epoch, or None.
        motion: For moved objects, the rigid world transform taking the
            before box onto the after box; None otherwise.
        support_id: Identifier of the object this one rests on, or None if
            it rests on the floor.
    """

    index: int
    object_id: str
    label: str
    status: ObjectStatus
    model: ObjectModel
    pose_before: SE3 | None
    pose_after: SE3 | None
    box_before: OrientedBox | None
    box_after: OrientedBox | None
    motion: SE3 | None
    support_id: str | None = None

    def exists_in(self, epoch: Epoch) -> bool:
        """Tells whether the object is present in an epoch.

        Args:
            epoch: The epoch to test.

        Returns:
            True if the object exists in ``epoch``.
        """
        pose = self.pose_before if epoch == Epoch.BEFORE else self.pose_after
        return pose is not None

    def pose(self, epoch: Epoch) -> SE3:
        """Returns the object pose in an epoch.

        Args:
            epoch: The epoch.

        Returns:
            The ``world_from_object`` pose.

        Raises:
            ValueError: If the object does not exist in ``epoch``.
        """
        pose = self.pose_before if epoch == Epoch.BEFORE else self.pose_after
        if pose is None:
            raise ValueError(f"{self.object_id} is absent in {epoch}")
        return pose

    def box(self, epoch: Epoch) -> OrientedBox:
        """Returns the tight world box in an epoch.

        Args:
            epoch: The epoch.

        Returns:
            The oriented box.

        Raises:
            ValueError: If the object does not exist in ``epoch``.
        """
        box = self.box_before if epoch == Epoch.BEFORE else self.box_after
        if box is None:
            raise ValueError(f"{self.object_id} is absent in {epoch}")
        return box

    @property
    def displacement_m(self) -> float:
        """Distance between the box centers of a moved object, else 0."""
        if self.box_before is None or self.box_after is None:
            return 0.0
        return float(
            np.linalg.norm(self.box_after.center - self.box_before.center)
        )


@dataclass(frozen=True, eq=False)
class SyntheticEpoch:
    """The rendered frames of one epoch.

    Attributes:
        epoch: Which capture this is.
        frames: Posed frames with image and depth.
        instance_maps: Per-frame ``(H, W)`` int32 object index maps, -1
            for the room shell.
    """

    epoch: Epoch
    frames: list[Frame]
    instance_maps: list[NDArray[np.int32]]


@dataclass(frozen=True, eq=False)
class SyntheticScenePair:
    """Two rendered epochs of one room with exact ground truth.

    Attributes:
        seed: Seed the pair was generated from.
        config: Generator parameters.
        objects: All objects; ``objects[i].index == i``.
        before: Rendered before epoch.
        after: Rendered after epoch.
        unobserved_zone: Box that no after camera sees.
        zone_seen_in: The epoch that sees the zone.
        occluder_id: Object that hides ``occluded_object_ids`` in the
            after epoch.
        occluded_object_ids: Unchanged objects inside the after frusta
            whose pixels are fully hidden behind the occluder.
        zone_object_ids: Unchanged objects inside the unobserved zone.
    """

    seed: int
    config: SceneConfig
    objects: tuple[SyntheticObject, ...]
    before: SyntheticEpoch
    after: SyntheticEpoch
    unobserved_zone: OrientedBox
    zone_seen_in: Epoch
    occluder_id: str
    occluded_object_ids: tuple[str, ...]
    zone_object_ids: tuple[str, ...]

    @property
    def room_size(self) -> tuple[float, float, float]:
        """The room extent ``(x, y, z)`` in meters."""
        return self.config.room_size

    def epoch_data(self, epoch: Epoch) -> SyntheticEpoch:
        """Returns the rendered data of an epoch.

        Args:
            epoch: The epoch.

        Returns:
            Its frames and instance maps.
        """
        return self.before if epoch == Epoch.BEFORE else self.after

    def objects_in(self, epoch: Epoch) -> list[SyntheticObject]:
        """Lists the objects present in an epoch.

        Args:
            epoch: The epoch.

        Returns:
            Objects that exist in ``epoch``.
        """
        return [o for o in self.objects if o.exists_in(epoch)]

    def object_by_id(self, object_id: str) -> SyntheticObject:
        """Looks an object up by identifier.

        Args:
            object_id: The identifier.

        Returns:
            The object.

        Raises:
            KeyError: If no object has that identifier.
        """
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        raise KeyError(object_id)

    def unobserved_region(
        self, region_id: str = "unobserved-0"
    ) -> UnobservedRegion:
        """Converts the unobserved zone to the report type.

        Args:
            region_id: Identifier of the region.

        Returns:
            The zone as an ``UnobservedRegion`` seen in ``zone_seen_in``.
        """
        return UnobservedRegion(
            region_id=region_id,
            box=BoxModel.from_box(self.unobserved_zone),
            epoch_seen=self.zone_seen_in,
        )


_DEFAULT_CONFIG = SceneConfig()


def generate_scene_pair(
    seed: int, config: SceneConfig = _DEFAULT_CONFIG
) -> SyntheticScenePair:
    """Generates a room in two epochs with exact ground truth.

    Args:
        seed: Seed of the random generator; the result is fully
            deterministic for a seed and config.
        config: Generator parameters.

    Returns:
        The rendered pair and its ground truth.

    Raises:
        RuntimeError: If no valid layout is found within
            ``config.max_attempts`` candidates.
    """
    rng = np.random.default_rng(seed)
    for _ in range(config.max_attempts):
        pair = _try_generate(seed, rng, config)
        if pair is not None:
            return pair
    raise RuntimeError(
        f"no valid layout for seed {seed} in {config.max_attempts} attempts"
    )


def _object_id(index: int, label: str) -> str:
    return f"{label}_{index}"


def _make_object(
    index: int,
    model: ObjectModel,
    status: ObjectStatus,
    pose_before: SE3 | None,
    pose_after: SE3 | None,
    support: SyntheticObject | None = None,
) -> SyntheticObject:
    box_before = None if pose_before is None else world_box(model, pose_before)
    box_after = None if pose_after is None else world_box(model, pose_after)
    motion = None
    if status == ObjectStatus.MOVED:
        assert pose_before is not None and pose_after is not None
        motion = pose_after.compose(pose_before.inverse())
    return SyntheticObject(
        index=index,
        object_id=_object_id(index, model.label),
        label=model.label,
        status=status,
        model=model,
        pose_before=pose_before,
        pose_after=pose_after,
        box_before=box_before,
        box_after=box_after,
        motion=motion,
        support_id=None if support is None else support.object_id,
    )


def _grown(box: OrientedBox, pad: float) -> OrientedBox:
    """Returns ``box`` with every side length increased by ``pad``."""
    return OrientedBox(box.center, box.extent + pad, box.rotation)


def _inset(intrinsics: Intrinsics, margin_px: int) -> Intrinsics:
    """Shrinks the image by a border so ``in_image`` tests the inner area."""
    return replace(
        intrinsics,
        cx=intrinsics.cx - margin_px,
        cy=intrinsics.cy - margin_px,
        width=intrinsics.width - 2 * margin_px,
        height=intrinsics.height - 2 * margin_px,
    )


def _visible_in_any(
    point: NDArray[np.float64], inset: Intrinsics, poses: list[SE3]
) -> bool:
    """Tells whether a point is comfortably inside some camera's view.

    The point must be farther than ``_NEAR_M`` and project into ``inset``,
    the image shrunk by ``_VIEW_MARGIN_PX`` pixels on every side.
    """
    for pose in poses:
        uv, depth = project(point[None, :], inset, pose)
        if in_image(uv, depth, inset)[0] and depth[0] > _NEAR_M:
            return True
    return False


@dataclass
class _Sampler:
    """Rejection sampler of floor slots shared by one candidate layout.

    Attributes:
        rng: Source of randomness.
        room_size: Room extent ``(x, y, z)`` in meters.
        inset: Camera intrinsics shrunk by ``_VIEW_MARGIN_PX`` on every
            side, the area a box center must project into.
        occupied: Footprints of the objects already placed.
        blocked: Rectangles no footprint may overlap.
    """

    rng: np.random.Generator
    room_size: tuple[float, float, float]
    inset: Intrinsics
    occupied: list[Disk]
    blocked: list[Rect]

    def sample(
        self, model: ObjectModel, views: Sequence[list[SE3]] = ()
    ) -> SE3 | None:
        """Draws a pose that fits the floor, optionally seen by cameras.

        Args:
            model: The model to place.
            views: Camera pose sets; the box center must project into at
                least one pose of every set.

        Returns:
            A floor pose, or None if no candidate fits.
        """
        room_x, room_y, _ = self.room_size
        for _ in range(_SLOT_TRIES):
            xy = np.array(
                [
                    self.rng.uniform(3.0, room_x - 0.5),
                    self.rng.uniform(0.5, room_y - 0.5),
                ]
            )
            pose = place(model, xy, self.rng.uniform(0.0, 2 * np.pi), 0.0)
            if self.fits(model, pose, views):
                return pose
        return None

    def sample_move(
        self,
        model: ObjectModel,
        pose_before: SE3,
        min_move_m: float,
        views: Sequence[list[SE3]],
    ) -> SE3 | None:
        """Draws a new pose for a model that must move at least a distance.

        Args:
            model: The model to move.
            pose_before: Its current floor pose.
            min_move_m: Minimum box-center displacement.
            views: Camera pose sets that must each see the box center.

        Returns:
            A floor pose, or None if no candidate fits.
        """
        center = world_box(model, pose_before).center[:2]
        yaw = np.arctan2(pose_before.rotation[1, 0], pose_before.rotation[0, 0])
        for _ in range(_SLOT_TRIES):
            angle = self.rng.uniform(0.0, 2 * np.pi)
            distance = min_move_m + self.rng.uniform(0.02, 0.5)
            yaw_change = (
                self.rng.uniform(-np.pi / 2, np.pi / 2)
                if self.rng.random() < 0.5
                else 0.0
            )
            candidate = place(
                model,
                center + distance * np.array([np.cos(angle), np.sin(angle)]),
                yaw + yaw_change,
                0.0,
            )
            if self.fits(model, candidate, views):
                return candidate
        return None

    def fits(
        self, model: ObjectModel, pose: SE3, views: Sequence[list[SE3]]
    ) -> bool:
        """Tests a candidate pose against room, neighbors, and cameras.

        Args:
            model: The model.
            pose: Candidate floor pose.
            views: Camera pose sets that must each see the box center.

        Returns:
            True if the candidate is acceptable.
        """
        box = world_box(model, pose)
        disk = footprint(box)
        return (
            disk_in_room(disk, self.room_size, _WALL_MARGIN)
            and all(disks_clear(disk, o, _FOOTPRINT_GAP) for o in self.occupied)
            and all(disk_clear_of_rect(disk, r) for r in self.blocked)
            and all(
                _visible_in_any(box.center, self.inset, poses)
                for poses in views
            )
        )


_ROLES = {
    "table": "table",
    "bottle": "bottle",
    "occluder": "shelf",
    "hidden": "stool",
    "zone_object": "crate",
    "moved": "chair",
    "added": "drum",
    "removed": "crate",
}
_STATUS_OF_ROLE = {
    "moved": ObjectStatus.MOVED,
    "added": ObjectStatus.ADDED,
    "removed": ObjectStatus.REMOVED,
}

_Poses = tuple[SE3 | None, SE3 | None]


@dataclass(frozen=True, eq=False)
class _Anchors:
    """The layout elements fixed before any free object is placed.

    Attributes:
        models: Object model of every role.
        pose_occluder: Pose of the cabinet that hides the stool.
        pose_hidden: Pose of the stool behind the cabinet.
        pose_zone_object: Pose of the crate inside the unobserved zone.
        zone: The unobserved zone.
        y_line: Room y of the line the rig and the occluder share.
        side: ``+1`` or ``-1``, the side the before cameras swing to.
    """

    models: dict[str, ObjectModel]
    pose_occluder: SE3
    pose_hidden: SE3
    pose_zone_object: SE3
    zone: OrientedBox
    y_line: float
    side: float


def _sample_anchors(rng: np.random.Generator, config: SceneConfig) -> _Anchors:
    _, room_y, _ = config.room_size
    y_line = room_y / 2 + rng.uniform(-0.2, 0.2)
    side = float(rng.choice([-1.0, 1.0]))
    models = {role: FACTORIES[label](rng) for role, label in _ROLES.items()}
    pose_occluder = place(
        models["occluder"],
        np.array([_X_OCCLUDER, y_line]),
        rng.uniform(-0.1, 0.1),
        0.0,
    )
    pose_hidden = place(
        models["hidden"],
        np.array([_X_HIDDEN, y_line + rng.uniform(-0.15, 0.15)]),
        rng.uniform(0.0, 2 * np.pi),
        0.0,
    )
    zone_y = rng.uniform(0.3, room_y - 0.3 - _ZONE_SIZE_Y)
    zone = OrientedBox(
        np.array(
            [sum(_ZONE_X) / 2, zone_y + _ZONE_SIZE_Y / 2, _ZONE_SIZE_Z / 2]
        ),
        np.array([_ZONE_X[1] - _ZONE_X[0], _ZONE_SIZE_Y, _ZONE_SIZE_Z]),
        np.eye(3),
    )
    pose_zone_object = place(
        models["zone_object"],
        zone.center[:2] + rng.uniform(-0.05, 0.05, size=2),
        rng.uniform(0.0, 2 * np.pi),
        0.0,
    )
    return _Anchors(
        models, pose_occluder, pose_hidden, pose_zone_object, zone, y_line, side
    )


def _camera_height(rng: np.random.Generator) -> float:
    return float(rng.uniform(1.4, 1.6))


def _build_paths(
    rng: np.random.Generator, config: SceneConfig, anchors: _Anchors
) -> tuple[list[SE3], list[SE3]]:
    """Samples the before and after camera poses.

    Args:
        rng: Source of randomness.
        config: Generator parameters.
        anchors: The fixed layout elements the paths look at.

    Returns:
        The before poses and the after poses.
    """
    y_line, side = anchors.y_line, anchors.side
    hidden_center = world_box(
        anchors.models["hidden"], anchors.pose_hidden
    ).center
    zone_center = world_box(
        anchors.models["zone_object"], anchors.pose_zone_object
    ).center
    after_path = [
        Waypoint(
            np.array([_RIG_X, y_line + k * 0.25, _camera_height(rng)]),
            np.array(
                [_X_HIDDEN + 0.5, y_line + k * 1.0, rng.uniform(0.3, 0.6)]
            ),
        )
        for k in (-1, 0, 1)
    ]
    before_path = [
        Waypoint(
            np.array([_RIG_X, y_line - 0.3 * side, _camera_height(rng)]),
            np.array([_X_HIDDEN + 0.8, y_line - 1.4 * side, 0.5]),
        ),
        Waypoint(
            np.array([_RIG_X, y_line + 0.3 * side, _camera_height(rng)]),
            np.array([_X_HIDDEN + 0.8, y_line + 1.0 * side, 0.5]),
        ),
        Waypoint(
            np.array(
                [_X_HIDDEN, y_line + _SIDE_OFFSET * side, _camera_height(rng)]
            ),
            hidden_center,
        ),
        Waypoint(
            np.array([3.0, y_line + 1.0 * side, _camera_height(rng)]),
            zone_center,
        ),
    ]
    n = config.frames_per_epoch
    after_poses = sweep(
        after_path, n, rng, _POSITION_JITTER_M, _TARGET_JITTER_M
    )
    before_poses = sweep(
        before_path, n, rng, _POSITION_JITTER_M, _TARGET_JITTER_M
    )
    return before_poses, after_poses


def _place_free_objects(
    rng: np.random.Generator,
    config: SceneConfig,
    anchors: _Anchors,
    before_poses: list[SE3],
    after_poses: list[SE3],
) -> dict[str, _Poses] | None:
    """Places the table, bottle, and the three changed objects.

    Args:
        rng: Source of randomness.
        config: Generator parameters.
        anchors: The fixed layout elements to keep clear of.
        before_poses: Before camera poses.
        after_poses: After camera poses.

    Returns:
        The ``(pose_before, pose_after)`` of each placed role, or None if
        some object finds no slot.
    """
    models, y_line = anchors.models, anchors.y_line
    corridor = Rect(_RIG_X - 0.5, _X_OCCLUDER + 0.4, y_line - 0.9, y_line + 0.9)
    channel = (
        Rect(_X_HIDDEN - 0.7, _X_HIDDEN + 0.7, y_line, config.room_size[1])
        if anchors.side > 0
        else Rect(_X_HIDDEN - 0.7, _X_HIDDEN + 0.7, 0.0, y_line)
    )
    shadow = Rect(_X_OCCLUDER, config.room_size[0], y_line - 0.9, y_line + 0.9)
    sampler = _Sampler(
        rng,
        config.room_size,
        _inset(config.intrinsics, _VIEW_MARGIN_PX),
        occupied=[
            footprint(world_box(models["occluder"], anchors.pose_occluder)),
            footprint(world_box(models["hidden"], anchors.pose_hidden)),
            footprint(
                world_box(models["zone_object"], anchors.pose_zone_object)
            ),
        ],
        blocked=[corridor, channel],
    )

    pose_table = sampler.sample(models["table"])
    if pose_table is None:
        return None
    table_box = world_box(models["table"], pose_table)
    sampler.occupied.append(footprint(table_box))
    top = float(table_box.center[2] + table_box.extent[2] / 2)
    slack = table_box.extent[:2] / 2 - 0.15
    on_top = table_box.center[:2] + pose_table.rotation[:2, :2] @ (
        rng.uniform(-1.0, 1.0, size=2) * slack
    )
    pose_bottle = place(models["bottle"], on_top, 0.0, top)
    # The table may stand in the occluder's shadow; the objects below may not.
    sampler.blocked.append(shadow)

    pose_moved_before = sampler.sample(
        models["moved"], (before_poses, after_poses)
    )
    if pose_moved_before is None:
        return None
    pose_moved_after = sampler.sample_move(
        models["moved"],
        pose_moved_before,
        config.min_move_m,
        (after_poses, before_poses),
    )
    if pose_moved_after is None:
        return None
    sampler.occupied.append(
        footprint(world_box(models["moved"], pose_moved_before))
    )
    sampler.occupied.append(
        footprint(world_box(models["moved"], pose_moved_after))
    )

    pose_added = sampler.sample(models["added"], (after_poses, before_poses))
    if pose_added is None:
        return None
    sampler.occupied.append(footprint(world_box(models["added"], pose_added)))
    pose_removed = sampler.sample(
        models["removed"], (before_poses, after_poses)
    )
    if pose_removed is None:
        return None
    return {
        "table": (pose_table, pose_table),
        "bottle": (pose_bottle, pose_bottle),
        "moved": (pose_moved_before, pose_moved_after),
        "added": (None, pose_added),
        "removed": (pose_removed, None),
    }


def _assemble_objects(
    anchors: _Anchors, placed: dict[str, _Poses]
) -> dict[str, SyntheticObject]:
    """Builds the objects of a layout, indexed in ``_ROLES`` order.

    Args:
        anchors: The fixed layout elements.
        placed: Poses of the free objects, by role.

    Returns:
        The objects keyed by role.
    """
    poses = {
        **placed,
        "occluder": (anchors.pose_occluder, anchors.pose_occluder),
        "hidden": (anchors.pose_hidden, anchors.pose_hidden),
        "zone_object": (anchors.pose_zone_object, anchors.pose_zone_object),
    }
    objects: dict[str, SyntheticObject] = {}
    for index, role in enumerate(_ROLES):
        before, after = poses[role]
        objects[role] = _make_object(
            index,
            anchors.models[role],
            _STATUS_OF_ROLE.get(role, ObjectStatus.UNCHANGED),
            before,
            after,
            objects["table"] if role == "bottle" else None,
        )
    return objects


def _try_generate(
    seed: int, rng: np.random.Generator, config: SceneConfig
) -> SyntheticScenePair | None:
    anchors = _sample_anchors(rng, config)
    before_poses, after_poses = _build_paths(rng, config, anchors)
    if _zone_in_any_frustum(anchors.zone, config.intrinsics, after_poses):
        return None
    placed = _place_free_objects(
        rng, config, anchors, before_poses, after_poses
    )
    if placed is None:
        return None
    by_role = _assemble_objects(anchors, placed)
    objects = list(by_role.values())
    poses = {Epoch.BEFORE: before_poses, Epoch.AFTER: after_poses}
    if not _cameras_are_clear(config.room_size, objects, poses):
        return None
    pair = SyntheticScenePair(
        seed=seed,
        config=config,
        objects=tuple(objects),
        before=_render_epoch(objects, Epoch.BEFORE, config, before_poses),
        after=_render_epoch(objects, Epoch.AFTER, config, after_poses),
        unobserved_zone=anchors.zone,
        zone_seen_in=Epoch.BEFORE,
        occluder_id=by_role["occluder"].object_id,
        occluded_object_ids=(by_role["hidden"].object_id,),
        zone_object_ids=(by_role["zone_object"].object_id,),
    )
    return pair if _renders_confirm_ground_truth(pair) else None


def _zone_in_any_frustum(
    zone: OrientedBox, intrinsics: Intrinsics, poses: list[SE3]
) -> bool:
    local = np.linspace(-0.5, 0.5, 7)
    grid = np.array(list(itertools.product(local, local, local)))
    samples = SE3(zone.rotation, zone.center).apply(grid * zone.extent)
    return any(sees_points(samples, intrinsics, pose).any() for pose in poses)


def _render_epoch(
    objects: list[SyntheticObject],
    epoch: Epoch,
    config: SceneConfig,
    poses: list[SE3],
) -> SyntheticEpoch:
    meshes = [
        PlacedMesh(
            o.index,
            world_vertices(o.model, o.pose(epoch)),
            o.model.triangles,
            o.model.color,
        )
        for o in objects
        if o.exists_in(epoch)
    ]
    scene = RenderScene(config.room_size, meshes)
    intrinsics = config.intrinsics
    frames: list[Frame] = []
    instance_maps: list[NDArray[np.int32]] = []
    for index, pose in enumerate(poses):
        view = scene.render(intrinsics, pose)
        frames.append(
            Frame(
                epoch=epoch,
                index=index,
                timestamp_s=index / config.fps,
                intrinsics=intrinsics,
                world_from_camera=pose,
                image=view.rgb,
                depth=view.depth,
            )
        )
        instance_maps.append(view.instance)
    return SyntheticEpoch(epoch, frames, instance_maps)


def _pixel_counts(data: SyntheticEpoch, n_objects: int) -> NDArray[np.int64]:
    """Counts the pixels of every object over all frames of an epoch.

    Args:
        data: The rendered epoch.
        n_objects: Number of objects in the pair.

    Returns:
        Array of length ``n_objects``; entry ``i`` is the pixel count of
        the object with index ``i``.
    """
    counts = np.zeros(n_objects + 1, dtype=np.int64)
    for instance in data.instance_maps:
        counts += np.bincount(instance.ravel() + 1, minlength=n_objects + 1)
    return counts[1:]


def _cameras_are_clear(
    room_size: tuple[float, float, float],
    objects: list[SyntheticObject],
    poses: dict[Epoch, list[SE3]],
) -> bool:
    room = np.array(room_size)
    for epoch in Epoch:
        centers = np.stack([pose.translation for pose in poses[epoch]])
        if np.any(centers < _CAMERA_MARGIN) or np.any(
            centers > room - _CAMERA_MARGIN
        ):
            return False
        for obj in objects:
            if not obj.exists_in(epoch):
                continue
            grown = _grown(obj.box(epoch), 2 * _CAMERA_CLEARANCE)
            if np.any(grown.contains(centers)):
                return False
    return True


def _occluder_covers_hidden(
    pair: SyntheticScenePair, hidden: SyntheticObject, occluder: SyntheticObject
) -> bool:
    box = hidden.box(Epoch.AFTER)
    after = pair.after
    for frame, instance in zip(after.frames, after.instance_maps, strict=True):
        uv, depth = project(
            box.corners(), frame.intrinsics, frame.world_from_camera
        )
        center_uv, center_depth = project(
            box.center[None, :], frame.intrinsics, frame.world_from_camera
        )
        if (
            np.any(depth <= 0)
            or not in_image(center_uv, center_depth, frame.intrinsics)[0]
        ):
            return False
        lo = np.clip(np.floor(uv.min(axis=0)), 0, None).astype(int)
        hi = np.ceil(uv.max(axis=0)).astype(int) + 1
        patch = instance[lo[1] : hi[1], lo[0] : hi[0]]
        if patch.size == 0 or not np.all(patch == occluder.index):
            return False
    return True


def _seen_through(box: OrientedBox, frames: list[Frame]) -> bool:
    """Tells whether cameras saw through a box's location.

    Its center and its corners shrunk by ``_BOX_INSET`` about the center
    count as seen through in a frame when they project into the image
    and the rendered depth there exceeds their own by
    ``_SEEN_THROUGH_MARGIN_M``, so nothing nearer blocks the view and the
    ray continues past them. The center must be seen through in some
    frame, and at least ``_MIN_SEEN_CORNERS`` of the corners in some
    frame each. The shrunk corners keep the floor-level ones off the
    floor surface, where the depth margin could never hold.

    Args:
        box: The box in world coordinates.
        frames: Rendered frames of the epoch that must see through it.

    Returns:
        True if the location was seen through.
    """
    corners = box.center + _BOX_INSET * (box.corners() - box.center)
    points = np.vstack([box.center, corners])
    through = np.zeros(len(points), dtype=bool)
    for frame in frames:
        assert frame.depth is not None
        uv, depth = project(points, frame.intrinsics, frame.world_from_camera)
        inside = in_image(uv, depth, frame.intrinsics)
        cols, rows = np.rint(uv[inside]).astype(int).T
        rendered = frame.depth[rows, cols]
        through[inside] |= rendered > depth[inside] + _SEEN_THROUGH_MARGIN_M
    return bool(through[0] and through[1:].mean() >= _MIN_SEEN_CORNERS)


def _changed_objects_are_seen_through(pair: SyntheticScenePair) -> bool:
    for obj in pair.objects:
        if obj.status == ObjectStatus.UNCHANGED:
            continue
        for epoch in Epoch:
            if not obj.exists_in(epoch):
                continue
            other = Epoch.AFTER if epoch == Epoch.BEFORE else Epoch.BEFORE
            if not _seen_through(obj.box(epoch), pair.epoch_data(other).frames):
                return False
    return True


def _zone_hits(zone: OrientedBox, frames: list[Frame]) -> int:
    """Counts the depth pixels that unproject into a box.

    Args:
        zone: The box in world coordinates.
        frames: Rendered frames of one epoch.

    Returns:
        The number of valid depth pixels over all frames whose world point
        lies inside ``zone``.
    """
    hits = 0
    for frame in frames:
        assert frame.depth is not None
        points = unproject_depth(
            frame.depth, frame.intrinsics, frame.world_from_camera
        )
        hits += int(np.count_nonzero(zone.contains(points)))
    return hits


def _zone_seen_only_before(pair: SyntheticScenePair) -> bool:
    zone = pair.unobserved_zone
    if _zone_hits(_grown(zone, 0.1), pair.after.frames):
        return False
    return _zone_hits(zone, pair.before.frames) >= _MIN_PIXELS


def _renders_confirm_ground_truth(pair: SyntheticScenePair) -> bool:
    n = len(pair.objects)
    counts = {
        epoch: _pixel_counts(pair.epoch_data(epoch), n) for epoch in Epoch
    }
    for obj in pair.objects:
        if obj.status == ObjectStatus.UNCHANGED:
            continue
        for epoch in Epoch:
            if obj.exists_in(epoch) and counts[epoch][obj.index] < _MIN_PIXELS:
                return False
    occluder = pair.object_by_id(pair.occluder_id)
    hidden = pair.object_by_id(pair.occluded_object_ids[0])
    zone_object = pair.object_by_id(pair.zone_object_ids[0])
    return (
        all(
            counts[Epoch.BEFORE][o.index] >= _MIN_PIXELS
            for o in (occluder, hidden, zone_object)
        )
        and counts[Epoch.AFTER][occluder.index] >= _MIN_PIXELS
        and _occluder_covers_hidden(pair, hidden, occluder)
        and _changed_objects_are_seen_through(pair)
        and _zone_seen_only_before(pair)
    )
