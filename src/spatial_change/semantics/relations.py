"""Spatial relations engine: from a detected change to a ``ChangeFact``.

The engine is pure geometry. It takes a ``Change`` plus static, labeled
landmarks and produces a structured fact (supporting surface, nearby
landmarks, direction of motion, distance and rotation with their rounded
text) that a describer turns into language.

Conventions:
    * The room frame is z-up with the floor at ``z = 0``, in meters. A
      floor-alignment stage guarantees this; the engine assumes it.
    * Directions are relative to a reference viewpoint, an ``SE3``
      ``world_from_camera`` in the OpenCV convention (x right, y down, z
      forward), because the horizontal axes of the room frame are arbitrary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict

from spatial_change.domain import SE3, Change, ChangeType, OrientedBox

_EPS = 1e-9
_MIN_FORWARD_NORM = 0.1
_TOWARD_COSINE = 0.8
_SECTORS = (
    "farther",
    "farther and to the right",
    "to the right",
    "closer and to the right",
    "closer",
    "closer and to the left",
    "to the left",
    "farther and to the left",
)


class _FactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


@dataclass(frozen=True)
class Landmark:
    """A static, labeled object used as spatial context.

    Attributes:
        landmark_id: Identifier, unique among the landmarks.
        label: Object class name, such as ``"table"``.
        box: Oriented bounding box in the room frame.
    """

    landmark_id: str
    label: str
    box: OrientedBox


@dataclass(frozen=True)
class RelationConfig:
    """Tunable thresholds of the relations engine.

    Attributes:
        support_tolerance_m: Maximum gap between an object's bottom and a
            supporting surface.
        next_to_m: Maximum surface distance for the relation "next to".
        near_m: Maximum surface distance for the relation "near".
        max_landmarks: Maximum number of nearby landmarks reported, an
            integer of at least 1.
        min_displacement_m: Smallest displacement worth describing.
        min_rotation_deg: Smallest rotation worth describing.
        vertical_ratio: Fraction in ``(0, 1]``. Vertical motion is reported
            when it is at least this fraction of the horizontal motion, and
            horizontal motion is reported when it is at least this fraction
            of the vertical one.
    """

    support_tolerance_m: float = 0.05
    next_to_m: float = 0.5
    near_m: float = 1.5
    max_landmarks: int = 3
    min_displacement_m: float = 0.05
    min_rotation_deg: float = 10.0
    vertical_ratio: float = 0.5

    def __post_init__(self) -> None:
        """Checks that every threshold is valid.

        Raises:
            ValueError: If a threshold is not a positive finite number,
                ``max_landmarks`` is not an integer of at least 1,
                ``vertical_ratio`` exceeds 1, or ``next_to_m`` exceeds
                ``near_m``.
        """
        for name in (
            "support_tolerance_m",
            "next_to_m",
            "near_m",
            "min_displacement_m",
            "min_rotation_deg",
            "vertical_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        count: object = self.max_landmarks
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("max_landmarks must be an integer of at least 1")
        if self.vertical_ratio > 1:
            raise ValueError("vertical_ratio must not exceed 1")
        if self.next_to_m > self.near_m:
            raise ValueError("next_to_m must not exceed near_m")


DEFAULT_CONFIG = RelationConfig()


def _check_amount(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")


def _count_steps(value: float, step: str) -> int:
    """Counts the ``step`` units nearest to ``value``, ties rounding up.

    A tie is decided on the shortest decimal form of ``value``, so 1.25 and
    1.15 round alike although their binary values lie on opposite sides.
    """
    units = Decimal(repr(float(value))) / Decimal(step)
    return int(units.to_integral_value(rounding=ROUND_HALF_UP))


def _round_steps(value: float, step: str) -> int | None:
    """Counts the ``step`` units nearest to ``value``, as ``_count_steps``.

    Returns:
        The count, or ``None`` if ``value`` is positive but rounds to zero,
        which the formatters word as "under".
    """
    steps = _count_steps(value, step)
    return None if steps == 0 and value > 0 else steps


def format_distance(meters: float) -> str:
    """Phrases a distance in metric units.

    Ties round up on the decimal value as written, so 0.025 gives
    ``"about 5 cm"`` and 1.25 gives ``"about 1.3 m"``.

    Args:
        meters: A non-negative distance.

    Returns:
        ``"about 80 cm"`` below one meter (nearest 5 cm), otherwise
        ``"about 1.2 m"`` (one decimal). Distances that round to 100 cm,
        from 0.975 m up, are phrased in meters. A positive distance that
        rounds to 0 cm, below 0.025 m, gives ``"under 5 cm"``; exactly 0
        gives ``"about 0 cm"``.

    Raises:
        ValueError: If ``meters`` is negative or not finite.
    """
    _check_amount(meters, "meters")
    steps = _round_steps(meters, "0.05")
    if steps is None:
        return "under 5 cm"
    centimeters = 5 * steps
    if centimeters < 100:
        return f"about {centimeters} cm"
    tenths = _count_steps(meters, "0.1")
    return f"about {tenths // 10}.{tenths % 10} m"


def format_rotation(degrees: float) -> str:
    """Phrases a rotation angle.

    Args:
        degrees: A non-negative angle.

    Returns:
        The angle rounded to the nearest 5 degrees, ties up, such as
        ``"about 45 degrees"``. A positive angle that rounds to 0, below
        2.5 degrees, gives ``"under 5 degrees"``; exactly 0 gives
        ``"about 0 degrees"``.

    Raises:
        ValueError: If ``degrees`` is negative or not finite.
    """
    _check_amount(degrees, "degrees")
    steps = _round_steps(degrees, "5")
    if steps is None:
        return "under 5 degrees"
    return f"about {5 * steps} degrees"


def _top(box: OrientedBox) -> float:
    return float(box.corners()[:, 2].max())


def _footprint_area(box: OrientedBox) -> float:
    """Returns the area of the box projected onto the floor plane."""
    ex, ey, ez = box.extent
    nx, ny, nz = np.abs(box.rotation[2])
    return float(ey * ez * nx + ex * ez * ny + ex * ey * nz)


def _covers_xy(box: OrientedBox, xy: NDArray[np.float64]) -> bool:
    """Tests whether the vertical line through ``xy`` meets the box."""
    origin = box.rotation.T @ (np.array([xy[0], xy[1], 0.0]) - box.center)
    slope = box.rotation[2]
    half = box.extent / 2.0
    low, high = -math.inf, math.inf
    for o, s, h in zip(origin, slope, half, strict=True):
        if abs(s) < _EPS:
            if abs(o) > h + _EPS:
                return False
            continue
        a, b = (-h - o) / s, (h - o) / s
        low, high = max(low, min(a, b)), min(high, max(a, b))
    return low <= high + _EPS


def _find_support(
    box: OrientedBox,
    landmarks: Sequence[Landmark],
    config: RelationConfig,
) -> tuple[str | None, Landmark | None]:
    """Finds the support label of an object and the landmark behind it.

    Returns:
        ``("floor", None)``, ``(label, landmark)`` for a supporting
        landmark, or ``(None, None)`` if nothing supports the object.
    """
    bottom = float(box.corners()[:, 2].min())
    if abs(bottom) <= config.support_tolerance_m:
        return "floor", None
    candidates = [
        lm
        for lm in landmarks
        if abs(_top(lm.box) - bottom) <= config.support_tolerance_m
        and _covers_xy(lm.box, box.center[:2])
    ]
    if not candidates:
        return None, None
    support = min(
        candidates,
        key=lambda lm: (
            -_top(lm.box),
            _footprint_area(lm.box),
            lm.landmark_id,
        ),
    )
    return support.label, support


def supporting_surface(
    box: OrientedBox,
    landmarks: Sequence[Landmark],
    config: RelationConfig,
) -> str | None:
    """Finds what an object rests on.

    Args:
        box: The object's box.
        landmarks: Static landmarks that may support the object.
        config: Thresholds.

    Returns:
        ``"floor"`` if the lowest corner is within the support tolerance of
        ``z = 0``; otherwise the label of the landmark whose top face is
        within tolerance of the object's bottom and whose footprint contains
        the object's center in xy (highest top, then smallest footprint,
        then ``landmark_id`` break ties); otherwise ``None`` (wall-mounted,
        floating, or unknown).
    """
    return _find_support(box, landmarks, config)[0]


class LandmarkRelation(_FactModel):
    """How close an object is to one landmark.

    Attributes:
        landmark_id: Identifier of the landmark.
        label: Landmark class name.
        distance_m: Distance from the object's center to the landmark's
            surface, zero if the center is inside the landmark.
        relation: ``"next to"`` within ``next_to_m``, else ``"near"``.
    """

    landmark_id: str
    label: str
    distance_m: float
    relation: Literal["next to", "near"]


def _closest_point(point: ArrayLike, box: OrientedBox) -> NDArray[np.float64]:
    local = box.rotation.T @ (np.asarray(point, dtype=np.float64) - box.center)
    half = box.extent / 2.0
    return box.rotation @ np.clip(local, -half, half) + box.center


def distance_to_box(point: ArrayLike, box: OrientedBox) -> float:
    """Measures the distance from a point to the surface of a box.

    Args:
        point: World point of shape ``(3,)``.
        box: The box.

    Returns:
        The Euclidean distance to the nearest point of the solid box, zero
        if the point is inside or on it.
    """
    delta = np.asarray(point, dtype=np.float64) - _closest_point(point, box)
    return float(np.linalg.norm(delta))


def _surroundings(
    box: OrientedBox | None,
    landmarks: Sequence[Landmark],
    config: RelationConfig,
) -> tuple[str | None, list[LandmarkRelation]]:
    """Finds the support and the nearby landmarks of an optional box.

    The landmark that supports the box is left out of the nearby ones, since
    it is already expressed as the support.

    Returns:
        The support label and the nearby landmarks, ``(None, [])`` if there
        is no box.
    """
    if box is None:
        return None, []
    surface, support = _find_support(box, landmarks, config)
    found = []
    for lm in landmarks:
        if lm is support:
            continue
        distance = distance_to_box(box.center, lm.box)
        if distance <= config.near_m:
            found.append((distance, lm))
    found.sort(key=lambda item: (item[0], item[1].landmark_id))
    return surface, [
        LandmarkRelation(
            landmark_id=lm.landmark_id,
            label=lm.label,
            distance_m=distance,
            relation="next to" if distance <= config.next_to_m else "near",
        )
        for distance, lm in found[: config.max_landmarks]
    ]


def nearest_landmarks(
    box: OrientedBox,
    landmarks: Sequence[Landmark],
    config: RelationConfig,
) -> list[LandmarkRelation]:
    """Lists the landmarks close to an object.

    Args:
        box: The object's box.
        landmarks: Static landmarks.
        config: Thresholds.

    Returns:
        At most ``max_landmarks`` landmarks within ``near_m`` of the
        object's center, sorted by distance and then ``landmark_id``. The
        landmark the object rests on is excluded, since it is already
        expressed as the support.
    """
    return _surroundings(box, landmarks, config)[1]


@dataclass(frozen=True)
class Direction:
    """The direction an object moved, relative to a viewpoint.

    Attributes:
        text: Phrase such as ``"closer and to the left"`` or ``"up"``, or
            ``None`` if no word applies (no viewpoint, or a camera looking
            straight up or down, and no vertical motion). A part of the
            motion is worded when it is large enough relative to the other
            part; see ``describe_direction``.
        toward: Label of the landmark the object moved toward, if any. It
            does not depend on the viewpoint.
    """

    text: str | None
    toward: str | None


def _horizontal_sector(
    horizontal: NDArray[np.float64], viewpoint: SE3
) -> str | None:
    """Names the 45 degree sector of a floor-plane motion, if defined."""
    forward = viewpoint.rotation[:2, 2]
    norm = float(np.linalg.norm(forward))
    if norm < _MIN_FORWARD_NORM:
        return None
    forward = forward / norm
    right = np.array([forward[1], -forward[0]])
    angle = math.degrees(math.atan2(horizontal @ right, horizontal @ forward))
    return _SECTORS[math.floor((angle + 22.5) / 45.0) % 8]


def _moved_toward(
    before: OrientedBox,
    after: OrientedBox,
    displacement: NDArray[np.float64],
    length: float,
    landmarks: Sequence[Landmark],
    config: RelationConfig,
) -> str | None:
    """Names the landmark an object moved straight toward, if any.

    ``length`` is the positive norm of ``displacement``. The landmark with
    the largest progress wins, ties broken by ``landmark_id``.
    """
    candidates = []
    for lm in landmarks:
        to_landmark = _closest_point(before.center, lm.box) - before.center
        gap = float(np.linalg.norm(to_landmark))
        if gap < _EPS:
            continue
        progress = gap - distance_to_box(after.center, lm.box)
        cosine = float(displacement @ to_landmark) / (length * gap)
        if progress < config.min_displacement_m or cosine <= _TOWARD_COSINE:
            continue
        candidates.append((-progress, lm.landmark_id, lm.label))
    best = min(candidates, default=None)
    return None if best is None else best[2]


def _is_worded(part: float, other: float, config: RelationConfig) -> bool:
    """Tells whether one part of a move is worth a word.

    The part must be at least ``vertical_ratio`` times the other part, and
    either reach ``min_displacement_m`` or be at least as large as the other
    part, so the dominant part of any describable move is never dropped.
    """
    return part >= config.vertical_ratio * other and (
        part >= config.min_displacement_m or part >= other
    )


def _describe_move(
    before: OrientedBox,
    after: OrientedBox,
    displacement: NDArray[np.float64],
    length: float,
    viewpoint: SE3 | None,
    landmarks: Sequence[Landmark],
    config: RelationConfig,
) -> Direction | None:
    """Describes a move whose displacement and its norm are already known.

    Args:
        before: The object's box before the move.
        after: The object's box after the move.
        displacement: Vector from the before center to the after center.
        length: Norm of ``displacement``.
        viewpoint: Reference pose, or ``None`` if there is none.
        landmarks: Static landmarks, used for the ``toward`` hint.
        config: Thresholds.

    Returns:
        The direction, or ``None`` if ``length`` is below
        ``min_displacement_m``.
    """
    if length < config.min_displacement_m:
        return None
    horizontal = float(np.linalg.norm(displacement[:2]))
    dz = float(displacement[2])
    words = []
    if viewpoint is not None and _is_worded(horizontal, abs(dz), config):
        sector = _horizontal_sector(displacement[:2], viewpoint)
        if sector is not None:
            words.append(sector)
    if _is_worded(abs(dz), horizontal, config):
        words.append("up" if dz > 0 else "down")
    return Direction(
        " and ".join(words) if words else None,
        _moved_toward(before, after, displacement, length, landmarks, config),
    )


def describe_direction(
    before: OrientedBox,
    after: OrientedBox,
    viewpoint: SE3 | None,
    landmarks: Sequence[Landmark],
    config: RelationConfig,
) -> Direction | None:
    """Describes how an object moved, relative to a viewpoint.

    Only the viewpoint's optical axis (z) is used: its projection onto the
    floor plane is "forward" and the floor-plane perpendicular to it is
    "right", so camera roll has no effect. The horizontal motion is
    classified into eight sectors of 45 degrees.

    The horizontal and the vertical part of the move are each worded when
    they are at least ``vertical_ratio`` times the other part and either
    reach ``min_displacement_m`` or are at least as large as the other part.
    The dominant part of a move long enough to describe is therefore always
    worded, even when it is below ``min_displacement_m`` (for example a
    diagonal move of 0.04 m both sideways and up); a smaller part below
    ``min_displacement_m`` is left out.

    Args:
        before: The object's box before the move.
        after: The object's box after the move.
        viewpoint: Reference ``world_from_camera`` pose in the OpenCV
            convention, or ``None`` if there is no reference view.
        landmarks: Static landmarks, used for the ``toward`` hint.
        config: Thresholds.

    Returns:
        The direction, or ``None`` if the object moved less than
        ``min_displacement_m``. If there is no viewpoint, or the camera
        looks straight up or down, only the vertical part is described, and
        ``text`` is ``None`` when there is none. ``toward`` is computed
        whenever the move is long enough, with or without words.
    """
    displacement = after.center - before.center
    return _describe_move(
        before,
        after,
        displacement,
        float(np.linalg.norm(displacement)),
        viewpoint,
        landmarks,
        config,
    )


class ChangeFact(_FactModel):
    """Structured spatial facts about one change, ready to be described.

    Numbers sit next to their text so generated language can be checked
    against the geometry.

    Attributes:
        change_id: Identifier of the source change.
        type: Kind of change.
        label: Object class name.
        support_before: What the object rested on before, if known.
        support_after: What the object rests on after, if known.
        near_before: Landmarks near the object before.
        near_after: Landmarks near the object after.
        direction: Direction of motion relative to the viewpoint.
        toward: Label of the landmark the object moved toward. It does not
            depend on the viewpoint.
        displacement_m: Distance between the box centers, in meters, so it
            always agrees with ``direction`` and ``toward``; ``None`` unless
            both boxes exist.
        displacement_text: Rounded phrase for ``displacement_m``, omitted
            below ``min_displacement_m``.
        rotation_deg: Rotation magnitude in degrees, as reported by the
            change. It is not recomputed from the boxes, whose axes cannot
            recover it reliably because of box symmetries.
        rotation_text: Rounded phrase for ``rotation_deg``, omitted below
            ``min_rotation_deg``.
    """

    change_id: str
    type: ChangeType
    label: str
    support_before: str | None
    support_after: str | None
    near_before: list[LandmarkRelation]
    near_after: list[LandmarkRelation]
    direction: str | None
    toward: str | None
    displacement_m: float | None
    displacement_text: str | None
    rotation_deg: float | None
    rotation_text: str | None


def build_change_fact(
    change: Change,
    landmarks: Sequence[Landmark],
    viewpoint: SE3 | None,
    config: RelationConfig = DEFAULT_CONFIG,
) -> ChangeFact:
    """Builds the structured spatial facts of a change.

    Args:
        change: The detected change.
        landmarks: Static, labeled landmarks in the room frame.
        viewpoint: Reference ``world_from_camera`` pose for directions, or
            ``None`` if there is none.
        config: Thresholds.

    Returns:
        The fact. Before-relations use ``before_box`` and after-relations
        use ``after_box``; direction, toward, and displacement need both
        boxes. The displacement is the distance between the box centers,
        not ``change.displacement_m``, so the numbers and the direction
        come from the same geometry; the rotation is ``change.rotation_deg``.

    Raises:
        ValueError: If ``change`` has neither a before nor an after box.
    """
    before = None if change.before_box is None else change.before_box.to_box()
    after = None if change.after_box is None else change.after_box.to_box()
    if before is None and after is None:
        raise ValueError(f"{change.type.value} needs at least one box")
    support_before, near_before = _surroundings(before, landmarks, config)
    support_after, near_after = _surroundings(after, landmarks, config)

    direction = None
    displacement_m = None
    displacement_text = None
    if before is not None and after is not None:
        displacement = after.center - before.center
        displacement_m = float(np.linalg.norm(displacement))
        direction = _describe_move(
            before,
            after,
            displacement,
            displacement_m,
            viewpoint,
            landmarks,
            config,
        )
        if displacement_m >= config.min_displacement_m:
            displacement_text = format_distance(displacement_m)
    rotation_text = None
    if (
        change.rotation_deg is not None
        and change.rotation_deg >= config.min_rotation_deg
    ):
        rotation_text = format_rotation(change.rotation_deg)

    return ChangeFact(
        change_id=change.change_id,
        type=change.type,
        label=change.label,
        support_before=support_before,
        support_after=support_after,
        near_before=near_before,
        near_after=near_after,
        direction=None if direction is None else direction.text,
        toward=None if direction is None else direction.toward,
        displacement_m=displacement_m,
        displacement_text=displacement_text,
        rotation_deg=change.rotation_deg,
        rotation_text=rotation_text,
    )
