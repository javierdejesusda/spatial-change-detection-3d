"""Procedural room point clouds shared by the geometry and alignment tests.

The static structure is a 6 x 5 m floor at ``z = 0``, four walls 2.5 m high,
a table top at 0.75 m, and one fixed cabinet. Movable boxes stand on the
floor. Every call samples with the given generator, so two calls with
different generators give independent samplings of the same surfaces.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ROOM_X = 6.0
ROOM_Y = 5.0
ROOM_Z = 2.5
TABLE_HEIGHT = 0.75
# Origin and the two edge vectors of a rectangle, in the room frame.
Face = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class Box:
    """An axis-aligned box standing on the floor.

    Attributes:
        x: Center x in meters.
        y: Center y in meters.
        size: Extent ``(sx, sy, sz)`` in meters.
    """

    x: float
    y: float
    size: tuple[float, float, float]


CABINET = Box(5.2, 4.2, (0.8, 0.8, 1.2))


def face(
    origin: tuple[float, float, float],
    u: tuple[float, float, float],
    v: tuple[float, float, float],
) -> Face:
    """Builds a rectangle from its origin and two edge vectors."""
    return np.array(origin), np.array(u), np.array(v)


def box_faces(box: Box) -> list[Face]:
    """Returns the top and four side faces of a box.

    Args:
        box: The box.

    Returns:
        Five rectangles as ``(origin, u, v)``.
    """
    sx, sy, sz = box.size
    x0, y0 = box.x - sx / 2, box.y - sy / 2
    return [
        face((x0, y0, sz), (sx, 0, 0), (0, sy, 0)),
        face((x0, y0, 0), (sx, 0, 0), (0, 0, sz)),
        face((x0, y0 + sy, 0), (sx, 0, 0), (0, 0, sz)),
        face((x0, y0, 0), (0, sy, 0), (0, 0, sz)),
        face((x0 + sx, y0, 0), (0, sy, 0), (0, 0, sz)),
    ]


def static_faces() -> list[Face]:
    """Returns the fixed room surfaces: floor, walls, table top, cabinet."""
    faces = [
        face((0, 0, 0), (ROOM_X, 0, 0), (0, ROOM_Y, 0)),
        face((0, 0, 0), (0, ROOM_Y, 0), (0, 0, ROOM_Z)),
        face((0, 0, 0), (ROOM_X, 0, 0), (0, 0, ROOM_Z)),
        face((ROOM_X, 0, 0), (0, ROOM_Y, 0), (0, 0, ROOM_Z)),
        face((0, ROOM_Y, 0), (ROOM_X, 0, 0), (0, 0, ROOM_Z)),
        face((1.0, 3.0, TABLE_HEIGHT), (1.4, 0, 0), (0, 1.0, 0)),
    ]
    return faces + box_faces(CABINET)


def sample_faces(
    rng: np.random.Generator,
    faces: Sequence[Face],
    count: int | None = None,
    density: float = 150.0,
    noise: float = 0.002,
) -> NDArray[np.float64]:
    """Samples points uniformly on rectangles.

    Args:
        rng: Source of randomness.
        faces: Rectangles as ``(origin, u, v)``.
        count: Total number of points, split by face area. When None, each
            face gets ``density * area`` points.
        density: Points per square meter when ``count`` is None.
        noise: Standard deviation of isotropic Gaussian noise in meters.

    Returns:
        Points of shape ``(N, 3)``.
    """
    areas = np.array([np.linalg.norm(np.cross(u, v)) for _, u, v in faces])
    if count is None:
        counts = np.rint(density * areas).astype(int)
    else:
        counts = np.rint(count * areas / areas.sum()).astype(int)
    chunks = []
    for (origin, u, v), n in zip(faces, counts, strict=True):
        ab = rng.random((n, 2))
        chunks.append(origin + ab[:, :1] * u + ab[:, 1:] * v)
    points = np.concatenate(chunks)
    return points + rng.normal(scale=noise, size=points.shape)


def sample_room_labelled(
    rng: np.random.Generator,
    boxes: Sequence[Box] = (),
    box_points: int = 1500,
    density: float = 150.0,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Samples the static room plus movable boxes.

    Args:
        rng: Source of randomness.
        boxes: Movable boxes to add.
        box_points: Points sampled on each movable box.
        density: Points per square meter of static surface.

    Returns:
        A tuple ``(points, is_object)``; ``is_object`` marks the points
        sampled on the movable boxes.
    """
    static = sample_faces(rng, static_faces(), density=density)
    chunks = [static] + [
        sample_faces(rng, box_faces(box), count=box_points) for box in boxes
    ]
    labels = [np.zeros(len(static), dtype=np.bool_)] + [
        np.ones(len(chunk), dtype=np.bool_) for chunk in chunks[1:]
    ]
    return np.concatenate(chunks), np.concatenate(labels)


def sample_room(
    rng: np.random.Generator,
    boxes: Sequence[Box] = (),
    box_points: int = 1500,
    density: float = 150.0,
) -> NDArray[np.float64]:
    """Samples the static room plus movable boxes, without labels.

    Args:
        rng: Source of randomness.
        boxes: Movable boxes to add.
        box_points: Points sampled on each movable box.
        density: Points per square meter of static surface.

    Returns:
        Points of shape ``(N, 3)``.
    """
    return sample_room_labelled(rng, boxes, box_points, density)[0]


def static_point_count(density: float = 150.0) -> int:
    """Returns how many static points ``sample_room`` draws at a density."""
    return len(sample_room(np.random.default_rng(0), density=density))
