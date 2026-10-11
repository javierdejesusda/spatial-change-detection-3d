import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from rotations import rotation_z

from spatial_change.domain import (
    SE3,
    BoxModel,
    Change,
    ChangeType,
    EvidenceCounts,
    OrientedBox,
)
from spatial_change.semantics.relations import (
    DEFAULT_CONFIG,
    ChangeFact,
    Direction,
    Landmark,
    LandmarkRelation,
    RelationConfig,
    build_change_fact,
    describe_direction,
    distance_to_box,
    format_distance,
    format_rotation,
    nearest_landmarks,
    supporting_surface,
)


def make_box(
    center: Sequence[float],
    extent: Sequence[float] = (0.2, 0.2, 0.2),
    yaw_deg: float = 0.0,
) -> OrientedBox:
    return OrientedBox(
        center=np.array(center, dtype=np.float64),
        extent=np.array(extent, dtype=np.float64),
        rotation=rotation_z(yaw_deg),
    )


def make_table(
    landmark_id: str = "table-1",
    label: str = "table",
    yaw_deg: float = 0.0,
    height: float = 0.75,
    center_xy: tuple[float, float] = (0.0, 0.0),
    extent_xy: tuple[float, float] = (1.0, 0.5),
) -> Landmark:
    return Landmark(
        landmark_id,
        label,
        make_box(
            (center_xy[0], center_xy[1], height / 2),
            (extent_xy[0], extent_xy[1], height),
            yaw_deg,
        ),
    )


def make_landmark(
    label: str, center: Sequence[float], extent: Sequence[float]
) -> Landmark:
    """Builds an unrotated landmark whose id is its label."""
    return Landmark(label, label, make_box(center, extent))


def make_sofa(x: float) -> Landmark:
    return make_landmark("sofa", (x, 0.0, 0.4), (1.0, 1.0, 0.8))


CHAIR = make_landmark("chair", (0.9, 0.0, 0.5), (0.4, 0.4, 1.0))


def footprint_point(
    center_xy: tuple[float, float], yaw_deg: float, along: float, across: float
) -> tuple[float, float]:
    """Maps offsets along and across a yawed footprint to world xy."""
    rotation = rotation_z(yaw_deg)
    return (
        center_xy[0] + along * rotation[0, 0] + across * rotation[0, 1],
        center_xy[1] + along * rotation[1, 0] + across * rotation[1, 1],
    )


def test_default_config_values() -> None:
    config = RelationConfig()

    assert config.support_tolerance_m == 0.05
    assert config.next_to_m == 0.5
    assert config.near_m == 1.5
    assert config.max_landmarks == 3
    assert config.min_displacement_m == 0.05
    assert config.min_rotation_deg == 10.0
    assert config.vertical_ratio == 0.5


@pytest.mark.parametrize(
    "field",
    [
        "support_tolerance_m",
        "next_to_m",
        "near_m",
        "max_landmarks",
        "min_displacement_m",
        "min_rotation_deg",
        "vertical_ratio",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_config_rejects_non_positive(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        RelationConfig(**{field: value})


def test_config_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="near_m"):
        RelationConfig(near_m=math.inf)
    with pytest.raises(ValueError, match="next_to_m"):
        RelationConfig(next_to_m=math.nan)


def test_config_rejects_vertical_ratio_above_one() -> None:
    with pytest.raises(ValueError, match="vertical_ratio"):
        RelationConfig(vertical_ratio=1.5)

    assert RelationConfig(vertical_ratio=1.0).vertical_ratio == 1.0


@pytest.mark.parametrize("value", [2.5, True])
def test_config_rejects_non_integer_max_landmarks(value: object) -> None:
    with pytest.raises(ValueError, match="max_landmarks"):
        RelationConfig(**{"max_landmarks": value})


def test_config_accepts_integer_max_landmarks() -> None:
    assert RelationConfig(max_landmarks=2).max_landmarks == 2


def test_config_rejects_next_to_beyond_near() -> None:
    with pytest.raises(ValueError, match="next_to_m"):
        RelationConfig(next_to_m=2.0, near_m=1.0)


# Ties (0.025, 0.075, 0.925, 0.975, 1.05, ..., 2.05) are decided on the
# decimal value as written, so 1.25 and 1.15 round alike even though their
# binary values sit on different sides.
@pytest.mark.parametrize(
    ("meters", "text"),
    [
        (0.0, "about 0 cm"),
        (0.0001, "under 5 cm"),
        (0.02, "under 5 cm"),
        (0.0249, "under 5 cm"),
        (0.025, "about 5 cm"),
        (0.03, "about 5 cm"),
        (0.075, "about 10 cm"),
        (0.80, "about 80 cm"),
        (0.82, "about 80 cm"),
        (0.83, "about 85 cm"),
        (0.925, "about 95 cm"),
        (0.97, "about 95 cm"),
        (0.9749, "about 95 cm"),
        (0.975, "about 1.0 m"),
        (0.98, "about 1.0 m"),
        (1.0, "about 1.0 m"),
        (1.05, "about 1.1 m"),
        (1.15, "about 1.2 m"),
        (1.24, "about 1.2 m"),
        (1.25, "about 1.3 m"),
        (1.26, "about 1.3 m"),
        (1.45, "about 1.5 m"),
        (1.65, "about 1.7 m"),
        (1.75, "about 1.8 m"),
        (2.05, "about 2.1 m"),
        (12.0, "about 12.0 m"),
    ],
)
def test_format_distance(meters: float, text: str) -> None:
    assert format_distance(meters) == text


@pytest.mark.parametrize("bad", [-0.01, math.nan, math.inf, -math.inf])
def test_format_distance_rejects_invalid(bad: float) -> None:
    with pytest.raises(ValueError, match="meters"):
        format_distance(bad)


# Ties (2.5, 7.5, 47.5, 177.5) round up.
@pytest.mark.parametrize(
    ("degrees", "text"),
    [
        (0.0, "about 0 degrees"),
        (0.01, "under 5 degrees"),
        (2.0, "under 5 degrees"),
        (2.4, "under 5 degrees"),
        (2.5, "about 5 degrees"),
        (2.6, "about 5 degrees"),
        (7.5, "about 10 degrees"),
        (44.0, "about 45 degrees"),
        (47.4, "about 45 degrees"),
        (47.5, "about 50 degrees"),
        (47.6, "about 50 degrees"),
        (177.5, "about 180 degrees"),
        (180.0, "about 180 degrees"),
    ],
)
def test_format_rotation(degrees: float, text: str) -> None:
    assert format_rotation(degrees) == text


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_format_rotation_rejects_invalid(bad: float) -> None:
    with pytest.raises(ValueError, match="degrees"):
        format_rotation(bad)


@pytest.mark.parametrize(
    ("z", "landmarks"),
    [
        pytest.param(0.1, [make_table()], id="resting"),
        pytest.param(0.1 + 0.04, [], id="within-tolerance"),
    ],
)
def test_supporting_surface_floor(z: float, landmarks: list[Landmark]) -> None:
    box = make_box((3.0, 3.0, z))

    assert supporting_surface(box, landmarks, DEFAULT_CONFIG) == "floor"


def test_supporting_surface_on_table() -> None:
    box = make_box((0.2, 0.1, 0.75 + 0.1))

    assert supporting_surface(box, [make_table()], DEFAULT_CONFIG) == "table"


@pytest.mark.parametrize(
    ("along", "across", "supported"),
    [
        (0.0, 0.0, True),
        (0.45, 0.0, True),
        (-0.45, 0.2, True),
        (0.45, -0.2, True),
        (0.55, 0.0, False),
        (-0.55, 0.0, False),
        (0.0, 0.3, False),
        (0.0, -0.3, False),
        (0.45, 0.3, False),
    ],
)
def test_supporting_surface_on_rotated_table_with_offset_center(
    along: float, across: float, supported: bool
) -> None:
    # The 1.0 x 0.5 footprint is centered at (2, -1) and yawed 37 degrees.
    # (0.45, 0) lies outside the same footprint left unrotated, and (0, 0.3)
    # lies inside it.
    table = make_table(yaw_deg=37.0, center_xy=(2.0, -1.0))
    x, y = footprint_point((2.0, -1.0), 37.0, along, across)

    result = supporting_surface(make_box((x, y, 0.85)), [table], DEFAULT_CONFIG)

    assert result == ("table" if supported else None)


@pytest.mark.parametrize(
    "z",
    [
        pytest.param(1.5, id="wall-mounted"),
        pytest.param(0.75 + 0.1 + 0.2, id="top-too-far"),
    ],
)
def test_supporting_surface_out_of_reach_is_none(z: float) -> None:
    box = make_box((0.0, 0.0, z))

    assert supporting_surface(box, [make_table()], DEFAULT_CONFIG) is None


def test_supporting_surface_prefers_highest_top() -> None:
    low = make_table("a-low", "bench", height=0.72)
    high = make_table("b-high", "desk", height=0.76)
    box = make_box((0.0, 0.0, 0.76 + 0.1))

    assert supporting_surface(box, [low, high], DEFAULT_CONFIG) == "desk"
    assert supporting_surface(box, [high, low], DEFAULT_CONFIG) == "desk"


def test_supporting_surface_ties_break_on_smaller_footprint() -> None:
    big = make_table("a-big", "desk", extent_xy=(2.0, 2.0))
    small = make_table("b-small", "tray", extent_xy=(0.6, 0.6))
    box = make_box((0.0, 0.0, 0.85))

    assert supporting_surface(box, [big, small], DEFAULT_CONFIG) == "tray"
    assert supporting_surface(box, [small, big], DEFAULT_CONFIG) == "tray"


def test_supporting_surface_footprint_counts_tilted_box_projection() -> None:
    # A 1.0 x 0.25 x 0.8 slab lying on its long side (rotated 90 degrees
    # about x) covers 1.0 x 0.8 of the floor, although the product of its
    # first two extents is only 0.25. The tray covers 0.3.
    lying = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    slab = Landmark(
        "a-slab",
        "crate",
        OrientedBox(
            center=np.array([0.0, 0.0, 0.625]),
            extent=np.array([1.0, 0.25, 0.8]),
            rotation=lying,
        ),
    )
    tray = make_table("b-tray", "tray", extent_xy=(0.6, 0.5))
    box = make_box((0.0, 0.0, 0.85))

    assert supporting_surface(box, [slab, tray], DEFAULT_CONFIG) == "tray"
    assert supporting_surface(box, [tray, slab], DEFAULT_CONFIG) == "tray"


def test_supporting_surface_final_tie_is_landmark_id() -> None:
    first = make_table("a", "desk")
    second = make_table("b", "shelf")
    box = make_box((0.0, 0.0, 0.85))

    assert supporting_surface(box, [second, first], DEFAULT_CONFIG) == "desk"


def test_distance_to_box_inside_is_zero() -> None:
    box = make_box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))

    assert distance_to_box(np.array([0.5, -0.5, 0.9]), box) == 0.0


def test_distance_to_box_on_face() -> None:
    box = make_box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))

    assert distance_to_box(np.array([1.0, 0.0, 0.0]), box) == 0.0
    assert distance_to_box(np.array([4.0, 0.3, -0.2]), box) == pytest.approx(
        3.0
    )


def test_distance_to_box_at_corner() -> None:
    box = make_box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))

    # Offsets of (2, 2, 1) beyond the corner (1, 1, 1): sqrt(4 + 4 + 1).
    assert distance_to_box(np.array([3.0, 3.0, 2.0]), box) == pytest.approx(3.0)


def test_distance_to_box_rotated() -> None:
    # A 2 x 2 x 2 box yawed 45 degrees has its vertex at (sqrt(2), 0, 0).
    box = make_box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), yaw_deg=45.0)

    assert distance_to_box(np.array([3.0, 0.0, 0.0]), box) == pytest.approx(
        3.0 - math.sqrt(2.0)
    )
    # The axis-aligned footprint would contain this point, the box does not.
    assert distance_to_box(np.array([0.95, 0.95, 0.0]), box) == pytest.approx(
        math.sqrt(2.0) * 0.95 - 1.0
    )


def test_distance_to_box_rotated_with_offset_center_and_uneven_extent() -> None:
    center = (2.0, -1.0)
    box = make_box((*center, 0.5), (1.0, 0.5, 1.0), yaw_deg=37.0)

    def point(along: float, across: float, z: float) -> np.ndarray:
        x, y = footprint_point(center, 37.0, along, across)
        return np.array([x, y, z])

    # 1.0 m past the short face (half-length 0.5), level with the center.
    assert distance_to_box(point(1.5, 0.0, 0.5), box) == pytest.approx(1.0)
    # 0.3 past the short face and 0.4 past the long face (half-width 0.25).
    assert distance_to_box(point(0.8, 0.65, 0.5), box) == pytest.approx(0.5)
    # The same corner offsets and 1.2 above the top face at z = 1.
    assert distance_to_box(point(0.8, 0.65, 2.2), box) == pytest.approx(1.3)
    assert distance_to_box(point(0.4, -0.2, 0.2), box) == pytest.approx(
        0.0, abs=1e-9
    )


def test_nearest_landmarks_classifies_filters_and_sorts() -> None:
    obj = make_box((0.0, 0.0, 0.1))
    sofa = make_sofa(1.0)
    shelf = make_landmark("shelf", (0.0, 1.9, 1.0), (1, 1, 2))
    far = Landmark("far", "plant", make_box((5.0, 5.0, 0.5), (1, 1, 1)))

    relations = nearest_landmarks(obj, [far, shelf, sofa], DEFAULT_CONFIG)

    assert relations == [
        LandmarkRelation(
            landmark_id="sofa", label="sofa", distance_m=0.5, relation="next to"
        ),
        LandmarkRelation(
            landmark_id="shelf", label="shelf", distance_m=1.4, relation="near"
        ),
    ]


def test_nearest_landmarks_boundaries_are_inclusive() -> None:
    obj = make_box((0.0, 0.0, 0.1))
    at_next_to = Landmark("a", "chair", make_box((1.0, 0.0, 0.1), (1, 1, 1)))
    at_near = Landmark("b", "bin", make_box((0.0, 2.0, 0.1), (1, 1, 1)))
    config = RelationConfig(next_to_m=0.5, near_m=1.5)

    relations = nearest_landmarks(obj, [at_near, at_next_to], config)

    assert [(r.landmark_id, r.relation) for r in relations] == [
        ("a", "next to"),
        ("b", "near"),
    ]
    assert relations[0].distance_m == pytest.approx(0.5)
    assert relations[1].distance_m == pytest.approx(1.5)


def test_nearest_landmarks_breaks_ties_on_id_and_caps() -> None:
    obj = make_box((0.0, 0.0, 0.1))
    landmarks = [
        Landmark(f"lm-{i}", "chair", make_box((x, y, 0.1), (1, 1, 1)))
        for i, (x, y) in enumerate([(0, 1.5), (1.5, 0), (-1.5, 0), (0, -1.5)])
    ]

    relations = nearest_landmarks(obj, landmarks[::-1], DEFAULT_CONFIG)

    assert [r.landmark_id for r in relations] == ["lm-0", "lm-1", "lm-2"]


def test_nearest_landmarks_breaks_rounding_noise_ties_on_id() -> None:
    # The object's center lies on the surface of both landmarks, so both are
    # exactly 0 m away; at x = 0.1 the float math leaves "a" about 3e-17 m
    # away, which must not put "b" first.
    obj = make_box((0.1, 0.0, 0.5))
    a = Landmark("a", "chair", make_box((0.6, 0.0, 0.0), (1, 1, 1)))
    b = Landmark("b", "chair", make_box((0.1, 0.0, 1.0), (1, 1, 1)))

    relations = nearest_landmarks(obj, [b, a], DEFAULT_CONFIG)

    assert [r.landmark_id for r in relations] == ["a", "b"]


def test_nearest_landmarks_excludes_supporting_landmark() -> None:
    obj = make_box((0.0, 0.0, 0.85))

    relations = nearest_landmarks(obj, [make_table(), CHAIR], DEFAULT_CONFIG)

    assert [r.landmark_id for r in relations] == ["chair"]


def test_nearest_landmarks_inside_landmark_has_zero_distance() -> None:
    obj = make_box((0.0, 0.0, 1.0))
    shelf = make_landmark("shelf", (0.0, 0.0, 1.0), (2, 2, 2))

    relations = nearest_landmarks(obj, [shelf], DEFAULT_CONFIG)

    assert relations[0].distance_m == 0.0
    assert relations[0].relation == "next to"


# World_from_camera rotation of a level camera looking along world +y.
LOOK_Y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def camera(rotation: np.ndarray) -> SE3:
    return SE3(rotation, np.array([0.0, 0.0, 1.5]))


LEVEL_CAMERA = camera(LOOK_Y)


def moved_boxes(
    delta: Sequence[float], start: Sequence[float] = (0.0, 0.0, 0.5)
) -> tuple[OrientedBox, OrientedBox]:
    return make_box(start), make_box(np.array(start) + np.array(delta))


def direction_of(
    delta: Sequence[float],
    viewpoint: SE3 | None,
    landmarks: Sequence[Landmark] = (),
    start: Sequence[float] = (0.0, 0.0, 0.5),
    config: RelationConfig = DEFAULT_CONFIG,
) -> Direction:
    before, after = moved_boxes(delta, start)
    result = describe_direction(before, after, viewpoint, landmarks, config)
    assert result is not None
    return result


SECTORS = [
    ((0.0, 1.0, 0.0), "farther"),
    ((1.0, 1.0, 0.0), "farther and to the right"),
    ((1.0, 0.0, 0.0), "to the right"),
    ((1.0, -1.0, 0.0), "closer and to the right"),
    ((0.0, -1.0, 0.0), "closer"),
    ((-1.0, -1.0, 0.0), "closer and to the left"),
    ((-1.0, 0.0, 0.0), "to the left"),
    ((-1.0, 1.0, 0.0), "farther and to the left"),
]


@pytest.mark.parametrize(("delta", "text"), SECTORS)
@pytest.mark.parametrize("yaw", [0.0, 90.0, 37.0, -140.0])
def test_direction_eight_sectors_for_any_viewpoint_yaw(
    delta: tuple[float, ...], text: str, yaw: float
) -> None:
    rotation = rotation_z(yaw)
    result = direction_of(rotation @ np.array(delta), camera(rotation @ LOOK_Y))

    assert result.text == text


def test_direction_sector_edges_are_stable() -> None:
    # 20 degrees off forward is still "farther"; 25 degrees is not.
    near_axis = (math.sin(math.radians(20)), math.cos(math.radians(20)), 0.0)
    past_edge = (math.sin(math.radians(25)), math.cos(math.radians(25)), 0.0)

    assert direction_of(near_axis, LEVEL_CAMERA).text == "farther"
    assert (
        direction_of(past_edge, LEVEL_CAMERA).text == "farther and to the right"
    )


@pytest.mark.parametrize("pitch_deg", [30.0, -30.0])
def test_direction_ignores_camera_pitch(pitch_deg: float) -> None:
    # Rotate about the camera x axis; with this matrix a positive angle
    # tilts the camera up and a negative one down.
    c, s = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    tilt = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    pitched = LOOK_Y @ tilt
    assert np.sign(pitched[2, 2]) == np.sign(pitch_deg)

    assert direction_of((0.0, 1.0, 0.0), camera(pitched)).text == "farther"


@pytest.mark.parametrize(
    ("delta", "text"),
    [
        ((0.5, 0.0, 0.4), "to the right and up"),
        ((0.0, -0.5, -0.4), "closer and down"),
        ((0.5, 0.0, 0.2), "to the right"),
        ((0.0, 0.0, 0.6), "up"),
        ((0.1, 0.0, -0.9), "down"),
    ],
)
def test_direction_vertical_part(delta: tuple[float, ...], text: str) -> None:
    assert direction_of(delta, LEVEL_CAMERA).text == text


@pytest.mark.parametrize(
    ("delta", "viewpoint", "text"),
    [
        ((0.042, 0.0, 0.042), LEVEL_CAMERA, "to the right and up"),
        ((0.0, -0.042, -0.042), LEVEL_CAMERA, "closer and down"),
        ((0.042, 0.0, 0.03), LEVEL_CAMERA, "to the right"),
        ((0.03, 0.0, 0.042), LEVEL_CAMERA, "up"),
        ((0.06, 0.0, 0.03), LEVEL_CAMERA, "to the right"),
        ((0.042, 0.0, 0.042), None, "up"),
    ],
)
def test_direction_words_dominant_part_of_a_short_move(
    delta: tuple[float, ...], viewpoint: SE3 | None, text: str
) -> None:
    # Moves 1 to 4 have a norm above 0.05 m but no part that reaches it, so a
    # part is worded when it is at least as large as the other one. In move 5
    # the 0.03 m rise is smaller and below the minimum, so it stays unworded.
    # Move 6 is move 1 without a viewpoint, so only the rise is worded.
    result = direction_of(delta, viewpoint, start=(0.0, 0.0, 0.0))

    assert result.text == text


@pytest.mark.parametrize(("delta", "text"), SECTORS)
def test_direction_ignores_camera_roll(
    delta: tuple[float, ...], text: str
) -> None:
    # Rolling 180 degrees about the optical axis flips the camera x axis, so
    # a "right" taken from it would swap left and right.
    rolled = LOOK_Y @ np.diag([-1.0, -1.0, 1.0])
    assert rolled[0, 0] == -LOOK_Y[0, 0]

    assert direction_of(delta, camera(rolled)).text == text


def test_direction_tiny_motion_is_none() -> None:
    small = moved_boxes((0.03, 0.02, 0.01))
    still = moved_boxes((0.0, 0.0, 0.0))

    assert describe_direction(*small, LEVEL_CAMERA, [], DEFAULT_CONFIG) is None
    assert describe_direction(*still, None, [], DEFAULT_CONFIG) is None


def test_direction_without_viewpoint_is_vertical_only() -> None:
    flat = direction_of((1.0, 0.0, 0.0), None)
    result = direction_of((0.2, 0.0, 0.5), None)

    assert flat == Direction(text=None, toward=None)
    assert result.text == "up"


def test_direction_camera_looking_down_has_no_horizontal_words() -> None:
    down_cam = camera(np.diag([1.0, -1.0, -1.0]))

    flat = direction_of((1.0, 0.0, 0.0), down_cam)
    result = direction_of((1.0, 0.0, -0.8), down_cam)

    assert flat == Direction(text=None, toward=None)
    assert result.text == "down"


def test_direction_toward_does_not_need_direction_words() -> None:
    sofa = make_sofa(4.0)
    down_cam = camera(np.diag([1.0, -1.0, -1.0]))

    no_view = direction_of((1.0, 0.0, 0.0), None, [sofa])
    looking_down = direction_of((1.0, 0.0, 0.0), down_cam, [sofa])

    assert no_view == Direction(text=None, toward="sofa")
    assert looking_down == Direction(text=None, toward="sofa")


def test_direction_toward_landmark() -> None:
    result = direction_of((1.0, 0.0, 0.0), LEVEL_CAMERA, [make_sofa(4.0)])

    assert result.text == "to the right"
    assert result.toward == "sofa"


def test_direction_not_toward_when_moving_away_or_sideways() -> None:
    sofa = make_sofa(4.0)
    away = direction_of((-1.0, 0.0, 0.0), LEVEL_CAMERA, [sofa])
    sideways = direction_of((0.0, 1.0, 0.0), LEVEL_CAMERA, [sofa])

    assert away.toward is None
    assert sideways.toward is None


def test_direction_toward_needs_enough_progress() -> None:
    # Moving 0.55 m at 30 degrees off the line to a distant sofa closes
    # about 0.48 m, below the 0.5 m threshold.
    sofa = make_landmark("sofa", (20.0, 0.0, 0.5), (1, 1, 1))
    config = RelationConfig(min_displacement_m=0.5)
    angle = math.radians(30.0)
    delta = (0.55 * math.cos(angle), 0.55 * math.sin(angle), 0.0)

    result = direction_of(delta, LEVEL_CAMERA, [sofa], config=config)

    assert result.toward is None


def test_direction_toward_picks_largest_progress() -> None:
    # Moving 1 m along +y closes 0.87 m on the offset lamp (cosine 0.91) but
    # the full 1.0 m on the desk dead ahead. The lamp has the smaller id, so
    # only the progress ranking can pick the desk.
    offset = Landmark("a", "lamp", make_box((1.5, 3.0, 0.5), (0.4, 0.4, 0.8)))
    ahead = Landmark("b", "desk", make_box((0.0, 5.0, 0.5), (0.4, 0.4, 0.8)))

    forward = direction_of((0.0, 1.0, 0.0), LEVEL_CAMERA, [offset, ahead])
    reversed_order = direction_of(
        (0.0, 1.0, 0.0), LEVEL_CAMERA, [ahead, offset]
    )

    assert forward.toward == "desk"
    assert reversed_order.toward == "desk"


def test_direction_toward_breaks_rounding_noise_ties_on_id() -> None:
    # The two landmarks mirror each other across the path, so the progress
    # on both is equal; with the scene shifted 0.1 m along x the float math
    # gives "b" about 5e-16 m more, which must not beat the smaller id.
    shift = 0.1
    size = (0.4, 0.4, 0.8)
    a = Landmark("a", "lamp", make_box((shift + 0.7, 3.0, 0.5), size))
    b = Landmark("b", "vase", make_box((shift - 0.7, 3.0, 0.5), size))

    result = direction_of(
        (0.0, 1.0, 0.0), None, [b, a], start=(shift, 0.0, 0.5)
    )

    assert result.toward == "lamp"


def test_direction_toward_breaks_progress_ties_on_id() -> None:
    first = Landmark("a", "lamp", make_box((0.0, 5.0, 0.5), (0.4, 0.4, 0.8)))
    second = Landmark("b", "desk", make_box((0.0, 5.0, 0.5), (0.4, 0.4, 0.8)))

    result = direction_of((0.0, 1.0, 0.0), LEVEL_CAMERA, [second, first])

    assert result.toward == "lamp"


def make_change(
    change_type: ChangeType,
    before: OrientedBox | None,
    after: OrientedBox | None,
    displacement_m: float | None = None,
    rotation_deg: float | None = None,
    label: str = "mug",
) -> Change:
    return Change(
        change_id="c1",
        type=change_type,
        label=label,
        description="",
        confidence=0.9,
        before_box=None if before is None else BoxModel.from_box(before),
        after_box=None if after is None else BoxModel.from_box(after),
        displacement_m=displacement_m,
        rotation_deg=rotation_deg,
        evidence=EvidenceCounts(consistent=1, free=0, occluded=0, unobserved=0),
    )


def moved_change(
    dx: float, displacement_m: float | None = None, rotation_deg: float = 0.0
) -> Change:
    """Builds a MOVED change of a floor box by ``dx`` along x.

    The reported displacement defaults to ``dx``.
    """
    return make_change(
        ChangeType.MOVED,
        make_box((0.0, 0.0, 0.1)),
        make_box((dx, 0.0, 0.1)),
        displacement_m=dx if displacement_m is None else displacement_m,
        rotation_deg=rotation_deg,
    )


def test_fact_for_added_object_on_table() -> None:
    change = make_change(ChangeType.ADDED, None, make_box((0.0, 0.0, 0.85)))

    fact = build_change_fact(change, [make_table(), CHAIR], LEVEL_CAMERA)

    assert fact.change_id == "c1"
    assert fact.type is ChangeType.ADDED
    assert fact.label == "mug"
    assert fact.support_before is None
    assert fact.support_after == "table"
    assert fact.near_before == []
    assert [r.label for r in fact.near_after] == ["chair"]
    assert fact.direction is None
    assert fact.toward is None
    assert fact.displacement_m is None
    assert fact.displacement_text is None
    assert fact.rotation_deg is None
    assert fact.rotation_text is None


def test_fact_for_removed_object() -> None:
    change = make_change(ChangeType.REMOVED, make_box((3.0, 3.0, 0.1)), None)

    fact = build_change_fact(change, [], None)

    assert fact.support_before == "floor"
    assert fact.support_after is None
    assert fact.near_after == []


def test_fact_for_moved_object() -> None:
    before = make_box((0.0, 0.0, 0.1))
    after = make_box((0.4, 0.0, 0.85))
    change = make_change(
        ChangeType.MOVED, before, after, displacement_m=0.85, rotation_deg=47.0
    )

    fact = build_change_fact(
        change, [make_sofa(3.0), make_table()], LEVEL_CAMERA
    )

    assert fact.support_before == "floor"
    assert fact.support_after == "table"
    assert fact.direction == "to the right and up"
    assert fact.toward is None
    assert fact.displacement_m == pytest.approx(0.85)
    assert fact.displacement_text == "about 85 cm"
    assert fact.rotation_deg == 47.0
    assert fact.rotation_text == "about 45 degrees"


@pytest.mark.parametrize(
    ("viewpoint", "direction"), [(LEVEL_CAMERA, "to the right"), (None, None)]
)
def test_fact_reports_landmark_moved_toward(
    viewpoint: SE3 | None, direction: str | None
) -> None:
    fact = build_change_fact(moved_change(1.0), [make_sofa(3.0)], viewpoint)

    assert fact.direction == direction
    assert fact.toward == "sofa"
    assert [r.relation for r in fact.near_after] == ["near"]


def test_fact_omits_text_below_thresholds() -> None:
    change = moved_change(0.03, rotation_deg=4.0)

    fact = build_change_fact(change, [], LEVEL_CAMERA)

    assert fact.direction is None
    assert fact.displacement_m == pytest.approx(0.03)
    assert fact.displacement_text is None
    assert fact.rotation_deg == 4.0
    assert fact.rotation_text is None


def test_fact_for_uncertain_uses_available_boxes() -> None:
    only_after = make_change(
        ChangeType.UNCERTAIN, None, make_box((3.0, 3.0, 0.1))
    )
    both = make_change(
        ChangeType.UNCERTAIN,
        make_box((0.0, 0.0, 0.1)),
        make_box((0.0, 1.0, 0.1)),
    )

    fact_after = build_change_fact(only_after, [], None)
    fact_both = build_change_fact(both, [], LEVEL_CAMERA)

    assert fact_after.support_before is None
    assert fact_after.support_after == "floor"
    assert fact_after.displacement_m is None
    assert fact_after.displacement_text is None
    assert fact_both.direction == "farther"
    assert fact_both.displacement_m == pytest.approx(1.0)
    assert fact_both.displacement_text == "about 1.0 m"
    assert fact_both.rotation_deg is None
    assert fact_both.rotation_text is None


@pytest.mark.parametrize("reported_m", [0.02, 0.9, 3.0])
def test_fact_displacement_comes_from_box_centers(reported_m: float) -> None:
    change = moved_change(1.0, displacement_m=reported_m)

    fact = build_change_fact(change, [], LEVEL_CAMERA)

    assert fact.direction == "to the right"
    assert fact.displacement_m == pytest.approx(1.0)
    assert fact.displacement_text == "about 1.0 m"


def test_fact_displacement_below_minimum_has_no_text() -> None:
    change = moved_change(0.02, displacement_m=3.0)

    fact = build_change_fact(change, [], LEVEL_CAMERA)

    assert fact.direction is None
    assert fact.displacement_m == pytest.approx(0.02)
    assert fact.displacement_text is None


def test_fact_displacement_needs_both_boxes() -> None:
    change = make_change(
        ChangeType.ADDED,
        None,
        make_box((0.0, 0.0, 0.1)),
        displacement_m=1.0,
        rotation_deg=30.0,
    )

    fact = build_change_fact(change, [], None)

    assert fact.displacement_m is None
    assert fact.displacement_text is None
    assert fact.rotation_deg == 30.0
    assert fact.rotation_text == "about 30 degrees"


def test_fact_rejects_uncertain_change_without_boxes() -> None:
    # The other types are checked by Change itself when it is built.
    change = make_change(ChangeType.UNCERTAIN, None, None)

    with pytest.raises(ValueError, match="at least one box"):
        build_change_fact(change, [], None)


def test_fact_is_frozen_and_forbids_extra_fields() -> None:
    change = make_change(ChangeType.ADDED, None, make_box((0, 0, 0.1)))
    fact = build_change_fact(change, [], None)

    with pytest.raises(ValidationError):
        fact.label = "cup"
    with pytest.raises(ValidationError):
        ChangeFact.model_validate({**fact.model_dump(), "extra": 1})


def test_fact_round_trips_through_json() -> None:
    change = make_change(
        ChangeType.MOVED,
        make_box((0.0, 0.0, 0.1)),
        make_box((0.0, 0.5, 0.1)),
        displacement_m=0.5,
        rotation_deg=0.0,
    )

    fact = build_change_fact(change, [make_sofa(1.0)], LEVEL_CAMERA)

    assert ChangeFact.model_validate_json(fact.model_dump_json()) == fact


finite = st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False)
yaws = st.floats(-180.0, 180.0, allow_nan=False, allow_infinity=False)


Scene = tuple[Change, list[Landmark], SE3]

# A scene is skipped when moving its boxes by this much changes a decision:
# it sits on a threshold, where float rounding may decide either way.
NUDGE_M = 1e-6


@st.composite
def scenes(draw: st.DrawFn) -> Scene:
    def box(z_low: float = 0.0, z_high: float = 1.5) -> OrientedBox:
        return make_box(
            (draw(finite), draw(finite), draw(st.floats(z_low, z_high))),
            (
                draw(st.floats(0.1, 1.0)),
                draw(st.floats(0.1, 1.0)),
                draw(st.floats(0.1, 1.0)),
            ),
            draw(yaws),
        )

    landmarks = [
        Landmark(f"lm-{i}", "item", box(0.0, 1.0))
        for i in range(draw(st.integers(0, 5)))
    ]
    before, after = box(), box()
    change = make_change(
        ChangeType.MOVED,
        before,
        after,
        displacement_m=float(np.linalg.norm(after.center - before.center)),
        rotation_deg=draw(st.floats(0.0, 180.0)),
    )
    view = SE3(
        rotation_z(draw(yaws)) @ LOOK_Y,
        np.array([draw(finite), draw(finite), 1.5]),
    )
    return change, landmarks, view


def remap_scene(
    scene: Scene, move: Callable[[int, OrientedBox], OrientedBox], view: SE3
) -> Scene:
    """Applies ``move`` to every box of a scene and swaps in ``view``.

    Boxes are numbered 0 for before, 1 for after, then the landmarks.
    """
    change, landmarks, _ = scene
    assert change.before_box is not None and change.after_box is not None
    new_change = make_change(
        change.type,
        move(0, change.before_box.to_box()),
        move(1, change.after_box.to_box()),
        change.displacement_m,
        change.rotation_deg,
    )
    new_landmarks = [
        Landmark(lm.landmark_id, lm.label, move(2 + i, lm.box))
        for i, lm in enumerate(landmarks)
    ]
    return new_change, new_landmarks, view


def shift_scene(scene: Scene, dx: float, dy: float) -> Scene:
    offset = np.array([dx, dy, 0.0])
    view = scene[2]

    def moved(_: int, box: OrientedBox) -> OrientedBox:
        return OrientedBox(box.center + offset, box.extent, box.rotation)

    return remap_scene(
        scene, moved, SE3(view.rotation, view.translation + offset)
    )


def spin_scene(scene: Scene, yaw: float) -> Scene:
    spin = rotation_z(yaw)
    view = scene[2]

    def turned(_: int, box: OrientedBox) -> OrientedBox:
        return OrientedBox(spin @ box.center, box.extent, spin @ box.rotation)

    return remap_scene(
        scene, turned, SE3(spin @ view.rotation, spin @ view.translation)
    )


def nudge_scene(scene: Scene, step: float) -> Scene:
    """Moves each box by ``step``, every one in its own floor direction."""

    def nudged(index: int, box: OrientedBox) -> OrientedBox:
        angle = 2.4 * index
        offset = step * np.array([math.cos(angle), math.sin(angle), 0.0])
        return OrientedBox(box.center + offset, box.extent, box.rotation)

    return remap_scene(scene, nudged, scene[2])


def decisions(fact: ChangeFact) -> tuple[object, ...]:
    """Lists the discrete outcomes that a rigid motion must not change."""
    return (
        fact.direction,
        fact.toward,
        fact.support_before,
        fact.support_after,
        [(r.landmark_id, r.relation) for r in fact.near_before],
        [(r.landmark_id, r.relation) for r in fact.near_after],
    )


def is_stable(scene: Scene) -> bool:
    """Tells whether no decision sits within ``NUDGE_M`` of a threshold."""
    expected = decisions(build_change_fact(*scene))
    return all(
        decisions(build_change_fact(*nudge_scene(scene, step))) == expected
        for step in (NUDGE_M, -NUDGE_M)
    )


def test_stability_filter_rejects_scenes_on_a_threshold() -> None:
    view = LEVEL_CAMERA
    before = make_box((0.0, 0.0, 0.5))
    after = make_box((0.0, 0.1, 0.5))
    # The corner of this landmark is 0.9 m by 1.2 m from the after center, so
    # it is exactly 1.5 m away: on the "near" threshold, where rounding of a
    # translated copy decides the relation.
    on_threshold = Landmark("lm", "item", make_box((1.4, 1.8, 0.5), (1, 1, 1)))
    clear = Landmark("lm", "item", make_box((1.4, 1.5, 0.5), (1, 1, 1)))

    def scene_with(landmark: Landmark) -> Scene:
        change = make_change(ChangeType.MOVED, before, after, 0.1, 0.0)
        return change, [landmark], view

    assert not is_stable(scene_with(on_threshold))
    assert is_stable(scene_with(clear))


@settings(max_examples=60, deadline=None)
@given(scenes(), finite, finite)
def test_fact_is_translation_invariant(
    scene: Scene, dx: float, dy: float
) -> None:
    assume(is_stable(scene))
    original = build_change_fact(*scene)

    shifted = build_change_fact(*shift_scene(scene, dx, dy))

    assert decisions(shifted) == decisions(original)


@settings(max_examples=60, deadline=None)
@given(scenes(), yaws)
def test_direction_is_yaw_invariant(scene: Scene, yaw: float) -> None:
    assume(is_stable(scene))
    original = build_change_fact(*scene)

    spun = build_change_fact(*spin_scene(scene, yaw))

    assert spun.direction == original.direction
    assert spun.toward == original.toward


@settings(max_examples=60, deadline=None)
@given(scenes(), st.integers(1, 6))
def test_nearest_landmarks_sorted_and_capped(scene: Scene, cap: int) -> None:
    change, landmarks, _ = scene
    assert change.after_box is not None
    config = RelationConfig(max_landmarks=cap)

    relations = nearest_landmarks(change.after_box.to_box(), landmarks, config)

    # Distances equal to within a nanometer are ties, broken by id.
    keys = [(round(r.distance_m, 9), r.landmark_id) for r in relations]
    assert keys == sorted(keys)
    assert len(relations) <= cap
    assert all(r.distance_m <= config.near_m for r in relations)
