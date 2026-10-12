import numpy as np
from report_scenes import box, change, one_of_each_report

from spatial_change.domain import ChangeType, OrientedBox
from spatial_change.visualization.segments import (
    Role,
    arrow_segments,
    box_edges,
    change_segments,
    region_segments,
)


def test_box_edges_join_corners_that_differ_on_one_axis() -> None:
    box = OrientedBox(
        np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]), np.eye(3)
    )

    edges = box_edges(box)

    assert edges.shape == (12, 2, 3)
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    assert sorted(lengths.tolist()) == [2.0] * 4 + [4.0] * 4 + [6.0] * 4
    corners = {tuple(c) for c in box.corners().tolist()}
    assert {tuple(p) for p in edges.reshape(-1, 3).tolist()} == corners


def test_arrow_has_a_shaft_and_two_barbs_at_the_tip() -> None:
    segments = arrow_segments(np.zeros(3), np.array([2.0, 0.0, 0.0]))

    assert segments.shape == (3, 2, 3)
    np.testing.assert_allclose(segments[0], [[0, 0, 0], [2, 0, 0]])
    np.testing.assert_allclose(segments[1:, 0], [[2, 0, 0]] * 2)
    barbs = segments[1:, 1]
    assert np.all(barbs[:, 0] < 2.0)
    np.testing.assert_allclose(barbs[0] + barbs[1], 2 * barbs[0] * [1, 0, 0])


def test_vertical_arrow_still_gets_barbs() -> None:
    segments = arrow_segments(np.zeros(3), np.array([0.0, 0.0, 1.0]))

    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    assert np.all(lengths > 0.0)


def test_zero_length_arrow_is_empty() -> None:
    assert arrow_segments(np.ones(3), np.ones(3)).shape == (0, 2, 3)


def test_change_segments_hold_the_boxes_a_change_has() -> None:
    moved, added, removed, uncertain = one_of_each_report().changes

    assert [r for r, _ in change_segments(moved)] == [
        Role.BEFORE_BOX,
        Role.AFTER_BOX,
        Role.ARROW,
    ]
    assert [r for r, _ in change_segments(added)] == [Role.AFTER_BOX]
    assert [r for r, _ in change_segments(removed)] == [Role.BEFORE_BOX]
    assert [r for r, _ in change_segments(uncertain)] == [
        Role.BEFORE_BOX,
        Role.AFTER_BOX,
    ]
    shapes = [seg.shape for _, seg in change_segments(moved)]
    assert shapes == [(12, 2, 3), (12, 2, 3), (3, 2, 3)]


def test_the_arrow_runs_between_the_box_centers() -> None:
    moved = one_of_each_report().changes[0]

    arrow = dict(change_segments(moved))[Role.ARROW]

    np.testing.assert_allclose(arrow[0, 0], moved.before_box.center)
    np.testing.assert_allclose(arrow[0, 1], moved.after_box.center)


def test_a_change_with_no_geometry_has_no_segments() -> None:
    nowhere = change("c9", ChangeType.UNCERTAIN, "rug", None, None)

    assert change_segments(nowhere) == []


def test_a_move_that_stayed_put_has_no_arrow() -> None:
    spot = box((1.0, 1.0, 0.4))
    stayed = change("c9", ChangeType.MOVED, "chair", spot, spot)

    assert [r for r, _ in change_segments(stayed)] == [
        Role.BEFORE_BOX,
        Role.AFTER_BOX,
    ]


def test_roles_are_the_names_the_outputs_write() -> None:
    assert [r.value for r in Role] == [
        "before_box",
        "after_box",
        "arrow",
        "region_box",
    ]


def test_a_region_is_one_box_with_the_region_role() -> None:
    region = one_of_each_report().unobserved_regions[0]

    ((role, edges),) = region_segments(region)

    assert role is Role.REGION_BOX
    assert edges.shape == (12, 2, 3)
    np.testing.assert_allclose(edges, box_edges(region.box.to_box()))
