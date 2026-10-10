import numpy as np
import pytest

from spatial_change.domain import SE3, Intrinsics
from spatial_change.geometry.projection import project, unproject_depth
from spatial_change.synthetic.render import PlacedMesh, RenderScene

ROOM = (4.0, 3.0, 2.5)
INTRINSICS = Intrinsics(fx=40.0, fy=40.0, cx=32.0, cy=24.0, width=65, height=49)
# Camera at (1, 1.5, 1.2) looking along +x: right is -y, down is -z.
FACING_X = SE3(
    np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
    np.array([1.0, 1.5, 1.2]),
)


def crate_mesh(
    index: int = 0, z_range: tuple[float, float] = (0.0, 2.0)
) -> PlacedMesh:
    # Box spanning x in [2.5, 3.0], y in [1.25, 1.75], and z in z_range.
    corners = np.array(
        [[x, y, z] for x in (2.5, 3.0) for y in (1.25, 1.75) for z in z_range],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ],
        dtype=np.int32,
    )  # fmt: skip
    return PlacedMesh(index, corners, triangles, (0.9, 0.2, 0.1))


def test_empty_room_has_z_depth_everywhere_and_no_instances() -> None:
    view = RenderScene(ROOM, []).render(INTRINSICS, FACING_X)

    assert view.depth.dtype == np.float32
    assert view.depth.shape == (49, 65)
    assert np.all(view.depth > 0)
    assert view.depth[24, 32] == pytest.approx(3.0, abs=1e-4)
    assert view.depth[20, 40] == pytest.approx(3.0, abs=1e-4)
    ray_length = 3.0 * np.sqrt(1 + (8 / 40) ** 2 + (4 / 40) ** 2)
    assert abs(view.depth[20, 40] - ray_length) > 0.05
    assert view.instance.dtype == np.int32
    assert np.all(view.instance == -1)


def test_rays_that_miss_give_zero_depth_and_black() -> None:
    outside = SE3(FACING_X.rotation, np.array([-5.0, 1.5, 1.2]))
    away = SE3(
        FACING_X.rotation @ np.diag([-1.0, 1.0, -1.0]), outside.translation
    )

    view = RenderScene(ROOM, []).render(INTRINSICS, away)

    assert np.all(view.depth == 0.0)
    assert np.all(view.instance == -1)
    assert np.all(view.rgb == 0)


def test_known_point_renders_at_its_projected_pixel() -> None:
    scene = RenderScene(ROOM, [crate_mesh(index=7)])
    view = scene.render(INTRINSICS, FACING_X)
    point = np.array([[2.5, 1.5 - 0.1, 1.2 - 0.2]])  # on the near face

    uv, depth = project(point, INTRINSICS, FACING_X)
    col, row = np.rint(uv[0]).astype(int)

    # u = 32 + 40 * 0.1 / 1.5 = 34.67 and v = 24 + 40 * 0.2 / 1.5 = 29.33.
    assert depth[0] == pytest.approx(1.5)
    assert (col, row) == (35, 29)
    assert view.instance[row, col] == 7
    assert view.depth[row, col] == pytest.approx(1.5, abs=1e-4)
    back = unproject_depth(view.depth, INTRINSICS, FACING_X)
    assert np.min(np.linalg.norm(back - point, axis=1)) < 0.05


def test_silhouette_rows_match_the_projected_vertical_extent() -> None:
    # The near face edges project to v = 10.67 and v = 37.33, so a correct
    # render covers rows 11 to 37 and a half-pixel shift either way moves
    # the top or the bottom row.
    mesh = crate_mesh(index=3, z_range=(0.7, 1.7))
    view = RenderScene(ROOM, [mesh]).render(INTRINSICS, FACING_X)
    edges = np.array([[2.5, 1.5, 1.7], [2.5, 1.5, 0.7]])

    uv, _ = project(edges, INTRINSICS, FACING_X)
    rows = np.flatnonzero((view.instance == 3).any(axis=1))

    assert uv[:, 1] == pytest.approx([32.0 / 3.0, 112.0 / 3.0])
    assert (rows.min(), rows.max()) == (11, 37)
    assert np.array_equal(rows, np.arange(11, 38))


def test_instance_pixels_unproject_onto_their_surface() -> None:
    mesh = crate_mesh(index=3)
    view = RenderScene(ROOM, [mesh]).render(INTRINSICS, FACING_X)

    mask = view.instance == 3
    points = unproject_depth(view.depth, INTRINSICS, FACING_X, mask)

    assert mask.sum() > 50
    assert np.all(points[:, 0] > 2.5 - 1e-3)
    assert np.all(points[:, 0] < 3.0 + 1e-3)
    assert np.all(np.abs(points[:, 1] - 1.5) < 0.25 + 1e-3)


def test_object_color_follows_base_color_with_shading() -> None:
    view = RenderScene(ROOM, [crate_mesh()]).render(INTRINSICS, FACING_X)
    pixels = view.rgb[view.instance == 0].astype(np.float64)

    ratio = pixels / pixels.max(axis=1, keepdims=True)
    np.testing.assert_allclose(
        ratio, [[1.0, 0.2 / 0.9, 0.1 / 0.9]] * len(ratio), atol=0.03
    )
    assert view.rgb.dtype == np.uint8


def test_shell_is_textured_and_render_is_deterministic() -> None:
    scene = RenderScene(ROOM, [crate_mesh()])
    first = scene.render(INTRINSICS, FACING_X)
    second = scene.render(INTRINSICS, FACING_X)

    wall = first.rgb[first.instance == -1].astype(np.float64)
    assert wall.std(axis=0).min() > 2.0
    np.testing.assert_array_equal(first.rgb, second.rgb)
    np.testing.assert_array_equal(first.depth, second.depth)
    np.testing.assert_array_equal(first.instance, second.instance)
