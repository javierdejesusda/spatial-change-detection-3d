import base64
import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from report_scenes import (
    box,
    cards,
    change,
    empty_report,
    inputs,
    one_of_each_report,
    scene_data,
)

from spatial_change.domain import ChangeReport, ChangeType, Epoch
from spatial_change.reporting import (
    ChangeCrops,
    HtmlConfig,
    ReportInputs,
    render_report,
    write_report,
)
from spatial_change.reporting import html as html_module
from spatial_change.reporting._page import SCRIPT
from spatial_change.visualization import ScenePoints, change_color_hex

LINE_SEPARATOR = chr(0x2028)
PARAGRAPH_SEPARATOR = chr(0x2029)
LONE_SURROGATE = chr(0xD800)


def _crop(height: int = 40, width: int = 60, value: int = 100) -> Any:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _decode(uri: str) -> Image.Image:
    head, payload = uri.split(",", 1)
    assert head.startswith("data:image/")
    return Image.open(io.BytesIO(base64.b64decode(payload)))


def _boxless() -> ChangeReport:
    return ChangeReport(
        changes=[change("c1", ChangeType.UNCERTAIN, "rug", None, None)],
        unobserved_regions=[],
        metadata={},
    )


def _single(label: str, **metadata: Any) -> ChangeReport:
    return ChangeReport(
        changes=[change("c1", ChangeType.ADDED, label, None, box((0, 0, 0)))],
        unobserved_regions=[],
        metadata=metadata,
    )


def test_there_is_one_card_per_change_in_report_order() -> None:
    page = render_report(inputs())

    ids = re.findall(r'<article class="card" id="change-(\w+)"', page)
    assert ids == ["c1", "c2", "c3", "c4"]


def test_each_card_has_a_badge_of_its_type_and_color() -> None:
    page = render_report(inputs())

    kinds = [
        ChangeType.MOVED,
        ChangeType.ADDED,
        ChangeType.REMOVED,
        ChangeType.UNCERTAIN,
    ]
    for card, kind in zip(cards(page), kinds, strict=True):
        assert f'data-type="{kind.value}"' in card
        assert f"--c:{change_color_hex(kind)}" in card
        assert re.search(
            rf'<span class="badge">.*?{kind.value.capitalize()}</span>',
            card,
            re.S,
        )


def test_the_card_shows_formatted_numbers() -> None:
    page = render_report(inputs())

    moved, added, *_ = cards(page)
    assert "87%" in moved
    assert "1.25 m" in moved
    assert "30.0°" in moved
    assert "1.25 m" not in added
    assert "Displacement" not in added
    assert re.search(r"Consistent</dt>\s*<dd>5</dd>", moved)
    assert re.search(r"Free space</dt>\s*<dd>3</dd>", moved)


def test_the_card_shows_label_and_description() -> None:
    page = render_report(inputs())

    moved = cards(page)[0]
    assert "<h3>chair</h3>" in moved
    assert "It changed." in moved


def test_a_crop_becomes_an_image_with_alt_text() -> None:
    crops = {"c1": ChangeCrops(before=_crop(), after=_crop())}

    page = render_report(inputs(crops=crops))

    images = re.findall(
        r'<img src="(data:[^"]+)" alt="([^"]+)"', cards(page)[0]
    )
    assert [alt for _, alt in images] == [
        "chair in the before capture",
        "chair in the after capture",
    ]
    assert _decode(images[0][0]).size == (60, 40)


def test_crops_are_downscaled_to_the_long_side() -> None:
    crops = {"c1": ChangeCrops(before=_crop(200, 400), after=_crop(30, 20))}

    page = render_report(inputs(crops=crops), HtmlConfig(crop_long_side=100))

    uris = re.findall(r'<img src="(data:[^"]+)"', cards(page)[0])
    assert _decode(uris[0]).size == (100, 50)
    assert _decode(uris[1]).size == (20, 30)


def test_a_missing_crop_gets_a_placeholder_that_says_why() -> None:
    crops = {"c2": ChangeCrops(before=None, after=None)}

    page = render_report(inputs(crops=crops))

    added = cards(page)[1]
    assert "<img" not in added
    assert "Not present in the before capture" in added
    assert "No crop available for the after capture" in added


def test_a_change_without_a_crops_entry_has_placeholders_too() -> None:
    moved = cards(render_report(inputs()))[0]

    assert "No crop available for the before capture" in moved
    assert "No crop available for the after capture" in moved


def test_added_and_removed_changes_say_not_present() -> None:
    _, added, removed, _ = cards(render_report(inputs()))

    assert "Not present in the before capture" in added
    assert "Not present in the after capture" in removed


def test_an_uncertain_change_is_not_called_absent() -> None:
    (card,) = cards(render_report(inputs(_boxless())))

    assert "Not localized in the before capture" in card
    assert "Not localized in the after capture" in card
    assert "Not present" not in card


def test_only_a_change_with_geometry_can_be_shown_in_3d() -> None:
    page = render_report(inputs())
    (boxless,) = cards(render_report(inputs(_boxless())))

    assert all('data-focus="' in card for card in cards(page))
    assert "data-focus" not in boxless
    assert "Show in 3D view" not in boxless


def test_unobserved_regions_are_listed_and_not_called_changes() -> None:
    page = render_report(inputs())

    match = re.search(r'<section id="unobserved".*?</section>', page, re.S)
    assert match is not None
    section = match.group(0).lower()
    assert "u1" in section
    assert "only the before capture" in section
    assert "not changes" in section


def test_no_regions_says_so() -> None:
    page = render_report(inputs(empty_report()))

    assert "Every region was seen in both captures" in page


def test_the_header_has_the_summary_and_counts() -> None:
    page = render_report(inputs())

    assert "Two captures of a living room." in page
    for kind in ChangeType:
        assert re.search(
            rf'class="tally-item"[^>]*data-type="{kind.value}"', page
        )
    assert "4 changes" in page


def test_an_empty_report_is_a_calm_page() -> None:
    page = render_report(inputs(empty_report()))

    assert "No changes were detected" in page
    assert cards(page) == []
    assert "0 changes" in page


def test_the_page_is_deterministic() -> None:
    crops = {"c1": ChangeCrops(before=_crop(), after=_crop())}

    assert render_report(inputs(crops=crops)) == render_report(
        inputs(crops=crops)
    )


def test_model_text_is_escaped_in_the_markup() -> None:
    evil = "<script>alert(1)</script>"
    report = ChangeReport(
        changes=[
            change(
                "c1",
                ChangeType.ADDED,
                evil,
                None,
                box((0, 0, 0)),
                evil + ' "quoted" & <b>',
            )
        ],
        unobserved_regions=[],
        metadata={},
    )

    page = render_report(inputs(report, summary=evil))

    assert evil not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert page.count("<script") == 2
    assert page.count("</script>") == 2
    assert "<b>" not in page


def test_model_text_cannot_break_out_of_the_json_block() -> None:
    label = "</script><img src=x onerror=alert(1)>"

    page = render_report(inputs(_single(label)))

    assert "</script><img" not in page
    assert "onerror=alert(1)>" not in page
    assert scene_data(page)["changes"][0]["label"] == label
    assert page.count("<script") == 2


def test_the_title_is_escaped() -> None:
    page = render_report(inputs(), HtmlConfig(title="A & B <i>"))

    assert "<title>A &amp; B &lt;i&gt;</title>" in page


def test_line_separators_in_text_do_not_reach_the_json_block() -> None:
    label = f"a{LINE_SEPARATOR}b{PARAGRAPH_SEPARATOR}c"

    page = render_report(inputs(_single(label)))

    match = re.search(r'id="scene-data">(.*?)</script>', page, re.S)
    assert match is not None
    assert LINE_SEPARATOR not in match.group(1)
    assert PARAGRAPH_SEPARATOR not in match.group(1)
    assert scene_data(page)["changes"][0]["label"] == label


def test_the_source_writes_line_separators_as_escapes() -> None:
    source = Path(html_module.__file__).read_text(encoding="utf-8")

    assert LINE_SEPARATOR not in source
    assert PARAGRAPH_SEPARATOR not in source


def test_a_lone_surrogate_is_replaced_not_fatal(tmp_path: Path) -> None:
    bad = f"a{LONE_SURROGATE}b"
    page_inputs = inputs(_single(bad, note=bad), summary=bad)
    path = tmp_path / "report.html"

    write_report(page_inputs, path)

    assert "a�b" in path.read_text(encoding="utf-8")
    assert LONE_SURROGATE not in render_report(page_inputs)


def test_the_page_is_self_contained() -> None:
    crops = {"c1": ChangeCrops(before=_crop(), after=_crop())}

    page = render_report(inputs(crops=crops))

    assert "http://" not in page
    assert "https://" not in page
    assert page.count("<link") == 1
    assert '<link rel="icon" href="data:,">' in page
    assert "@import" not in page
    assert "url(" not in page
    assert not re.search(r"<script[^>]*\ssrc=", page)
    for value in re.findall(r'\s(?:src|href)="([^"]*)"', page):
        assert value.startswith(("data:", "#")), value
    assert not re.search(r"""(?:fetch|XMLHttpRequest|import)\s*\(""", page)
    assert "//" not in re.sub(r"data:[^\"]+", "", page).replace("</", "")


def test_the_page_is_accessible_and_responsive() -> None:
    crops = {"c1": ChangeCrops(before=_crop(), after=_crop())}

    page = render_report(inputs(crops=crops))

    assert page.startswith("<!doctype html>")
    assert '<html lang="en">' in page
    assert (
        'name="viewport" content="width=device-width, initial-scale=1"' in page
    )
    assert "prefers-color-scheme: dark" in page
    assert "prefers-reduced-motion" in page
    assert "max-width: 640px" in page
    assert ":focus-visible" in page
    for tag in re.findall(r"<img\b[^>]*>", page):
        assert re.search(r'alt="[^"]+"', tag)
    assert re.search(r'<canvas[^>]*aria-label="[^"]+"', page)
    assert "<button" in page


def test_the_data_block_has_capped_millimeter_points() -> None:
    points = ScenePoints(
        np.array([[1.23456, -0.0004, 2.0]] + [[0.0, 0.0, 0.0]] * 499),
        np.tile(np.array([[10, 20, 30]], dtype=np.uint8), (500, 1)),
    )
    page_inputs = ReportInputs(
        report=one_of_each_report(),
        summary="s",
        crops={},
        points={Epoch.BEFORE: points},
    )

    page = render_report(page_inputs, HtmlConfig(max_points=100))

    before = scene_data(page)["epochs"]["before"]
    assert len(before["points"]) == 300
    assert len(before["colors"]) == 100
    assert before["points"][:3] == [1235, 0, 2000]
    assert before["colors"][0] == (10 << 16) | (20 << 8) | 30
    assert all(isinstance(v, int) for v in before["points"])
    assert scene_data(page)["epochs"]["after"] == {"points": [], "colors": []}


def test_the_data_block_describes_changes_for_the_viewer() -> None:
    data = scene_data(render_report(inputs()))

    moved, added, removed, uncertain = data["changes"]
    assert (moved["id"], moved["type"], moved["label"]) == (
        "c1",
        "moved",
        "chair",
    )
    assert all("color" not in item for item in data["changes"])
    assert [line["role"] for line in moved["lines"]] == [
        "before_box",
        "after_box",
        "arrow",
    ]
    assert len(moved["lines"][0]["segments"]) == 12
    assert len(moved["lines"][2]["segments"]) == 3
    assert [line["role"] for line in added["lines"]] == ["after_box"]
    assert [line["role"] for line in removed["lines"]] == ["before_box"]
    assert len(uncertain["lines"]) == 2
    first = moved["lines"][0]["segments"][0]
    assert len(first) == 6
    assert all(isinstance(v, int) for v in first)


def test_a_change_with_no_geometry_has_no_lines() -> None:
    data = scene_data(render_report(inputs(_boxless())))

    assert data["changes"][0]["lines"] == []


def test_regions_and_the_color_map_are_in_the_data_block() -> None:
    data = scene_data(render_report(inputs()))

    assert [r["id"] for r in data["regions"]] == ["u1"]
    assert len(data["regions"][0]["segments"]) == 12
    assert set(data["colors"]) == {k.value for k in ChangeType}
    assert data["colors"]["moved"] == change_color_hex(ChangeType.MOVED)


def test_the_view_is_centered_on_the_content() -> None:
    data = scene_data(render_report(inputs()))

    center, radius = data["view"]["center"], data["view"]["radius"]
    assert len(center) == 3
    assert 0 < center[0] < 6000
    assert radius > 1000


def test_the_view_of_an_empty_scene_has_a_default() -> None:
    empty = ReportInputs(report=empty_report(), summary="", crops={}, points={})

    data = scene_data(render_report(empty))

    assert data["view"] == {"center": [0, 0, 0], "radius": 1000}
    assert data["changes"] == []


def test_the_viewer_reads_colors_from_the_data_map() -> None:
    assert "data.colors[" in SCRIPT
    assert not re.search(r"#[0-9a-fA-F]{6}", SCRIPT)


def test_the_viewer_resizes_when_width_or_height_changes() -> None:
    assert "canvas.width !==" in SCRIPT
    assert "canvas.height !==" in SCRIPT


def test_the_viewer_clips_segments_at_the_near_plane() -> None:
    assert "function clip" in SCRIPT
    assert "near" in SCRIPT


def test_the_viewer_reuses_one_image_buffer() -> None:
    assert SCRIPT.count("createImageData") == 1
    assert "pixels.fill(0)" in SCRIPT


def test_write_report_writes_utf8_and_creates_directories(
    tmp_path: Path,
) -> None:
    path = tmp_path / "out" / "report.html"

    write_report(inputs(summary="café"), path)

    text = path.read_text(encoding="utf-8")
    assert text == render_report(inputs(summary="café"))
    assert "café" in text


def test_crops_for_unknown_changes_are_rejected() -> None:
    crops = {"nope": ChangeCrops(before=None, after=None)}

    with pytest.raises(ValueError, match="nope"):
        inputs(crops=crops)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.float32),
        np.zeros((0, 4, 3), dtype=np.uint8),
    ],
)
def test_crops_must_be_nonempty_uint8_rgb(bad: Any) -> None:
    with pytest.raises(ValueError, match="before"):
        ChangeCrops(before=bad, after=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_points", 0), ("crop_long_side", 7), ("jpeg_quality", 0)],
)
def test_config_rejects_invalid_values(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        HtmlConfig(**{field: value})
