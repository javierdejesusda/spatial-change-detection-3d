"""The self-contained HTML report of a change report.

The page is one file: inline style, inline script, and images as base64
data URIs, so it opens offline and makes no request. Every text that comes
from the report is escaped for HTML, and the data of the 3D view is a JSON
block in which ``<``, ``>``, and ``&`` are written as escapes, so no text
can close the block or start markup. The view code reads that block and
never builds code from text.

The room frame is z-up. The 3D view receives coordinates as integer
millimeters.
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import (
    Change,
    ChangeReport,
    ChangeType,
    Epoch,
    UnobservedRegion,
)
from spatial_change.domain.transforms import check_array
from spatial_change.reporting._page import SCRIPT, STYLE
from spatial_change.visualization.palette import (
    CHANGE_COLORS,
    change_color_hex,
)
from spatial_change.visualization.points import ScenePoints, subsample
from spatial_change.visualization.segments import (
    Role,
    change_segments,
    region_segments,
)

_MIN_LONG_SIDE = 8
_MAX_JPEG_QUALITY = 95
_DEFAULT_RADIUS_MM = 1000
_EVIDENCE_LABELS = (
    ("consistent", "Consistent"),
    ("free", "Free space"),
    ("occluded", "Occluded"),
    ("unobserved", "Out of view"),
)
_SURROGATES = re.compile("[\ud800-\udfff]")
_JSON_ESCAPES = {
    "<": "\\u003c",
    ">": "\\u003e",
    "&": "\\u0026",
    "\u2028": "\\u2028",
    "\u2029": "\\u2029",
}


@dataclass(frozen=True)
class HtmlConfig:
    """Parameters of the report page.

    Attributes:
        title: Page title and heading.
        max_points: Largest number of points per epoch in the 3D view.
        crop_long_side: Longest side in pixels of an embedded crop; larger
            crops are scaled down, smaller ones are kept.
        jpeg_quality: JPEG quality of the embedded crops, in 1 to 95.
    """

    title: str = "Scene change report"
    max_points: int = 20_000
    crop_long_side: int = 320
    jpeg_quality: int = 85

    def __post_init__(self) -> None:
        """Checks the values.

        Raises:
            ValueError: If a number is out of range.
        """
        if self.max_points < 1:
            raise ValueError(
                f"max_points must be at least 1, got {self.max_points}"
            )
        if self.crop_long_side < _MIN_LONG_SIDE:
            raise ValueError(
                f"crop_long_side must be at least {_MIN_LONG_SIDE}, "
                f"got {self.crop_long_side}"
            )
        if not 1 <= self.jpeg_quality <= _MAX_JPEG_QUALITY:
            raise ValueError(
                f"jpeg_quality must be in 1 to {_MAX_JPEG_QUALITY}, "
                f"got {self.jpeg_quality}"
            )


def _check_crop(image: NDArray[np.uint8] | None, name: str) -> None:
    if image is None:
        return
    check_array(image, (None, None, 3), np.uint8, name)
    if image.size == 0:
        raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, eq=False)
class ChangeCrops:
    """The views of one change in each capture.

    Attributes:
        before: Crop of the object in the ``before`` capture as an
            ``(H, W, 3)`` uint8 RGB array, or ``None`` if there is none.
        after: Crop in the ``after`` capture, or ``None``.
    """

    before: NDArray[np.uint8] | None
    after: NDArray[np.uint8] | None

    def __post_init__(self) -> None:
        """Checks the arrays.

        Raises:
            ValueError: If a crop is not a non-empty uint8 RGB image.
        """
        _check_crop(self.before, "before")
        _check_crop(self.after, "after")


@dataclass(frozen=True, eq=False)
class ReportInputs:
    """Everything the report page shows.

    Attributes:
        report: The detected changes and unobserved regions.
        summary: One short paragraph about the scene, shown under the
            title.
        crops: Crops by change id. A change without an entry has none.
        points: Colored points per epoch in the z-up room frame for the
            3D view. A missing epoch is not drawn.
    """

    report: ChangeReport
    summary: str
    crops: Mapping[str, ChangeCrops]
    points: Mapping[Epoch, ScenePoints]

    def __post_init__(self) -> None:
        """Checks that every crop belongs to a change.

        Raises:
            ValueError: If a crop names a change that is not in the report.
        """
        known = {c.change_id for c in self.report.changes}
        unknown = sorted(set(self.crops) - known)
        if unknown:
            raise ValueError(f"crops for unknown changes: {unknown}")


def _clean(text: object) -> str:
    """Replaces lone surrogates, which no UTF-8 file can hold."""
    return _SURROGATES.sub("\ufffd", str(text))


def _esc(text: object) -> str:
    return html.escape(_clean(text), quote=True)


def _image_uri(
    image: NDArray[np.uint8], config: HtmlConfig
) -> tuple[str, int, int]:
    """Encodes a crop as a JPEG data URI, scaled down to the long side."""
    from PIL import Image

    picture = Image.fromarray(image)
    scale = config.crop_long_side / max(picture.size)
    if scale < 1.0:
        width, height = picture.size
        picture = picture.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG", quality=config.jpeg_quality)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    width, height = picture.size
    return f"data:image/jpeg;base64,{payload}", width, height


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _tally(changes: list[Change]) -> str:
    counts = {kind: 0 for kind in ChangeType}
    for change in changes:
        counts[change.type] += 1
    segments = "".join(
        f'<span class="tally-seg" style="--c:{change_color_hex(kind)};'
        f'--n:{n}"></span>'
        for kind, n in counts.items()
        if n
    )
    items = "".join(
        f'<li class="tally-item" data-type="{kind.value}" '
        f'style="--c:{change_color_hex(kind)}">'
        f'<span class="dot" aria-hidden="true"></span>'
        f'<span class="n">{n}</span> {kind.value.capitalize()}</li>'
        for kind, n in counts.items()
    )
    return (
        f'<div class="tally-bar" aria-hidden="true">{segments}</div>'
        f'<ul class="tally-list">{items}</ul>'
        f'<p class="tally-total">{_plural(len(changes), "change")}</p>'
    )


def _crop_figure(
    change: Change,
    epoch: Epoch,
    image: NDArray[np.uint8] | None,
    config: HtmlConfig,
) -> str:
    box = change.before_box if epoch is Epoch.BEFORE else change.after_box
    if image is not None:
        uri, width, height = _image_uri(image, config)
        alt = _esc(f"{change.label} in the {epoch.value} capture")
        inner = (
            f'<div class="crop-frame"><img src="{uri}" alt="{alt}" '
            f'width="{width}" height="{height}" decoding="async"></div>'
        )
    else:
        if box is not None:
            text = f"No crop available for the {epoch.value} capture"
        elif change.type is ChangeType.UNCERTAIN:
            text = f"Not localized in the {epoch.value} capture"
        else:
            text = f"Not present in the {epoch.value} capture"
        inner = f'<div class="crop-frame empty"><span>{text}</span></div>'
    caption = epoch.value.capitalize()
    return f"<figure>{inner}<figcaption>{caption}</figcaption></figure>"


def _card(change: Change, crops: ChangeCrops | None, config: HtmlConfig) -> str:
    kind = change.type
    facts = [("Confidence", f"{change.confidence * 100:.0f}%")]
    if change.displacement_m is not None:
        facts.append(("Displacement", f"{change.displacement_m:.2f} m"))
    if change.rotation_deg is not None:
        facts.append(("Rotation", f"{change.rotation_deg:.1f}°"))
    facts_html = "".join(
        f"<div><dt>{name}</dt><dd>{value}</dd></div>" for name, value in facts
    )
    evidence = change.evidence.model_dump()
    evidence_html = "".join(
        f"<div><dt>{name}</dt><dd>{evidence[key]}</dd></div>"
        for key, name in _EVIDENCE_LABELS
    )
    before = crops.before if crops else None
    after = crops.after if crops else None
    ident = _esc(change.change_id)
    action = (
        '<div class="card-actions"><button type="button" class="btn" '
        f'data-focus="{ident}">Show in 3D view</button></div>'
        if change_segments(change)
        else ""
    )
    return (
        f'<article class="card" id="change-{ident}" '
        f'data-type="{kind.value}" style="--c:{change_color_hex(kind)}">'
        f'<div class="card-text"><div class="card-head">'
        f'<span class="badge"><span class="dot" aria-hidden="true"></span>'
        f"{kind.value.capitalize()}</span>"
        f'<span class="cid">{ident}</span></div>'
        f"<h3>{_esc(change.label)}</h3>"
        f'<p class="desc">{_esc(change.description)}</p>'
        f'<dl class="facts">{facts_html}</dl>'
        f'<p class="evidence-title">Visibility evidence, in observations</p>'
        f'<dl class="evidence">{evidence_html}</dl>'
        f"{action}</div>"
        f'<div class="crops">'
        f"{_crop_figure(change, Epoch.BEFORE, before, config)}"
        f"{_crop_figure(change, Epoch.AFTER, after, config)}</div>"
        f"</article>"
    )


def _changes_section(inputs: ReportInputs, config: HtmlConfig) -> str:
    changes = inputs.report.changes
    if changes:
        body = '<div class="cards">' + "".join(
            _card(change, inputs.crops.get(change.change_id), config)
            for change in changes
        )
        body += "</div>"
    else:
        body = (
            '<p class="empty-state">No changes were detected between the '
            "two captures.</p>"
        )
    return (
        '<section id="changes" aria-labelledby="changes-h">'
        f'<h2 id="changes-h">Changes</h2>{body}</section>'
    )


def _region_item(region: UnobservedRegion) -> str:
    x, y, z = region.box.extent
    return (
        f'<li><span class="rid">{_esc(region.region_id)}</span>'
        f'<span class="rtext">Seen by only the {region.epoch_seen.value} '
        f"capture</span>"
        f'<span class="rtext num">{x:.2f} x {y:.2f} x {z:.2f} m</span></li>'
    )


def _regions_section(report: ChangeReport) -> str:
    if report.unobserved_regions:
        body = (
            '<p class="note">Each region below was seen by only one capture, '
            "so nothing can be said about whether it changed. These regions "
            'are not changes.</p><ul class="regions">'
            + "".join(_region_item(r) for r in report.unobserved_regions)
            + "</ul>"
        )
    else:
        body = (
            '<p class="note">Every region was seen in both captures, so '
            "none is left unjudged.</p>"
        )
    return (
        '<section id="unobserved" aria-labelledby="unobserved-h">'
        f'<h2 id="unobserved-h">Not observed in both captures</h2>{body}'
        "</section>"
    )


def _format_value(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _run_section(report: ChangeReport) -> str:
    if not report.metadata:
        return ""
    items = "".join(
        f"<div><dt>{_esc(key)}</dt><dd>{_esc(_format_value(value))}</dd></div>"
        for key, value in report.metadata.items()
    )
    return f'<footer class="run"><h2>Run details</h2><dl>{items}</dl></footer>'


def _viewer_section() -> str:
    legend = "".join(
        f'<li style="--c:{change_color_hex(kind)}">'
        f'<span class="dot" aria-hidden="true"></span>'
        f"{kind.value.capitalize()}</li>"
        for kind in ChangeType
    )
    label = (
        "Interactive 3D view of the scene with the changed objects outlined. "
        "Drag to orbit, scroll or press plus and minus to zoom, use the "
        "arrow keys to orbit, and press 0 to reset."
    )
    return (
        '<section id="view-section" aria-labelledby="view-h">'
        '<h2 id="view-h">3D view</h2><div class="viewer">'
        f'<canvas id="view" tabindex="0" role="img" aria-label="{label}">'
        "</canvas>"
        '<div class="toolbar">'
        '<label><input type="checkbox" id="show-before" checked>'
        "Before points</label>"
        '<label><input type="checkbox" id="show-after" checked>'
        "After points</label>"
        '<span class="spacer"></span>'
        '<button type="button" class="btn" id="zoom-out" '
        'aria-label="Zoom out">−</button>'
        '<button type="button" class="btn" id="zoom-in" '
        'aria-label="Zoom in">+</button>'
        '<button type="button" class="btn" id="reset">Reset view</button>'
        "</div>"
        f'<ul class="legend">{legend}'
        "<li>Dashed outline: before. Solid outline: after.</li></ul>"
        "<noscript><p>The 3D view needs JavaScript.</p></noscript>"
        "</div></section>"
    )


def _mm(values: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.rint(values * 1000.0).astype(np.int64)


def _lines(
    parts: list[tuple[Role, NDArray[np.float64]]],
) -> list[tuple[Role, NDArray[np.int64]]]:
    """Converts segments to millimeter rows ``(x1, y1, z1, x2, y2, z2)``."""
    return [(role, _mm(seg.reshape(-1, 6))) for role, seg in parts]


def _view(extents: NDArray[np.int64]) -> dict[str, Any]:
    """Centers the view on the bounds of every point and segment end."""
    if len(extents) == 0:
        return {"center": [0, 0, 0], "radius": _DEFAULT_RADIUS_MM}
    low, high = extents.min(axis=0), extents.max(axis=0)
    radius = max(_DEFAULT_RADIUS_MM, int(np.linalg.norm(high - low) / 2))
    return {"center": ((low + high) // 2).tolist(), "radius": radius}


def _scene_data(inputs: ReportInputs, config: HtmlConfig) -> dict[str, Any]:
    report = inputs.report
    clouds = {
        epoch: subsample(
            inputs.points.get(epoch, ScenePoints.empty()), config.max_points
        )
        for epoch in Epoch
    }
    millimeters = {e: _mm(c.points) for e, c in clouds.items()}
    change_lines = [_lines(change_segments(c)) for c in report.changes]
    region_lines = [
        _lines(region_segments(r)) for r in report.unobserved_regions
    ]
    ends = [
        rows.reshape(-1, 3)
        for group in change_lines + region_lines
        for _, rows in group
    ]
    epochs = {}
    for epoch, cloud in clouds.items():
        colors = cloud.colors.astype(np.int64)
        packed = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
        epochs[epoch.value] = {
            "points": millimeters[epoch].reshape(-1).tolist(),
            "colors": packed.tolist(),
        }
    stacked = np.concatenate([*millimeters.values(), *ends])
    return {
        "view": _view(stacked),
        "colors": {k.value: change_color_hex(k) for k in CHANGE_COLORS},
        "epochs": epochs,
        "changes": [
            {
                "id": _clean(change.change_id),
                "type": change.type.value,
                "label": _clean(change.label),
                "lines": [
                    {"role": role.value, "segments": rows.tolist()}
                    for role, rows in lines
                ],
            }
            for change, lines in zip(report.changes, change_lines, strict=True)
        ],
        "regions": [
            {"id": _clean(region.region_id), "segments": rows.tolist()}
            for region, ((_, rows),) in zip(
                report.unobserved_regions, region_lines, strict=True
            )
        ],
    }


def _json_block(data: dict[str, Any]) -> str:
    text = json.dumps(data, separators=(",", ":"), allow_nan=False)
    for char, escape in _JSON_ESCAPES.items():
        text = text.replace(char, escape)
    return text


def render_report(
    inputs: ReportInputs, config: HtmlConfig | None = None
) -> str:
    """Renders the report as one self-contained HTML document.

    Args:
        inputs: The report, summary, crops, and points to show.
        config: Page parameters; the defaults when ``None``.

    Returns:
        The HTML text. Equal input gives equal output.
    """
    config = config or HtmlConfig()
    report = inputs.report
    summary = (
        f'<p class="lede">{_esc(inputs.summary)}</p>' if inputs.summary else ""
    )
    data = _json_block(_scene_data(inputs, config))
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="color-scheme" content="light dark">\n'
        '<link rel="icon" href="data:,">\n'
        f"<title>{_esc(config.title)}</title>\n"
        f"<style>{STYLE}</style>\n</head>\n<body>\n"
        '<a class="skip" href="#changes">Skip to the changes</a>\n'
        '<div class="page">\n'
        f'<header class="top"><div><h1>{_esc(config.title)}</h1>{summary}'
        f"</div><div>{_tally(report.changes)}</div></header>\n"
        f"<main>\n{_viewer_section()}\n"
        f"{_changes_section(inputs, config)}\n"
        f"{_regions_section(report)}\n</main>\n"
        f"{_run_section(report)}\n</div>\n"
        f'<script type="application/json" id="scene-data">{data}</script>\n'
        f"<script>{SCRIPT}</script>\n</body>\n</html>\n"
    )


def write_report(
    inputs: ReportInputs, path: Path, config: HtmlConfig | None = None
) -> None:
    """Writes the report page to a file.

    Args:
        inputs: The report, summary, crops, and points to show.
        path: Output file; missing parent directories are created.
        config: Page parameters; the defaults when ``None``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(inputs, config), encoding="utf-8")
