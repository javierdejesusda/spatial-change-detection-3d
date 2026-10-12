"""The self-contained HTML report of a change report.

Importing the package does not load torch, open3d, cv2, rerun, or Pillow.
"""

from spatial_change.reporting.html import (
    ChangeCrops,
    HtmlConfig,
    ReportInputs,
    render_report,
    write_report,
)

__all__ = [
    "ChangeCrops",
    "HtmlConfig",
    "ReportInputs",
    "render_report",
    "write_report",
]
