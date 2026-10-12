import spatial_change.reporting as reporting


def test_package_exports_the_public_names() -> None:
    assert reporting.__all__ == sorted(reporting.__all__)
    assert set(reporting.__all__) == {
        "ChangeCrops",
        "HtmlConfig",
        "ReportInputs",
        "render_report",
        "write_report",
    }
    for name in reporting.__all__:
        assert hasattr(reporting, name)
