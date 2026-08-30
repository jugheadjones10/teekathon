from pathlib import Path

from PIL import Image

from gemini_paper_crop.images import (
    materialize_page_detections,
    normalized_box_to_pixels,
)
from gemini_paper_crop.models import DetectedRegion, PageDetections


def test_normalized_box_to_pixels_descales_and_adds_padding() -> None:
    box = normalized_box_to_pixels(
        (100, 200, 900, 800),
        width=1000,
        height=2000,
        padding=10,
    )

    assert box == (190, 190, 810, 1810)


def test_normalized_box_to_pixels_clamps_padding_to_image_edges() -> None:
    box = normalized_box_to_pixels(
        (0, 0, 1000, 1000),
        width=600,
        height=800,
        padding=50,
    )

    assert box == (0, 0, 600, 800)


def test_materialize_page_detections_writes_crop_and_overlay(tmp_path: Path) -> None:
    page_path = tmp_path / "page-001.png"
    Image.new("RGB", (1000, 2000), "white").save(page_path)
    result = PageDetections(
        page_number=1,
        regions=[
            DetectedRegion(
                type="oe_question",
                question_number="29",
                fragment_index=1,
                box_2d=(100, 200, 900, 800),
                needs_review=False,
            )
        ],
    )

    artifacts = materialize_page_detections(
        page_path,
        result,
        tmp_path / "paper-run",
        padding=0,
    )

    assert artifacts.overlay_path.exists()
    assert len(artifacts.crops) == 1
    crop = artifacts.crops[0]
    assert crop.path.exists()
    assert crop.pixel_box == (200, 200, 800, 1800)
    with Image.open(crop.path) as image:
        assert image.size == (600, 1600)
