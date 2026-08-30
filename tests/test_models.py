import json

import pytest
from pydantic import ValidationError

from gemini_paper_crop.models import DetectedRegion, PageDetections


def test_detected_region_accepts_a_valid_normalized_box() -> None:
    region = DetectedRegion(
        type="oe_question",
        question_number="29",
        fragment_index=1,
        box_2d=(100, 50, 900, 950),
        needs_review=False,
    )

    assert region.box_2d == [100, 50, 900, 950]


@pytest.mark.parametrize(
    "box",
    [
        (-1, 0, 100, 100),
        (0, 0, 1001, 100),
        (100, 0, 100, 100),
        (0, 100, 100, 100),
    ],
)
def test_detected_region_rejects_invalid_normalized_boxes(
    box: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValidationError):
        DetectedRegion(
            type="oe_question",
            question_number="29",
            fragment_index=1,
            box_2d=box,
            needs_review=False,
        )


def test_page_detections_defaults_to_no_regions() -> None:
    result = PageDetections(page_number=4)

    assert result.regions == []


def test_page_detection_json_schema_uses_homogeneous_arrays() -> None:
    schema_text = json.dumps(PageDetections.model_json_schema())

    assert "prefixItems" not in schema_text
