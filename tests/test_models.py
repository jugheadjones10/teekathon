import json

import pytest
from pydantic import ValidationError

from gemini_paper_crop.models import (
    DetectedRegion,
    PageDetections,
    gemini_response_schema,
)


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


def test_gemini_response_schema_is_compact_but_preserves_the_output_shape() -> None:
    schema = gemini_response_schema()
    schema_text = json.dumps(schema)
    region_schema = schema["properties"]["regions"]["items"]

    assert schema["required"] == ["page_number", "regions"]
    assert region_schema["required"] == ["type", "box_2d"]
    assert region_schema["properties"]["type"]["enum"] == [
        "mcq_question",
        "oe_question",
        "ak_mcq",
        "ak_oe",
    ]
    assert region_schema["properties"]["question_number"]["type"] == [
        "string",
        "null",
    ]
    assert "$ref" not in schema_text
    assert "$defs" not in schema_text
    assert "default" not in schema_text
    assert "minimum" not in schema_text
    assert "maxItems" not in schema_text


def test_page_detections_rejects_an_unreasonable_region_count() -> None:
    region = DetectedRegion(
        type="oe_question",
        question_number="1",
        box_2d=[10, 10, 20, 20],
    )

    with pytest.raises(ValidationError):
        PageDetections(page_number=1, regions=[region] * 101)
