from typing import Literal

from pydantic import BaseModel, Field, field_validator

RegionType = Literal["mcq_question", "oe_question", "ak_mcq", "ak_oe"]


class DetectedRegion(BaseModel):
    type: RegionType = Field(description="The kind of question or answer region.")
    question_number: str | None = Field(
        default=None,
        max_length=32,
        description="Printed question number, or null for an MCQ answer table.",
    )
    fragment_index: int = Field(
        default=1,
        ge=1,
        description="Reading-order fragment number when an item spans pages.",
    )
    box_2d: list[int] = Field(
        min_length=4,
        max_length=4,
        description=(
            "Bounding box [ymin, xmin, ymax, xmax], normalized to integers 0-1000."
        ),
    )
    needs_review: bool = Field(
        default=False,
        description="True when the region boundary or label is uncertain.",
    )

    @field_validator("box_2d")
    @classmethod
    def validate_box(cls, box: list[int]) -> list[int]:
        ymin, xmin, ymax, xmax = box
        if any(coordinate < 0 or coordinate > 1000 for coordinate in box):
            raise ValueError("box coordinates must be between 0 and 1000")
        if ymin >= ymax or xmin >= xmax:
            raise ValueError("box must have positive width and height")
        return box


class PageDetections(BaseModel):
    page_number: int = Field(ge=1, description="One-based source PDF page number.")
    regions: list[DetectedRegion] = Field(default_factory=list, max_length=100)
