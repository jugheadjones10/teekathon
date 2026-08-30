from typing import Literal

from pydantic import BaseModel, Field, field_validator

RegionType = Literal["mcq_question", "oe_question", "ak_mcq", "ak_oe"]


class DetectedRegion(BaseModel):
    type: RegionType
    question_number: str | None = None
    fragment_index: int = Field(default=1, ge=1)
    box_2d: tuple[int, int, int, int]
    needs_review: bool = False

    @field_validator("box_2d")
    @classmethod
    def validate_box(cls, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        ymin, xmin, ymax, xmax = box
        if any(coordinate < 0 or coordinate > 1000 for coordinate in box):
            raise ValueError("box coordinates must be between 0 and 1000")
        if ymin >= ymax or xmin >= xmax:
            raise ValueError("box must have positive width and height")
        return box


class PageDetections(BaseModel):
    page_number: int = Field(ge=1)
    regions: list[DetectedRegion] = Field(default_factory=list)
