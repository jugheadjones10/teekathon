import math
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from gemini_paper_crop.models import PageDetections

PixelBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class CropArtifact:
    path: Path
    pixel_box: PixelBox
    region_index: int


@dataclass(frozen=True)
class PageArtifacts:
    overlay_path: Path
    crops: list[CropArtifact]


def normalized_box_to_pixels(
    box_2d: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    padding: int,
) -> PixelBox:
    ymin, xmin, ymax, xmax = box_2d
    left = max(0, math.floor(xmin * width / 1000) - padding)
    top = max(0, math.floor(ymin * height / 1000) - padding)
    right = min(width, math.ceil(xmax * width / 1000) + padding)
    bottom = min(height, math.ceil(ymax * height / 1000) + padding)
    return left, top, right, bottom


def _safe_label(value: str | None) -> str:
    if value is None:
        return "unlabelled"
    cleaned = re.sub(r"[^a-zA-Z0-9.-]+", "-", value).strip("-")
    return cleaned or "unlabelled"


def materialize_page_detections(
    page_path: Path,
    detections: PageDetections,
    output_dir: Path,
    *,
    padding: int = 12,
) -> PageArtifacts:
    overlays_dir = output_dir / "overlays"
    crops_dir = output_dir / "crops"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    crop_artifacts: list[CropArtifact] = []
    with Image.open(page_path) as source:
        page = source.convert("RGB")
        overlay = page.copy()
        draw = ImageDraw.Draw(overlay)

        for index, region in enumerate(detections.regions, start=1):
            pixel_box = normalized_box_to_pixels(
                region.box_2d,
                width=page.width,
                height=page.height,
                padding=padding,
            )
            region_dir = crops_dir / region.type
            region_dir.mkdir(parents=True, exist_ok=True)
            label = _safe_label(region.question_number)
            filename = (
                f"page-{detections.page_number:03d}_region-{index:02d}_"
                f"{label}_fragment-{region.fragment_index}.png"
            )
            crop_path = region_dir / filename
            page.crop(pixel_box).save(crop_path)
            crop_artifacts.append(
                CropArtifact(
                    path=crop_path,
                    pixel_box=pixel_box,
                    region_index=index,
                )
            )

            color = "#d62728" if region.needs_review else "#087f5b"
            draw.rectangle(pixel_box, outline=color, width=5)
            draw.text(
                (pixel_box[0] + 6, pixel_box[1] + 6),
                f"{index}: {region.type} {label}",
                fill=color,
                stroke_width=2,
                stroke_fill="white",
            )

        overlay_path = overlays_dir / page_path.name
        overlay.save(overlay_path)

    return PageArtifacts(overlay_path=overlay_path, crops=crop_artifacts)
