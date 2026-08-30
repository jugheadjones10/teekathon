import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from gemini_paper_crop.detector import DetectionCall
from gemini_paper_crop.images import CropArtifact, materialize_page_detections
from gemini_paper_crop.models import PageDetections
from gemini_paper_crop.render import render_pdf


class PageDetector(Protocol):
    model: str

    def detect_page(
        self, page_path: Path, *, page_number: int, prompt: str
    ) -> DetectionCall: ...


@dataclass(frozen=True)
class PageRunResult:
    page_number: int
    page_path: Path
    detection_path: Path
    detections: PageDetections | None
    overlay_path: Path | None
    crops: list[CropArtifact]
    interaction_id: str | None
    usage: dict[str, Any] | None
    elapsed_seconds: float | None
    resumed: bool
    error: str | None


@dataclass(frozen=True)
class PaperRunResult:
    pdf_name: str
    pdf_path: Path
    paper_dir: Path
    model: str
    pages: list[PageRunResult]


def _read_saved_call(path: Path) -> DetectionCall:
    record = json.loads(path.read_text())
    detections = PageDetections.model_validate(record["detections"])
    return DetectionCall(
        detections=detections,
        raw_text=record.get("raw_text", detections.model_dump_json()),
        interaction_id=record.get("interaction_id"),
        usage=record.get("usage"),
        elapsed_seconds=record.get("elapsed_seconds", 0.0),
    )


def _write_call(path: Path, model: str, call: DetectionCall) -> None:
    record = {
        "model": model,
        "interaction_id": call.interaction_id,
        "elapsed_seconds": call.elapsed_seconds,
        "usage": call.usage,
        "detections": call.detections.model_dump(mode="json"),
        "raw_text": call.raw_text,
    }
    path.write_text(json.dumps(record, indent=2) + "\n")


def run_paper(
    pdf_path: Path,
    paper_dir: Path,
    *,
    detector: PageDetector,
    prompt: str,
    dpi: int = 200,
    padding: int = 12,
    page_limit: int | None = None,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> PaperRunResult:
    pages_dir = paper_dir / "pages"
    detections_dir = paper_dir / "detections"
    detections_dir.mkdir(parents=True, exist_ok=True)
    rendered_pages = render_pdf(
        pdf_path,
        pages_dir,
        dpi=dpi,
        page_limit=page_limit,
    )

    page_results: list[PageRunResult] = []
    for rendered in rendered_pages:
        if progress:
            progress(
                f"  page {rendered.page_number}/{len(rendered_pages)}"
                f"{' (resume)' if resume else ''}"
            )
        detection_path = detections_dir / f"page-{rendered.page_number:03d}.json"
        was_resumed = False
        try:
            if resume and detection_path.exists():
                call = _read_saved_call(detection_path)
                was_resumed = True
            else:
                call = detector.detect_page(
                    rendered.path,
                    page_number=rendered.page_number,
                    prompt=prompt,
                )
                _write_call(detection_path, detector.model, call)

            artifacts = materialize_page_detections(
                rendered.path,
                call.detections,
                paper_dir,
                padding=padding,
            )
            page_results.append(
                PageRunResult(
                    page_number=rendered.page_number,
                    page_path=rendered.path,
                    detection_path=detection_path,
                    detections=call.detections,
                    overlay_path=artifacts.overlay_path,
                    crops=artifacts.crops,
                    interaction_id=call.interaction_id,
                    usage=call.usage,
                    elapsed_seconds=call.elapsed_seconds,
                    resumed=was_resumed,
                    error=None,
                )
            )
        except Exception as error:
            error_record = {
                "model": detector.model,
                "page_number": rendered.page_number,
                "error": f"{type(error).__name__}: {error}",
            }
            detection_path.write_text(json.dumps(error_record, indent=2) + "\n")
            page_results.append(
                PageRunResult(
                    page_number=rendered.page_number,
                    page_path=rendered.path,
                    detection_path=detection_path,
                    detections=None,
                    overlay_path=None,
                    crops=[],
                    interaction_id=None,
                    usage=None,
                    elapsed_seconds=None,
                    resumed=False,
                    error=error_record["error"],
                )
            )

    return PaperRunResult(
        pdf_name=pdf_path.name,
        pdf_path=pdf_path,
        paper_dir=paper_dir,
        model=detector.model,
        pages=page_results,
    )
