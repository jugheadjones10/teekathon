import json
from pathlib import Path

import pymupdf

from gemini_paper_crop.detector import DetectionCall
from gemini_paper_crop.models import DetectedRegion, PageDetections
from gemini_paper_crop.pipeline import run_paper
from gemini_paper_crop.report import write_paper_report, write_run_index


class FakeDetector:
    model = "fake-model"

    def __init__(self) -> None:
        self.call_count = 0

    def detect_page(
        self, page_path: Path, *, page_number: int, prompt: str
    ) -> DetectionCall:
        self.call_count += 1
        detections = PageDetections(
            page_number=page_number,
            regions=[
                DetectedRegion(
                    type="oe_question",
                    question_number=str(page_number + 20),
                    fragment_index=1,
                    box_2d=[100, 100, 900, 900],
                    needs_review=False,
                )
            ],
        )
        return DetectionCall(
            detections=detections,
            raw_text=detections.model_dump_json(),
            interaction_id=f"fake-{page_number}",
            usage={"total_tokens": 100},
            elapsed_seconds=0.25,
        )


class FailingDetector:
    model = "fake-model"

    def detect_page(
        self, page_path: Path, *, page_number: int, prompt: str
    ) -> DetectionCall:
        raise RuntimeError("temporary failure")


def make_pdf(path: Path, page_count: int = 1) -> None:
    document = pymupdf.open()
    for _ in range(page_count):
        document.new_page(width=72, height=144)
    document.save(path)
    document.close()


def test_run_paper_persists_pages_detections_overlays_and_crops(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    make_pdf(pdf_path)
    detector = FakeDetector()

    result = run_paper(
        pdf_path,
        tmp_path / "run" / "paper",
        detector=detector,
        prompt="Find regions",
        dpi=72,
        padding=0,
    )

    assert result.pdf_name == "paper.pdf"
    assert len(result.pages) == 1
    page = result.pages[0]
    assert page.error is None
    assert page.page_path.exists()
    assert page.overlay_path is not None and page.overlay_path.exists()
    assert len(page.crops) == 1 and page.crops[0].path.exists()
    record = json.loads(page.detection_path.read_text())
    assert record["model"] == "fake-model"
    assert record["detections"]["regions"][0]["question_number"] == "21"


def test_run_paper_resume_reuses_saved_detection(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    make_pdf(pdf_path)
    detector = FakeDetector()
    paper_dir = tmp_path / "run" / "paper"
    run_paper(pdf_path, paper_dir, detector=detector, prompt="Find regions", dpi=72)

    resumed = run_paper(
        pdf_path,
        paper_dir,
        detector=detector,
        prompt="Find regions",
        dpi=72,
        resume=True,
    )

    assert detector.call_count == 1
    assert resumed.pages[0].resumed is True


def test_run_paper_resume_retries_a_saved_error(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    make_pdf(pdf_path)
    paper_dir = tmp_path / "run" / "paper"
    failed = run_paper(
        pdf_path,
        paper_dir,
        detector=FailingDetector(),
        prompt="Find regions",
        dpi=72,
    )
    detector = FakeDetector()

    resumed = run_paper(
        pdf_path,
        paper_dir,
        detector=detector,
        prompt="Find regions",
        dpi=72,
        resume=True,
    )

    assert failed.pages[0].error is not None
    assert detector.call_count == 1
    assert resumed.pages[0].error is None
    assert resumed.pages[0].resumed is False


def test_write_paper_report_links_visual_artifacts(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    make_pdf(pdf_path)
    paper_dir = tmp_path / "run" / "paper"
    result = run_paper(
        pdf_path,
        paper_dir,
        detector=FakeDetector(),
        prompt="Find regions",
        dpi=72,
    )

    report_path = write_paper_report(result)

    html = report_path.read_text()
    assert "paper.pdf" in html
    assert "page-001.png" in html
    assert "oe_question" in html
    assert "Question 21" in html


def test_write_run_index_links_each_paper_report(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    make_pdf(pdf_path)
    run_dir = tmp_path / "run"
    result = run_paper(
        pdf_path,
        run_dir / "paper",
        detector=FakeDetector(),
        prompt="Find regions",
        dpi=72,
    )
    write_paper_report(result)

    index_path = write_run_index(run_dir, [result])

    html = index_path.read_text()
    assert "paper.pdf" in html
    assert 'href="paper/report.html"' in html
    assert "1 detected region" in html
