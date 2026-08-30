import json
from pathlib import Path

from teekathon_dataset.validator import validate_dataset


def test_validate_dataset_accepts_twenty_papers_per_group(tmp_path: Path) -> None:
    papers = []
    for level, subject in (
        ("P5", "Maths"),
        ("P5", "Science"),
        ("P6", "Maths"),
        ("P6", "Science"),
    ):
        for number in range(20):
            paper_dir = tmp_path / "data" / level / subject / f"paper-{number}"
            paper_dir.mkdir(parents=True)
            (paper_dir / "source.pdf").write_bytes(b"pdf")
            (paper_dir / "crop.jpg").write_bytes(b"jpg")
            gold_path = paper_dir / "gold.json"
            gold_path.write_text(
                json.dumps(
                    {
                        "regions": [
                            {
                                "type": "mcq_question",
                                "label": "1",
                                "paper": 1,
                                "fragments": ["crop.jpg"],
                            }
                        ]
                    }
                )
            )
            papers.append(
                {
                    "id": f"{level}-{subject}-{number}",
                    "level": level,
                    "subject": subject,
                    "directory": str(paper_dir.relative_to(tmp_path)),
                    "source_pdf": "source.pdf",
                    "gold_manifest": "gold.json",
                }
            )
    manifest = tmp_path / "dataset.json"
    manifest.write_text(json.dumps({"schema_version": 1, "papers": papers}))

    report = validate_dataset(manifest)

    assert report.paper_count == 80
    assert report.crop_count == 80
    assert report.errors == []


def test_validate_dataset_reports_missing_crop(tmp_path: Path) -> None:
    paper_dir = tmp_path / "data" / "P6" / "Science" / "paper"
    paper_dir.mkdir(parents=True)
    (paper_dir / "source.pdf").write_bytes(b"pdf")
    (paper_dir / "gold.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "type": "oe_question",
                        "label": "2",
                        "paper": 1,
                        "fragments": ["missing.jpg"],
                    }
                ]
            }
        )
    )
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "papers": [
                    {
                        "id": "paper",
                        "level": "P6",
                        "subject": "Science",
                        "directory": "data/P6/Science/paper",
                        "source_pdf": "source.pdf",
                        "gold_manifest": "gold.json",
                    }
                ],
            }
        )
    )

    report = validate_dataset(manifest, require_balanced_groups=False)

    assert report.errors == ["paper: missing crop missing.jpg"]


def test_validate_dataset_reports_empty_crop(tmp_path: Path) -> None:
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "source.pdf").write_bytes(b"pdf")
    (paper_dir / "empty.jpg").write_bytes(b"")
    (paper_dir / "gold.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "type": "mcq_answer_key",
                        "label": None,
                        "paper": None,
                        "fragments": ["empty.jpg"],
                    }
                ]
            }
        )
    )
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "papers": [
                    {
                        "id": "paper",
                        "level": "P6",
                        "subject": "Maths",
                        "directory": "paper",
                    }
                ],
            }
        )
    )

    report = validate_dataset(manifest, require_balanced_groups=False)

    assert report.errors == ["paper: empty crop empty.jpg"]
