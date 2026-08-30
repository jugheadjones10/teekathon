import json
from pathlib import Path

import pytest

from teekathon_dataset.exporter import build_dataset, export_paper


def _write_image(path: Path, contents: bytes = b"image") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)


def test_export_paper_applies_trex_corrections(tmp_path: Path) -> None:
    source_pdf = tmp_path / "input" / "P6" / "Science-P6-demo.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"pdf")

    trex = tmp_path / "output" / "P6" / source_pdf.name
    _write_image(trex / "mcq" / "1_1.jpg", b"mcq-1")
    _write_image(trex / "mcq" / "2_1.jpg", b"deleted")
    _write_image(trex / "oe" / "paper2" / "7" / "question.jpg", b"oe-7")
    _write_image(trex / "oe" / "paper2" / "7" / "question.1.jpg", b"oe-7b")
    _write_image(trex / "ak" / "paper2" / "7" / "answer.jpg", b"answer-7")
    _write_image(trex / "ak" / "mcq" / "created_1.jpg", b"mcq-ak")
    (trex / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "label_edits": {"mcq/1_1.jpg": "3"},
                "created_items": [
                    {
                        "path": "oe/paper2/7/question.1.jpg",
                        "label": "7",
                        "type": "oe_question",
                    },
                    {
                        "path": "ak/mcq/created_1.jpg",
                        "label": "1",
                        "type": "ak_mcq",
                    },
                ],
                "deleted_items": ["mcq/2_1.jpg"],
            }
        )
    )

    destination = tmp_path / "dataset" / "P6" / "Science" / "demo"
    paper = export_paper(
        source_pdf=source_pdf,
        trex_paper_dir=trex,
        destination=destination,
        level="P6",
        subject="Science",
    )

    assert paper["source_pdf"] == "source.pdf"
    assert [region["type"] for region in paper["regions"]] == [
        "mcq_question",
        "oe_question",
        "mcq_answer_key",
        "oe_answer_key",
    ]
    assert paper["regions"][0]["label"] == "3"
    assert paper["regions"][1]["paper"] == 2
    assert len(paper["regions"][1]["fragments"]) == 2
    assert paper["regions"][2]["label"] == "1"
    assert not any("2_1" in str(path) for path in destination.rglob("*"))
    assert (destination / "trex_manifest.json").is_file()


def test_export_paper_uses_mcq_labels_file(tmp_path: Path) -> None:
    source_pdf = tmp_path / "paper.pdf"
    source_pdf.write_bytes(b"pdf")
    trex = tmp_path / "gold"
    _write_image(trex / "mcq" / "strange-name.jpg")
    (trex / "mcq" / "labels.json").write_text(json.dumps({"strange-name.jpg": "12"}))

    paper = export_paper(
        source_pdf=source_pdf,
        trex_paper_dir=trex,
        destination=tmp_path / "export",
        level="P5",
        subject="Maths",
    )

    assert paper["regions"][0]["label"] == "12"


def test_build_dataset_requires_reviewed_papers(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    filename = "Maths-P5-demo.pdf"
    source = artifacts / "input" / "P5" / filename
    source.parent.mkdir(parents=True)
    source.write_bytes(b"pdf")
    output = artifacts / "output" / "P5"
    _write_image(output / filename / "mcq" / "1.jpg")
    (output / "paper_checkmarks.json").write_text(json.dumps({filename: True}))
    (output / "paper_ak_checkmarks.json").write_text(json.dumps({filename: True}))
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {"papers": [{"level": "P5", "subject": "Maths", "filename": filename}]}
        )
    )

    manifest = build_dataset(
        artifacts_root=artifacts,
        selection_path=selection,
        repository_root=tmp_path / "handoff",
    )

    assert manifest["papers"][0]["directory"].startswith("data/P5/Maths/")
    assert (tmp_path / "handoff" / "dataset.json").is_file()


def test_build_dataset_rejects_unsafe_selection_paths(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "papers": [
                    {
                        "level": "P5",
                        "subject": "../outside",
                        "filename": "paper.pdf",
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="unsupported dataset group"):
        build_dataset(
            artifacts_root=tmp_path,
            selection_path=selection,
            repository_root=tmp_path / "handoff",
        )
