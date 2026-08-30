from pathlib import Path

import pytest

from gemini_paper_crop.cli import paper_output_slug, prepare_run_dir, resolve_papers


def test_resolve_papers_uses_all_sorted_pdfs_by_default(tmp_path: Path) -> None:
    (tmp_path / "b.pdf").touch()
    (tmp_path / "a.pdf").touch()
    (tmp_path / "notes.txt").touch()

    papers = resolve_papers([], tmp_path)

    assert [paper.name for paper in papers] == ["a.pdf", "b.pdf"]


def test_resolve_papers_accepts_a_copied_paper_basename(tmp_path: Path) -> None:
    expected = tmp_path / "paper.pdf"
    expected.touch()

    papers = resolve_papers([Path("paper.pdf")], tmp_path)

    assert papers == [expected]


def test_prepare_run_dir_refuses_to_overwrite_without_resume(tmp_path: Path) -> None:
    existing = tmp_path / "experiment"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        prepare_run_dir(tmp_path, "experiment", resume=False)


def test_prepare_run_dir_reuses_existing_directory_for_resume(tmp_path: Path) -> None:
    existing = tmp_path / "experiment"
    existing.mkdir()

    assert prepare_run_dir(tmp_path, "experiment", resume=True) == existing


def test_paper_output_slug_removes_path_control_characters() -> None:
    assert paper_output_slug(Path("Science P6 (A).pdf")) == "Science-P6-A"
    assert paper_output_slug(Path("...pdf")) == "paper"
