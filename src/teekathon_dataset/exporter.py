from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

TYPE_ORDER = {
    "mcq_question": 0,
    "oe_question": 1,
    "mcq_answer_key": 2,
    "oe_answer_key": 3,
}
DATASET_GROUPS = {
    ("P5", "Maths"),
    ("P5", "Science"),
    ("P6", "Maths"),
    ("P6", "Science"),
}


@dataclass(frozen=True)
class Crop:
    source: Path
    relative_source: str
    region_type: str
    label: str | None
    paper: int | None


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text())


def _paper_number(path: PurePosixPath) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"paper(\d+)", part, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _is_deleted(relative_path: str, deleted: set[str]) -> bool:
    path = PurePosixPath(relative_path)
    return any(parent.as_posix() in deleted for parent in (path, *path.parents))


def _natural_key(value: str) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def _safe_label(label: str | None) -> str:
    if label is None or not label.strip():
        return "unlabelled"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip())


def _materialize_file(source: Path, target: Path, *, space_efficient: bool) -> None:
    if space_efficient:
        try:
            clone = subprocess.run(
                ["cp", "-c", str(source), str(target)],
                capture_output=True,
                check=False,
            )
            if clone.returncode == 0:
                return
        except OSError:
            pass
        target.unlink(missing_ok=True)
    shutil.copy2(source, target)


def _manifest_label(
    relative_path: str,
    fallback: str | None,
    label_edits: dict[str, str],
    created_items: dict[str, dict[str, Any]],
) -> str | None:
    path = PurePosixPath(relative_path)
    candidates = (path.as_posix(), path.parent.as_posix())
    for candidate in candidates:
        if candidate in label_edits:
            return str(label_edits[candidate])
    created = created_items.get(path.as_posix())
    if created and created.get("label") is not None:
        return str(created["label"])
    return fallback


def _mcq_fallback_label(path: Path, labels: dict[str, Any]) -> str | None:
    if path.name in labels:
        return str(labels[path.name])
    match = re.match(r"(\d+)", path.stem)
    return match.group(1) if match else None


def _discover_crops(trex_paper_dir: Path) -> list[Crop]:
    manifest = _load_json(trex_paper_dir / "manifest.json", {})
    deleted = {str(path) for path in manifest.get("deleted_items", [])}
    label_edits = {
        str(path): str(label) for path, label in manifest.get("label_edits", {}).items()
    }
    created_items = {
        str(item["path"]): item
        for item in manifest.get("created_items", [])
        if item.get("path")
    }
    mcq_labels = _load_json(trex_paper_dir / "mcq" / "labels.json", {})
    crops: list[Crop] = []

    def add(
        path: Path,
        region_type: str,
        fallback_label: str | None,
        paper: int | None,
    ) -> None:
        relative = path.relative_to(trex_paper_dir).as_posix()
        if _is_deleted(relative, deleted):
            return
        label = _manifest_label(relative, fallback_label, label_edits, created_items)
        crops.append(
            Crop(
                source=path,
                relative_source=relative,
                region_type=region_type,
                label=label,
                paper=paper,
            )
        )

    for path in sorted((trex_paper_dir / "mcq").glob("*.jpg")):
        add(path, "mcq_question", _mcq_fallback_label(path, mcq_labels), 1)

    for path in sorted((trex_paper_dir / "oe").glob("paper*/**/question*.jpg")):
        relative = PurePosixPath(path.relative_to(trex_paper_dir).as_posix())
        add(path, "oe_question", relative.parent.name, _paper_number(relative))

    for path in sorted((trex_paper_dir / "ak" / "mcq").glob("*.jpg")):
        relative = path.relative_to(trex_paper_dir).as_posix()
        add(
            path,
            "mcq_answer_key",
            _manifest_label(relative, None, label_edits, created_items),
            None,
        )

    for path in sorted((trex_paper_dir / "ak").glob("paper*/**/answer*.jpg")):
        relative = PurePosixPath(path.relative_to(trex_paper_dir).as_posix())
        add(path, "oe_answer_key", relative.parent.name, _paper_number(relative))

    return crops


def _group_crops(crops: Iterable[Crop]) -> list[list[Crop]]:
    grouped: dict[tuple[str, int | None, str | None], list[Crop]] = defaultdict(list)
    for crop in crops:
        grouped[(crop.region_type, crop.paper, crop.label)].append(crop)

    def group_key(item: tuple[tuple[str, int | None, str | None], list[Crop]]):
        (region_type, paper, label), _ = item
        return (
            TYPE_ORDER[region_type],
            paper or 0,
            _natural_key(label or ""),
        )

    return [
        sorted(group, key=lambda crop: _natural_key(crop.relative_source))
        for _, group in sorted(grouped.items(), key=group_key)
    ]


def export_paper(
    *,
    source_pdf: Path,
    trex_paper_dir: Path,
    destination: Path,
    level: str,
    subject: str,
    space_efficient: bool = False,
) -> dict[str, Any]:
    """Copy one source PDF and its corrected TREX crops into a stable layout."""
    destination.mkdir(parents=True, exist_ok=True)
    _materialize_file(
        source_pdf,
        destination / "source.pdf",
        space_efficient=space_efficient,
    )

    manifest_path = trex_paper_dir / "manifest.json"
    if manifest_path.is_file():
        _materialize_file(
            manifest_path,
            destination / "trex_manifest.json",
            space_efficient=space_efficient,
        )

    regions: list[dict[str, Any]] = []
    for group in _group_crops(_discover_crops(trex_paper_dir)):
        first = group[0]
        paper_folder = f"paper{first.paper}" if first.paper is not None else "all"
        target_dir = (
            destination
            / "gold"
            / first.region_type
            / paper_folder
            / _safe_label(first.label)
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        fragments: list[str] = []
        source_paths: list[str] = []
        for index, crop in enumerate(group, start=1):
            target = target_dir / f"{index:02d}{crop.source.suffix.lower()}"
            _materialize_file(crop.source, target, space_efficient=space_efficient)
            fragments.append(target.relative_to(destination).as_posix())
            source_paths.append(crop.relative_source)
        regions.append(
            {
                "type": first.region_type,
                "label": first.label,
                "paper": first.paper,
                "fragments": fragments,
                "trex_source_paths": source_paths,
            }
        )

    paper = {
        "schema_version": 1,
        "id": source_pdf.stem,
        "filename": source_pdf.name,
        "level": level,
        "subject": subject,
        "source_pdf": "source.pdf",
        "regions": regions,
    }
    (destination / "gold.json").write_text(json.dumps(paper, indent=2) + "\n")
    return paper


def build_dataset(
    *,
    artifacts_root: Path,
    selection_path: Path,
    repository_root: Path,
    space_efficient: bool = True,
) -> dict[str, Any]:
    """Export every reviewed paper listed in the checked-in selection file."""
    selection = json.loads(selection_path.read_text())
    repository_root.mkdir(parents=True, exist_ok=True)
    paper_entries: list[dict[str, Any]] = []
    checkmarks: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    for selected in selection.get("papers", []):
        level = str(selected["level"])
        subject = str(selected["subject"])
        filename = str(selected["filename"])
        if (level, subject) not in DATASET_GROUPS:
            raise ValueError(f"unsupported dataset group: {level} {subject}")
        if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
            raise ValueError(f"unsafe paper filename: {filename}")
        if level not in checkmarks:
            output_root = artifacts_root / "output" / level
            checkmarks[level] = (
                _load_json(output_root / "paper_checkmarks.json", {}),
                _load_json(output_root / "paper_ak_checkmarks.json", {}),
            )
        paper_marks, answer_marks = checkmarks[level]
        paper_reviewed = paper_marks.get(filename) is True
        answers_reviewed = answer_marks.get(filename) is True
        if not paper_reviewed or not answers_reviewed:
            raise ValueError(f"{filename} has not passed both TREX review checks")

        source_pdf = artifacts_root / "input" / level / filename
        trex_paper_dir = artifacts_root / "output" / level / filename
        if not source_pdf.is_file():
            raise FileNotFoundError(source_pdf)
        if not trex_paper_dir.is_dir():
            raise FileNotFoundError(trex_paper_dir)

        relative_dir = Path("data") / level / subject / source_pdf.stem
        destination = repository_root / relative_dir
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing dataset directory: {destination}"
            )
        paper = export_paper(
            source_pdf=source_pdf,
            trex_paper_dir=trex_paper_dir,
            destination=destination,
            level=level,
            subject=subject,
            space_efficient=space_efficient,
        )
        paper_entries.append(
            {
                "id": paper["id"],
                "level": level,
                "subject": subject,
                "directory": relative_dir.as_posix(),
                "source_pdf": "source.pdf",
                "gold_manifest": "gold.json",
            }
        )

    dataset = {"schema_version": 1, "papers": paper_entries}
    (repository_root / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    return dataset
