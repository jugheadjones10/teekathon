from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

EXPECTED_GROUPS = {
    ("P5", "Maths"): 20,
    ("P5", "Science"): 20,
    ("P6", "Maths"): 20,
    ("P6", "Science"): 20,
}
VALID_REGION_TYPES = {
    "mcq_question",
    "oe_question",
    "mcq_answer_key",
    "oe_answer_key",
}


@dataclass(frozen=True)
class ValidationReport:
    paper_count: int
    region_count: int
    crop_count: int
    groups: dict[tuple[str, str], int]
    errors: list[str]


def validate_dataset(
    manifest_path: Path, *, require_balanced_groups: bool = True
) -> ValidationReport:
    root = manifest_path.parent
    dataset = json.loads(manifest_path.read_text())
    papers = dataset.get("papers", [])
    groups = Counter((paper.get("level"), paper.get("subject")) for paper in papers)
    errors: list[str] = []
    region_count = 0
    crop_count = 0

    if dataset.get("schema_version") != 1:
        errors.append("dataset.json must use schema_version 1")

    if require_balanced_groups:
        for group, expected in EXPECTED_GROUPS.items():
            actual = groups.get(group, 0)
            if actual != expected:
                errors.append(
                    f"{group[0]} {group[1]}: expected {expected} papers, found {actual}"
                )

    for paper in papers:
        paper_id = str(paper.get("id", "<unknown>"))
        paper_dir = root / str(paper.get("directory", ""))
        source_pdf = paper_dir / str(paper.get("source_pdf", "source.pdf"))
        gold_path = paper_dir / str(paper.get("gold_manifest", "gold.json"))
        if not source_pdf.is_file():
            errors.append(f"{paper_id}: missing source PDF {source_pdf.name}")
        if not gold_path.is_file():
            errors.append(f"{paper_id}: missing gold manifest {gold_path.name}")
            continue
        gold = json.loads(gold_path.read_text())
        for region in gold.get("regions", []):
            region_count += 1
            region_type = region.get("type")
            if region_type not in VALID_REGION_TYPES:
                errors.append(f"{paper_id}: invalid region type {region_type!r}")
            for fragment in region.get("fragments", []):
                crop_count += 1
                crop_path = paper_dir / fragment
                if not crop_path.is_file():
                    errors.append(f"{paper_id}: missing crop {fragment}")
                elif crop_path.stat().st_size == 0:
                    errors.append(f"{paper_id}: empty crop {fragment}")

    return ValidationReport(
        paper_count=len(papers),
        region_count=region_count,
        crop_count=crop_count,
        groups=dict(groups),
        errors=errors,
    )
