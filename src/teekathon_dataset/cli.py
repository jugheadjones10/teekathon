from __future__ import annotations

import argparse
from pathlib import Path

from .exporter import build_dataset
from .validator import validate_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the Teekathon dataset")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="copy the selected data out of TREX")
    build.add_argument("--artifacts-root", type=Path, required=True)
    build.add_argument("--selection", type=Path, default=Path("dataset_selection.json"))
    build.add_argument("--repository-root", type=Path, default=Path.cwd())

    validate = commands.add_parser("validate", help="validate the included dataset")
    validate.add_argument(
        "manifest", type=Path, nargs="?", default=Path("dataset.json")
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        dataset = build_dataset(
            artifacts_root=args.artifacts_root,
            selection_path=args.selection,
            repository_root=args.repository_root,
        )
        print(f"Exported {len(dataset['papers'])} papers")
        return 0

    report = validate_dataset(args.manifest)
    print(
        f"{report.paper_count} papers, {report.region_count} regions, "
        f"{report.crop_count} crop images"
    )
    if report.errors:
        for error in report.errors:
            print(f"ERROR: {error}")
        return 1
    print("Dataset is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
