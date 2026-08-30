import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from gemini_paper_crop.detector import GeminiDetector
from gemini_paper_crop.pipeline import PaperRunResult, run_paper
from gemini_paper_crop.report import write_paper_report, write_run_index


def resolve_papers(requested: list[Path], papers_dir: Path) -> list[Path]:
    candidates = requested or sorted(papers_dir.glob("*.pdf"))
    resolved: list[Path] = []
    for candidate in candidates:
        path = candidate if candidate.exists() else papers_dir / candidate
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"PDF not found: {candidate}")
        resolved.append(path)
    if not resolved:
        raise FileNotFoundError(f"No PDFs found in {papers_dir}")
    return resolved


def prepare_run_dir(output_root: Path, run_name: str, *, resume: bool) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError(
            "run name may only contain letters, numbers, dot, dash, underscore"
        )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=resume)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask Gemini 3.5 Flash to crop questions and answer keys."
    )
    parser.add_argument(
        "--paper",
        action="append",
        type=Path,
        default=[],
        help="PDF path or basename under papers/. Repeat to select multiple papers.",
    )
    parser.add_argument("--papers-dir", type=Path, default=Path("papers"))
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--prompt", type=Path, default=Path("prompts/crop-v1.txt"))
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument(
        "--thinking-level",
        choices=["minimal", "low", "medium", "high"],
        default="medium",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument(
        "--page-limit",
        type=int,
        help="Process only the first N pages of each selected paper.",
    )
    parser.add_argument(
        "--run-name",
        default=datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse saved detections in an existing named run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is missing. Add it to .env or export it.")

    try:
        papers = resolve_papers(args.paper, args.papers_dir)
        run_dir = prepare_run_dir(args.output_root, args.run_name, resume=args.resume)
        prompt = args.prompt.read_text()
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    detector = GeminiDetector(
        model=args.model,
        thinking_level=args.thinking_level,
    )
    results: list[PaperRunResult] = []
    for index, pdf_path in enumerate(papers, start=1):
        print(f"[{index}/{len(papers)}] {pdf_path.name}", flush=True)
        paper_dir = run_dir / pdf_path.stem
        result = run_paper(
            pdf_path,
            paper_dir,
            detector=detector,
            prompt=prompt,
            dpi=args.dpi,
            padding=args.padding,
            page_limit=args.page_limit,
            resume=args.resume,
            progress=lambda message: print(message, flush=True),
        )
        write_paper_report(result)
        results.append(result)

    index_path = write_run_index(run_dir, results)
    metadata = {
        "model": args.model,
        "thinking_level": args.thinking_level,
        "dpi": args.dpi,
        "padding": args.padding,
        "page_limit": args.page_limit,
        "prompt": str(args.prompt),
        "papers": [paper.name for paper in papers],
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nReport: {index_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
