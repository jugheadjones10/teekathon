from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    path: Path
    width: int
    height: int


def render_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 200,
    page_limit: int | None = None,
) -> list[RenderedPage]:
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if page_limit is not None and page_limit <= 0:
        raise ValueError("page_limit must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[RenderedPage] = []
    with pymupdf.open(pdf_path) as document:
        page_count = document.page_count
        if page_limit is not None:
            page_count = min(page_count, page_limit)

        for page_index in range(page_count):
            page = document[page_index]
            pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            path = output_dir / f"page-{page_index + 1:03d}.png"
            pixmap.save(path)
            rendered.append(
                RenderedPage(
                    page_number=page_index + 1,
                    path=path,
                    width=pixmap.width,
                    height=pixmap.height,
                )
            )
    return rendered
