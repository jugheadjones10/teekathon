from pathlib import Path

import pymupdf
from PIL import Image

from gemini_paper_crop.render import render_pdf


def test_render_pdf_creates_numbered_png_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    document.new_page(width=72, height=144)
    document.new_page(width=144, height=72)
    document.save(pdf_path)
    document.close()

    rendered = render_pdf(pdf_path, tmp_path / "pages", dpi=72)

    assert [page.page_number for page in rendered] == [1, 2]
    assert [page.path.name for page in rendered] == ["page-001.png", "page-002.png"]
    with Image.open(rendered[0].path) as image:
        assert image.size == (72, 144)


def test_render_pdf_honours_page_limit(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    document.new_page()
    document.new_page()
    document.save(pdf_path)
    document.close()

    rendered = render_pdf(pdf_path, tmp_path / "pages", dpi=72, page_limit=1)

    assert len(rendered) == 1
