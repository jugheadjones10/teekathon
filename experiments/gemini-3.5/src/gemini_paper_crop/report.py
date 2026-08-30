import html
import json
from pathlib import Path

from gemini_paper_crop.pipeline import PaperRunResult


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_paper_report(result: PaperRunResult) -> Path:
    page_sections: list[str] = []
    for page in result.pages:
        page_src = html.escape(_relative(page.page_path, result.paper_dir))
        if page.error:
            detail = f'<p class="error">{html.escape(page.error)}</p>'
            overlay = ""
            crops = ""
        else:
            detail = (
                f"<p>{len(page.crops)} regions · "
                f"{page.elapsed_seconds:.2f}s"
                f"{' · resumed' if page.resumed else ''}</p>"
            )
            overlay_src = html.escape(
                _relative(page.overlay_path, result.paper_dir)  # type: ignore[arg-type]
            )
            overlay = (
                "<figure><figcaption>Gemini overlay</figcaption>"
                f'<a href="{overlay_src}"><img loading="lazy" src="{overlay_src}"></a>'
                "</figure>"
            )
            crop_cards: list[str] = []
            assert page.detections is not None
            for crop in page.crops:
                region = page.detections.regions[crop.region_index - 1]
                crop_src = html.escape(_relative(crop.path, result.paper_dir))
                question = (
                    f"Question {html.escape(region.question_number)}"
                    if region.question_number
                    else "Unlabelled/table"
                )
                crop_cards.append(
                    '<figure class="crop">'
                    f"<figcaption>{html.escape(region.type)} · {question}</figcaption>"
                    f'<a href="{crop_src}"><img loading="lazy" src="{crop_src}"></a>'
                    "</figure>"
                )
            crops = '<div class="crops">' + "".join(crop_cards) + "</div>"

        usage = (
            html.escape(json.dumps(page.usage, sort_keys=True)) if page.usage else "n/a"
        )
        page_sections.append(
            f'<section id="page-{page.page_number}">'
            f"<h2>Page {page.page_number}</h2>{detail}"
            '<div class="pages">'
            "<figure><figcaption>Rendered page</figcaption>"
            f'<a href="{page_src}"><img loading="lazy" src="{page_src}"></a>'
            f"</figure>{overlay}</div>{crops}"
            f"<details><summary>Usage metadata</summary><pre>{usage}</pre></details>"
            "</section>"
        )

    report = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(result.pdf_name)} – Gemini crops</title>
<style>
body {{ font: 16px/1.45 system-ui, sans-serif; margin: 0 auto;
  max-width: 1500px; padding: 24px; color: #17212b; }}
h1 {{ margin-bottom: 4px; }}
section {{ border-top: 1px solid #ccd2d8; margin-top: 32px;
  padding-top: 16px; }}
.pages {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }}
.crops {{ display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px; margin-top: 20px; }}
figure {{ margin: 0; }} figcaption {{ font-weight: 650; margin-bottom: 6px; }}
img {{ border: 1px solid #ccd2d8; height: auto; max-width: 100%; }}
.crop img {{ max-height: 520px; object-fit: contain; }}
.error {{ color: #b42318; font-weight: 650; }}
pre {{ overflow-wrap: anywhere; white-space: pre-wrap; }}
@media (max-width: 760px) {{ .pages {{ grid-template-columns: 1fr; }} }}
</style>
<h1>{html.escape(result.pdf_name)}</h1>
<p>Model: <code>{html.escape(result.model)}</code> ·
  {len(result.pages)} processed pages</p>
{"".join(page_sections)}
</html>
"""
    report_path = result.paper_dir / "report.html"
    report_path.write_text(report)
    return report_path


def write_run_index(run_dir: Path, results: list[PaperRunResult]) -> Path:
    cards: list[str] = []
    for result in results:
        relative_report = (result.paper_dir / "report.html").relative_to(run_dir)
        region_count = sum(len(page.crops) for page in result.pages)
        error_count = sum(page.error is not None for page in result.pages)
        cards.append(
            "<li>"
            f'<a href="{html.escape(relative_report.as_posix())}">'
            f"{html.escape(result.pdf_name)}</a>"
            f"<span>{region_count} detected region"
            f"{'s' if region_count != 1 else ''} · {error_count} page errors</span>"
            "</li>"
        )

    index = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemini paper crop run</title>
<style>
body {{ font: 17px/1.5 system-ui, sans-serif; margin: 0 auto;
  max-width: 920px; padding: 32px; color: #17212b; }}
ul {{ list-style: none; padding: 0; }}
li {{ border: 1px solid #ccd2d8; border-radius: 10px; margin: 12px 0;
  padding: 18px; }}
a {{ display: block; font-size: 1.05rem; font-weight: 700; }}
span {{ color: #52606d; }}
</style>
<h1>Gemini paper crop run</h1>
<p>Open a paper to inspect its rendered pages, overlays, and crops.</p>
<ul>{"".join(cards)}</ul>
</html>
"""
    index_path = run_dir / "index.html"
    index_path.write_text(index)
    return index_path
