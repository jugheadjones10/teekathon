# Optional Gemini 3.5 cropping experiment

This is an earlier zero-shot baseline that asks Gemini 3.5 Flash to find exam
questions and answer-key regions. It does **not** fine-tune a model and is not
part of the required Tinker submission.

It may still be useful to candidates as:

- a reference for rendering PDF pages and parsing bounding-box JSON;
- a quick baseline to compare against a fine-tuned model;
- an HTML visualizer for inspecting predicted boxes and crops; or
- a starting point for matching the supplied gold crops back to PDF pages.

## Contents

```text
papers/   three small PDF fixtures
prompts/  the Gemini detection prompt
src/      detector, cropper, renderer, and HTML report code
tests/    tests for the experiment
example-runs/  two saved example reports that can be viewed without an API key
runs/          ignored output from any new runs
```

To inspect the existing results, open either `example-runs/smoke-3-pages/index.html`
or `example-runs/one-paper/index.html` in a browser.

## Run it

From this directory:

```bash
uv sync
cp .env.example .env
# Add GEMINI_API_KEY to .env

uv run crop-papers \
  --paper Science-P6-2024-CA1-Anglo_Chinese-3149.pdf \
  --page-limit 3 \
  --run-name smoke-3-pages
```

Then open `runs/smoke-3-pages/index.html` in a browser. To process every page of
all three fixture papers, run:

```bash
uv run crop-papers --run-name all-three-papers
```

Use `--resume` with the same run name to reuse successful page responses after
an interrupted run.

The generated report shows each rendered page, Gemini's bounding boxes, and the
resulting crops. New run outputs and API keys are deliberately excluded from
Git; the two small `example-runs` are retained as reference outputs.
