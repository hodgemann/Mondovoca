# Development

[Back to the README](../README.md)

## Checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
node --check src/mondovoca/static/app.js
```

The tests use fake Qwen/AuK/Fish/Robin engines; they never load model weights. With a Python that
has Playwright and an existing Chromium executable:

```bash
python scripts/check_model_ui.py --chromium /path/to/chrome
python scripts/check_mondovoca_ui.py --python /path/to/tiny/python --chromium /path/to/chrome
```

The first uses fixture audio for all Qwen modes, AuK design/cloning, Fish
generation/cloning, Robin design/reference with every Voice LoRA choice, and
portable setup fields for all five engines. The second runs actual Tiny Voca
synthesis using the configured engine paths and isolated history. Separate
live checks exercised all three Qwen modes, both Base cloning options, Robin
voice design with and without the character LoRA, and Robin reference
conditioning. Both engines rendered the same 529-character passage. Chromium
played the saved takes and cancelled a running Robin CUDA job.

Install this package with `pip install --no-deps .` for the `mondovoca` foreground
command. Engine dependencies are separate. Static assets have no build step or
CDN. Model weights, virtual environments, local settings and generated audio are
excluded from the package and Git.

## Versions

The sidebar shows the version the server reports, which comes from
`src/mondovoca/__init__.py`. `pyproject.toml` reads that same value for the package
version, and `static/app.js` and `static/index.html` carry matching fallbacks, so a
release moves all of them together. `/api/health` returns the running version.

Each published release is tagged `v<version>` and listed under
[Releases](https://github.com/hodgemann/Mondovoca/releases); 0.9.2 is the first.
[CHANGELOG.md](../CHANGELOG.md) describes every version, including the ones before
the first release.
