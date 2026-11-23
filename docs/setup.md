# Setup and saved takes

[Back to the README](../README.md)

## Starting and stopping

```bash
./start.sh
# Open http://127.0.0.1:8093
./stop.sh
```

The studio server requires Python 3.10+ and its standard library. It starts even
without any models installed. `start.sh` uses this checkout's `.venv/bin/python`
when present, otherwise `python3`. Set `MONDOVOCA_PYTHON` to choose another server
Python. Start is idempotent; status is `python3 scripts/mondovoca_service.py status`.
The scripts manage only their own server. Set `MONDOVOCA_PORT` for another port.

Open **Model setup…** below the audio model dropdown and enter your existing
engine locations. Save, then select an audio model. Each engine can have its own
Python environment; its Python executable must have the engine and dependencies
installed. Saving paths and browsing models never loads weights or starts GPU
work. File availability is an installation check, not a runtime health test.
Missing locations can be saved before installing an engine.

Model settings are kept in `.mondovoca/models.json` in the checkout, which Git
ignores. Until you save
locations, each engine defaults to the server Python and a folder under
`~/.local/share/mondovoca/models/`: `tiny-voca/` for the checkpoint and phonemizer,
`qwen3-tts/`, `auk/`, `auk-encoder/`, `fish-drama3/`, `fish-runtime/`,
`fish-comfy/` and `robin/`. Environment overrides take precedence and appear
read-only in the setup dialog.

| Variable | Purpose |
| --- | --- |
| `MONDOVOCA_PYTHON` | Python used to run the lightweight server |
| `MONDOVOCA_MODEL_CONFIG` | Model settings JSON path |
| `MONDOVOCA_TINY_PYTHON` | Python with tiny_voca, dizione, torch, numpy, soundfile |
| `MONDOVOCA_CHECKPOINT` | Tiny Voca checkpoint |
| `DIZIONE_DATA` | Tiny Voca phonemizer data folder |
| `MONDOVOCA_QWEN_PYTHON` | Python with qwen_tts and its compatible dependencies |
| `MONDOVOCA_QWEN_ROOT` | Parent folder of the three Qwen checkpoints |
| `MONDOVOCA_AUK_PYTHON` | Separate Python with the installed AuK package |
| `MONDOVOCA_AUK_ROOT` | Folder containing AuK config, base checkpoint and VAE |
| `MONDOVOCA_AUK_ENCODER` | Complete local Qwen2.5-Omni-3B encoder snapshot |
| `MONDOVOCA_FISH_PYTHON` | Isolated Python for the compatible Drama 3 INT8 runtime |
| `MONDOVOCA_FISH_ROOT` | Folder with the converted INT8 checkpoint and codec |
| `MONDOVOCA_FISH_RUNTIME` | Separate PedroMarinhoDev/ComfyUI-Drama3 source checkout |
| `MONDOVOCA_FISH_COMFY` | Separate compatible ComfyUI library checkout |
| `MONDOVOCA_ROBIN_PYTHON` | Isolated Python with the installed Robin package |
| `MONDOVOCA_ROBIN_ROOT` | Robin model folder with the checkpoint, VAE, ONNX tokenizer and config |
| `MONDOVOCA_ROBIN_LORA` | Optional Robin voice LoRA used when a request does not choose one |
| `MONDOVOCA_PORT` | Loopback port, default 8093 |
| `MONDOVOCA_DATA_DIR` | Fixes folder mode to this location; read-only in the studio |
| `MONDOVOCA_THREADS` | Tiny Voca CPU threads, default 4 |

The foreground entry point also accepts `--checkpoint`, `--phonemizer`,
`--data-dir`, `--model-config`, `--port` and `--instance`; `--instance` is how
`scripts/mondovoca_service.py` recognises the server it started, so it can manage
that one process and no other. `--data-dir` selects folder mode for
that server. Explicit saved model settings or
environment overrides supply the effective Tiny Voca paths.

## Takes, presets and cancellation

Type or import up to 5,000 characters, generate speech, replay/download WAVs,
export JSON recipes, or restore a take's text and settings. Saved voice presets
and colour themes use browser storage; scripts are not stored there. The **Colour
theme** menu offers all 20 Mondocraft palettes, including the green and amber
phosphor CRT skins, and this browser remembers the choice. Completed
scripts and settings are saved with their audio on disk. References in saved
recipes are local IDs; transferring a recipe also requires its reference audio.

WAVs and JSON metadata live in `.mondovoca/clips/` by default; uploaded references
are in its `references/` subfolder. Random UUID filenames and exclusive creation
preserve existing clips. The latest 100 completed takes are restored on startup.

**Where takes are saved.** Each take is kept once, never twice. Open **Save
location…** under *Saved takes* in the sidebar. Browser-download mode is the
default: every finished take downloads automatically to your browser's download
folder, following that browser's own download folder and save-dialog settings.
The studio keeps one working copy in `.mondovoca/spool/` while it runs, so takes
stay playable and reusable as clone references, and clears that spool when the
studio stops or when the setting changes. Folder mode instead saves each take
permanently in a folder on this computer and does not download it. **Choose
folder…** opens your system's folder dialog; browsers reveal only the chosen
folder's name, so Mondovoca looks for it beside the current folder or inside your
home folder. Enter the full path for any other location. A chosen folder must be
absolute and writable, and is created when missing.

Takes saved earlier stay in the folder that saved them and remain listed, playable
and downloadable; the studio remembers the last eight folders. Changing the save
location while a take is generating is refused. Uploaded references live in the
same folder, so in browser-download mode they are cleared with the spool. API
clients are unaffected by the choice: `/v1/audio/speech` always returns the WAV
bytes to the caller. `--data-dir` or
`MONDOVOCA_DATA_DIR` fixes folder mode for that server and appears read-only in
the dialog.

The Stop button cancels the owned worker and pauses playback. Only one generation
runs at a time. Worker time limits are three minutes for Tiny Voca, fifteen
minutes for Qwen, AuK and Fish, and one hour for Robin, whose long scripts render
chunk by chunk; every limit includes model loading. Each take releases its model when the
worker exits. Ctrl/Cmd+Enter generates speech. Browser autoplay restrictions may
require pressing play.
