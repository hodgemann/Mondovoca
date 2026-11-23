# Changelog

## 0.9.2 — 2026-11-23

- Replaced the “M∿” sidebar mark with the Mondocraft goose, and serve
  Mondocraft's goose `favicon.ico` so the goose shows in the browser tab.
- Replaced the portrait README screenshot with a 1600 by 900 landscape capture
  at native scale.
- Rewrote the README around working in the studio, with the screenshot under the
  opening paragraph. Engine setup, saved takes, the API and development checks
  moved to `docs/engines.md`, `docs/setup.md`, `docs/api.md` and
  `docs/development.md`.

## 0.9.1 — 2026-11-23

- README pass for the first public release: added Further reading, Versions,
  Author and contributors and Licence sections, and renamed *Run and configure* to
  *Run locally*. Documented the per-engine default installation folders, the
  `--instance` launch flag, the colour theme menu, and the Tiny Voca voice an API
  request gets when it omits `voice` — Male/M2, where the studio selects Female/F2.
  Dropped the references to local-only files that the published repository does not
  contain.
- Reconciled the repository notice with the packaged notice so both record the same
  third-party provenance and the same contributors.
- Verified live CUDA generation in all three Qwen3-TTS modes, including
  transcript-guided and embedding-only Base cloning. Verified Robin voice
  design, the supplied character LoRA, reference conditioning, four-chunk
  rendering and Stop during CUDA inference. Both engines rendered a
  529-character script and produced playable mono WAVs.
- Added a studio screenshot showing three separate Qwen3-TTS test takes.

## 0.9.0 — 2026-11-23

- Offer all 20 Mondocraft colour themes in the voice studio, including the green
  and amber phosphor CRT skins.

## 0.8.1 — 2026-11-21

- The Green and Amber phosphor CRT skins now show their scanline texture and
  phosphor heading glow in the voice studio. `themes.css` stays a verbatim copy
  of Mondocraft's, so `studio.css` maps those two decorations from Mondocraft's
  markup onto the studio's own sidebar and headings. Both switch off under
  `prefers-reduced-motion`, exactly as in Mondocraft.

## 0.8.0 — 2026-11-21

- Add Green phosphor CRT and Amber phosphor CRT to the Colour theme choice. Each
  is a single-hue skin on pure black: green uses #33ff33 text with #66ff66
  accents, amber uses #ffb000 text with #ffc94d accents. Both are imported
  verbatim from Mondocraft v1.125.0, so the two `themes.css` copies stay
  byte-identical and round-trip.
- Like Mondocraft, the retro skins also set a monospace page font. Their scanline
  texture and phosphor heading glow target Mondocraft's own markup, so the voice
  studio shows the palette and the monospace page.

## 0.7.1 — 2026-11-17

- Document a matching Robin PyTorch/torchaudio CUDA runtime after reproducing a
  CUDA initialization segfault in the previously installed environment.
- Report worker termination signals and exit codes instead of a generic synthesis
  failure. Enable native crash traces and preserve worker errors in the server log.

## 0.7.0 — 2026-11-17

- Give Robin full studio controls: voice description, English/Japanese/mixed
  language, reference conditioning with an optional transcript, and every sampling
  control (temperature, top-k, top-p, voice guidance, decoder steps, sampler,
  read-as-written) as native inputs with shared tooltip hints. Choosing Robin hides
  and disables the Tiny Voca, Qwen3-TTS, AuK and Fish controls it does not use.
- Add the Voice LoRA choice: Model setup default, None, the found character
  adapter, or a typed absolute `.safetensors` path, with strength 0–2 as a linked
  number and slider. A saved take restores the adapter it actually used, choosing
  the matching option or a custom path.
- Add Robin Python, model-folder and optional voice LoRA fields to Model setup.
  Saving setup refreshes the available LoRA choices without loading a model or
  disturbing a typed custom path; selecting Robin never starts generation.
- Robin takes now appear in history, presets, recipe export, the connection
  dialog, downloads and the save-location flows, and a completed take can become a
  Robin reference. The takes heading no longer claims a fixed 24 kHz rate.

## 0.6.0 — 2026-11-17

- Add Robin as the fifth catalog model with offline discovery, a dedicated CUDA
  worker, voice-description generation, optional voice LoRA with strength,
  reference conditioning, English/Japanese/mixed language, and Robin's own
  sampling controls: temperature, top-k, top-p, voice guidance, decoder steps,
  sampler, read-as-written and seed.
- Add Robin Python/model-folder setup with portable defaults, environment
  overrides and actionable missing-file reporting. The base checkpoint, VAE, ONNX
  speech tokenizer, config and tokenizer are required; the supplied character
  LoRA stays optional.
- Long scripts keep Robin's own chunking, retakes and continuity with a one-hour
  worker limit, 48 kHz output and the existing exclusive-create and Stop
  behaviour. Studio controls for Robin are a separate next step; live Robin
  generation is not tested yet.
- A Robin take may leave the voice description empty, so a voice LoRA or a
  reference alone can carry the timbre. Each take records the adapter it
  actually used, so later Model setup changes do not rewrite a saved recipe,
  and an existing output file is refused before any weight loading. The studio
  dropdown lists Robin but keeps the choice disabled until its controls arrive.

## 0.5.2 — 2026-11-17

- Create and clear the working spool when switching from a saved output folder
  to browser downloads, so reference uploads and generated takes work immediately.
  Preserve takes saved in permanent folders and their history.

## 0.5.1 — 2026-11-17

- Show completed takes discovered in an existing output folder immediately after
  selecting it, without requiring a reload or downloading restored takes again.

## 0.5.0 — 2026-11-17

- Add a user-facing save-location setting with two modes: browser download (the
  default, which downloads each finished take and keeps only a working spool that
  is cleared when the studio stops) or a chosen writable folder on this computer.
  Each take is kept once, never in both places.
- Show the current location, offer the system folder chooser where the browser
  supports it, validate absolute writable folders, refuse changes during
  generation, and keep takes saved earlier listed, playable and downloadable.

## 0.4.0 — 2026-11-17

- Add Fish Audio Drama 3 INT8 with RAM-backed experts, a 16 GiB VRAM cache,
  supplied-transcript cloning, inline directions, sampling and saved recipes.
- Add portable shared runtime locations, offline workers, cancellation, and
  missing-file diagnostics for the separate model and audio codec.

## 0.3.0 — 2026-11-16

- Add AuK Base as the third model with voice descriptions, reference cloning,
  duration and diffusion controls, CPU offload, recipes, history and speech API.
- Add portable AuK Python/checkpoint/encoder setup and missing-file diagnostics.
  Keep model loading in cancellable offline workers and shared engines separate.
- Document manual AuK/VAE copies, the official Omni 3B encoder and its license.
- Verify the native API with fake modules and all three models in browser fixtures.

## 0.2.2 — 2026-11-17

- Select Female/F2 by default.
- Start the script empty with suggested text as a placeholder; keep examples
  available through Try an example.
- Start generated and restored takes at 40% playback volume, preserving WAV levels.

## 0.2.1 — 2026-11-16

- Default Qwen3-TTS to English in the studio and API; retain explicit Auto and
  other language choices. Publish the default language in the model catalog.

## 0.2.0 — 2026-11-16

- Add an audio model dropdown with Tiny Voca and Qwen3-TTS 1.7B.
- Expose Qwen preset voices, voice design, transcript/embedding voice cloning,
  ten languages, native instructions and generation controls.
- Reuse completed takes as cloning references; preserve model-specific recipes.
- Add portable local setup for both engines and separate Python environments;
  keep installation paths out of source and model loading out of discovery.
- Verify Qwen integration with fake engines and browser fixture audio. Live
  Qwen generation was verified in 0.9.1.

## 0.1.0 — 2026-11-16

- Add a standalone local Mondovoca voice studio with Mondocraft themes and tooltips.
- Add F2/M2 gender selection, heuristic age styling from 5–100, five-year choices,
  delivery presets, speed/pitch/intonation controls and reproducible seeds.
- Add saved voices, text import, two-voice comparisons, waveform previews,
  playback, WAV downloads, JSON recipes, restored settings and persistent takes.
- Add CPU worker cancellation, input validation, loopback HTTP boundaries,
  Open WebUI speech compatibility and owned start/stop scripts.
- Keep tiny-voca a separate installation, independent of the studio.
