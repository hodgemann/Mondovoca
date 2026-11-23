# Speech engines

[Back to the README](../README.md)

Installation, file layouts, controls, API parameters and licences for each engine Mondovoca can run. Every engine is installed once outside this repository, in its own Python environment, and can be shared with other projects. Enter its locations in **Model setup…**.

## Tiny Voca

Install [Tiny Voca](https://github.com/n4ze3m/tiny-voca) and its dependencies
separately, following its installation instructions. Point Model setup to that
Python, `tiny-voca.pt`, and the matching phonemizer folder. The engine and its
original CLI/demo work independently of Mondovoca.

Tiny Voca uses CPU and produces English, 24 kHz mono WAV with two fixed voices,
Female/F2 and Male/M2. The studio selects Female/F2; an API request that omits
`voice` gets Male/M2, so name `"voice": "F2"` to match the studio default. The UI
provides age styling from 5–100, speed, pitch,
pitch movement, delivery starting styles, and both-voice comparison. Age styling
is a heuristic using the two fixed voices; it is not an age-conditioned model.
Explicit speed, semitones and pitch movement override the age preset. Pitch maps
to log F0 as `semitones * ln(2)/12`. Extreme pitch can sound artificial. This model
does not support voice cloning or directed accent/breathiness.

## Qwen3-TTS

Follow the [official Qwen3-TTS setup](https://github.com/QwenLM/Qwen3-TTS) in a
separate Python environment. This integration uses the native `qwen_tts` API,
PyTorch, numpy and soundfile; the installed API was checked with transformers
4.57.3. Keep that environment independent of engines requiring other versions.
The 1.7B worker uses CUDA, bfloat16 and PyTorch SDPA. A compatible GPU and enough
free VRAM/system RAM are required; there is no automatic CPU fallback.

Choose its Python executable and a model parent folder containing the complete
local checkpoint directories (including each checkpoint's speech tokenizer):

```text
qwen3-tts/
  Qwen3-TTS-12Hz-1.7B-CustomVoice/
  Qwen3-TTS-12Hz-1.7B-VoiceDesign/
  Qwen3-TTS-12Hz-1.7B-Base/
```

Only the checkpoint for the requested mode is loaded, in one owned subprocess
per take. Available modes work independently when only some checkpoints are
installed. Worker loading is offline, from local paths; missing dependencies or
files produce an error. Opening the dropdown never loads a model.

- **Preset speaker / CustomVoice:** nine speakers: Vivian, Serena, Uncle Fu,
  Dylan, Eric, Ryan, Aiden, Ono Anna and Sohee. Direct emotion, speaking pace,
  accent and delivery in the instruction field while keeping the preset timbre.
- **VoiceDesign:** describe a new voice, including age, timbre, accent, emotion
  and pace. A description is required. Generate a take, then choose **Use as
  clone reference** to keep that voice identity for another script.
- **Base voice clone:** upload a clean 1–60 second mono/stereo PCM WAV, or select
  a completed take. Enter its exact transcript for context-guided cloning, or
  choose speaker embedding only when no transcript is available. Clone mode
  follows the reference voice; it has no separate instruction field.
- English is the default language in the studio and API. All modes offer Auto language detection and Chinese, English, Japanese, Korean,
  German, French, Russian, Portuguese, Spanish and Italian. Cross-language cloning
  uses a reference voice and a different target language.
- Advanced settings expose speech-token sampling, temperature, top-p, top-k,
  repetition penalty, acoustic subtalker sampling and its temperature/top-p/top-k,
  maximum new tokens, and full-text versus native simulated streaming text mode.
  Defaults match the local 1.7B generation configuration. Preset/design default to full-text mode and cloning to simulated streaming.
  Both text modes return a
  complete WAV; incremental audio playback is not implemented.
- Seeds are applied through PyTorch before model loading; identical settings do
  not guarantee bit-identical GPU results. A low token limit can truncate speech.

Tiny Voca's age/pitch/speed sliders are hidden for Qwen; use native descriptions
and instructions where supported. Every take preserves its model and settings.

Live CUDA checks produced 24 kHz mono WAVs in CustomVoice, VoiceDesign and
Base cloning modes. Base passed with a reference transcript and with speaker
embedding only. A 529-character CustomVoice script produced a 36-second take.

## AuK · Base

Install the [official AuK source](https://github.com/Tencent-Hunyuan/AuK) in a
shared folder outside this repository with its own Python environment. Other
programs can use that environment's `auk-infer` CLI or `AukInfer` Python API.
The tested environment uses Python 3.10, PyTorch/TorchAudio 2.7.1,
TorchVision 0.22.1 and Transformers 4.57.6, plus soundfile and librosa.
An interpreter with the standard-library `lzma` module is required.

Choose these three paths in **Model setup…**: the environment's Python, the AuK
checkpoint folder, and the Omni encoder folder. Use this layout:

```text
shared/AuK/
  .venv/bin/python
  ckpts/
    AuK/
      config.yaml
      auk_base.safetensors
      vae.safetensors
    Qwen2.5-Omni-3B/
      config.json
      model-00001-of-00003.safetensors
      model-00002-of-00003.safetensors
      model-00003-of-00003.safetensors
      model.safetensors.index.json
      ... complete tokenizer and processor files ...
```

Copy AuK weights manually from [Tencent's checkpoint repository](https://huggingface.co/tencent/AuK).
The VAE is a separate required file. The encoder is also required: use the official
[Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) snapshot. GGUF, Qwen3-TTS checkpoints, text-only Qwen models and the Omni 7B variant
are incompatible with this adapter. AuK's learned layer fusion expects the
multimodal 3B Thinker's 2048-wide features. Changing encoder weights can affect
speech quality even when the shapes match.

- **Describe a voice:** enter a voice description and target script. English is
  the default instruction; describe a different target language explicitly when
  needed. AuK has no native language selector or fixed speaker catalog.
- **Clone a reference voice:** upload a clean 1–60 second PCM WAV or choose a
  completed take. No transcript is required. Optional delivery instructions are
  passed alongside the script and reference.
- Set the **target duration** to fit the script. It determines generated audio
  length; too short can truncate words, too long can add silence or repetition.
  The studio accepts 0.1–120 seconds per take (an integration limit).
- Sampling exposes diffusion steps, guidance, sway and seed. CPU offloading is
  optional and disabled by default; generation requires CUDA and substantial
  VRAM/RAM. Upstream measured about 17 GiB allocated VRAM with
  offload on short examples; this is not a guarantee for longer takes.
- A separate offline worker loads the local files only on Generate and releases
  them on exit. Opening model setup/discovery does not import AuK or load weights.
  AuK speech editing, enhancement and separation remain available through its
  standalone tools; this studio integration exposes speech generation.

AuK uses MIT licensing. The official Omni 3B encoder uses the **Qwen research
license**, restricted to noncommercial research/evaluation; commercial use needs
separate permission. Derivative releases must preserve its terms and notices.
See the [encoder license](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/main/LICENSE).
Built with Qwen.

Example request after all files are installed:

```json
{"model":"auk","input":"Welcome to the studio.","mode":"design",
 "instruct":"A warm, clear English narrator","gen_seconds":3,
 "cpu_offload":false,"seed":0,"response_format":"wav"}
```

Use `/v1/audio/speech` for a blocking WAV response or `/api/jobs` for cancellable
jobs. AuK uses `mode: design|clone`, `instruct`, `gen_seconds`, `nfe` (default 32),
`cfg_strength` (2), `sway_sampling_coef` (-1), `cpu_offload` (false), and `seed`.
Clone mode additionally needs a local `reference` ID. `/v1/audio/voices?model=auk`
returns an empty list because voices come from descriptions or references.

## Fish Audio · Drama 3 INT8

This adapter uses [PedroMarinhoDev/ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3)
with ComfyUI's Python libraries in a separate shared installation. No ComfyUI web
server is needed. The [official Fish Speech source](https://github.com/fishaudio/fish-speech)
uses a different checkpoint layout; its stock Drama 3 loader requires at least
48 GB VRAM and does not support this converted INT8 file. Keep these dependency
environments separate from other engines and from the official Fish API server.

In Model setup, choose the runtime Python, checkpoint directory, compatible
runtime source, and ComfyUI library source. Other projects can use the same Python
and source paths. Required layout:

```text
shared/fish-speech/
  .venv/bin/python
  checkpoints/drama-3/
    drama3_int8experts_bf16.safetensors
    drama3_codec.safetensors
shared/fish-speech-runtime/
  drama3/loader.py
  drama3/fish/tokenizer.json
  drama3/fish/line_head.safetensors
shared/fish-speech-comfy/
  comfy/text_encoders/qwen35.py
```

The main checkpoint is a manual copy. The separate converted codec is required;
its [checkpoint repository](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI)
contains both files. Runtime loading is offline. Configuration, tokenizer and
supplemental line-head rows come with the compatible runtime. No separate text encoder, VAE or ASR checkpoint
is needed. Cloning uses a supplied transcript rather than automatic ASR.
Discovery only checks paths; Generate loads local files with network loading disabled.

- **Generate a voice:** enter an English script by default. Use `[whispers]`,
  `[laughs]` and natural-language style cues inline. The model chooses the voice;
  this is not a fixed speaker catalog. Write other languages explicitly in the
  script. There is no separate language parameter.
- **Clone a reference voice:** upload a clean 1–60 second PCM WAV or reuse a
  completed take, and provide the exact spoken transcript. Upstream speaker tags
  such as `<|speaker:0|>` work in scripts and transcripts. The studio exposes
  one reference per take; additional references, editing and director revisions
  remain available through standalone upstream tools.
- Experts stay in **system RAM**, with a default **16 GiB expert cache in VRAM**.
  The rest of the model and working buffers need additional VRAM; the cache value
  is not total usage. Upstream measured about 21.5 GiB total on a 4090 with a
  16 GiB cache, and expects roughly 48 GB system RAM. Actual peaks depend on the
  prompt, driver, runtime and concurrent use. The cache is bounded by free memory.
- Temperature, top-p, top-k, maximum frames per chunk and seed are saved in takes
  and recipes. Long text is chunked with first/latest take context. Low frame
  limits can truncate speech. Seed reuse does not guarantee bit-identical results
  when CPU expert work and GPU execution differ.
- Performance tuning preserves the supplied INT8 checkpoint and normal sampling;
  it does not apply further quantization or reduce output length for speed.
- The owned worker releases RAM/VRAM on exit. Stop cancels only that worker;
  the adapter never stops another project's engines. Timeout is fifteen minutes,
  including loading. WAV output uses the existing exclusive-create policy.

Example API request:

```json
{"model":"fish-drama3","input":"[with warmth] Welcome to the studio.",
 "mode":"tts","expert_cache_gb":16,"temperature":1,"top_p":0.9,
 "top_k":30,"max_frames":1024,"seed":23,"response_format":"wav"}
```

Use `/api/jobs` for cancellable jobs or `/v1/audio/speech` for a blocking WAV.
Clone requests additionally use `mode: "clone"`, `reference` and `ref_text`.
`/v1/audio/voices?model=fish-drama3` returns an empty preset list.

Fish weights and derived runtime components carry the **Fish Audio Community
License**; preserve the upstream license and notices. ComfyUI is separately
licensed GPL-3.0. These sources and weights are not bundled in Mondovoca.

## Robin · Fixed Seed

Robin's source, isolated Python environment and model
folder are installed once outside this repository and can be shared with other
projects; Mondovoca never vendors Robin or copies its weights. Choose **Robin ·
Fixed Seed** in the audio model dropdown and the sidebar switches to Robin's own
controls: the Tiny Voca, Qwen3-TTS, AuK and Fish controls it does not use are
hidden and disabled. Selecting Robin, and saving Model setup, never start
inference or load weights.

Fill in **Model setup…** → *Robin · Fixed Seed · CUDA*: Python executable, model
folder, and an optional voice LoRA that may stay empty. Environment overrides
`MONDOVOCA_ROBIN_PYTHON`, `MONDOVOCA_ROBIN_ROOT` and `MONDOVOCA_ROBIN_LORA` take
precedence and appear read-only there. Saving only checks file availability.
Required layout, with every file in one flat folder:

```text
shared/robin/
  .venv/bin/python
  models/
    robin_v1.safetensors
    robin_vae.safetensors
    speech_tokenizer_v3_25hz.onnx
    config.json
    tokenizer.json
    robin_character_lora.safetensors   (optional voice adapter)
```

Use matching PyTorch and torchaudio builds in Robin's isolated environment.
This CUDA 13.0 combination has been verified with Mondovoca:

```bash
shared/robin/.venv/bin/python -m pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
```

A PyTorch 2.15.0/CUDA 13.2 environment was seen to segfault during CUDA
initialization, before loading weights. File discovery cannot detect runtime
compatibility, so generate one real take after installing. A worker crash names
its termination signal in the studio, with diagnostics in the
server log (`.mondovoca/server.log` when started with `start.sh`).

The VAE and the ONNX speech tokenizer must stay beside the main checkpoint:
Robin's loader resolves both relative to it. The tokenizer is always used as a
full local path, so it never fetches another copy by its canonical name. The
character LoRA is an optional voice adapter and never makes the base model
unavailable. Discovery only checks these paths: it does not import Robin, load
weights or fetch files.

- **Voice design:** choose *Describe a voice* and write the voice in plain words.
  `[laughs]` and `[coughs]` work inline anywhere in the script. The description may
  be empty: upstream keeps a language-only prompt, so a voice LoRA or a reference
  alone can carry the timbre.
- **Reference conditioning:** choose *Condition from a recording*, then upload a
  clean 1–60 second PCM WAV or pick a completed take from the same dropdown. **Use
  as clone reference** on any take fills this in for you. The exact transcript is
  optional. This is not Robin's separate voice-changing feature, which is not
  exposed yet, and voice LoRA training stays an upstream feature.
- **Languages:** the Language dropdown offers English (default), Japanese and
  English + Japanese; the API names them `en`, `ja` and `mix`.
- **Sampling:** every control is adjustable under *Robin sampling* with native
  inputs: temperature 0–2 (default 0.9, 0 is greedy), top-k 0–500 (0 = no limit,
  default 25), top-p 0.01–1 (default 0.8), voice guidance `cfg` 1–10 (default 1;
  above 1 follows the description more closely), decoder steps 1–128 (default 32),
  sampler `dpm_2` or `dpmpp_2m`, **Read the script exactly as written**
  (`as_written`) to skip spelling out English numbers and symbols first, and the shared
  variation seed.
- **Voice LoRA:** the Voice LoRA dropdown offers *Model setup default* (the adapter
  chosen in Model setup, or none when that field is empty), *None (base voice)*,
  the character adapter found beside the checkpoint or in Model setup, and *Custom
  path…* for any absolute `.safetensors` file. Strength is 0–2 as a linked number
  and slider (0 leaves the base voice, 1 is the adapter as trained). Installing an
  adapter and saving Model setup refreshes the choices without loading a model and
  keeps a typed custom path. Each take records the adapter it actually used, so
  changing Model setup later does not rewrite a saved recipe, and **Reuse
  settings** or a saved preset selects that adapter again — as the matching choice
  or as a custom path. The API uses `lora: null` for the Model setup choice,
  `"lora": ""` for no adapter, and a path for an explicit one.
- Long scripts are split by Robin's own reading pipeline, with retakes of weak
  chunks and continuity between them, so no 45 second cap applies to a whole
  request; that budget only bounds one rendered chunk. The Robin worker gets one
  hour, including loading. Output is 48 kHz mono WAV written with the existing
  exclusive-create rule, so an existing take is never overwritten. Robin takes
  play, download, restore and export recipes like every other model's.
- Each take runs in an owned subprocess with `HF_HUB_OFFLINE=1` and an absolute
  local model folder, so nothing is downloaded. Loading an adapter hashes the base
  checkpoint once into Robin's own local cache; that is normal work, not a
  download. Robin jobs in different projects each load their own weights: the
  shared installation holds no resident model and no global generation lock.
- Live CUDA checks produced 48 kHz mono WAVs with the supplied character LoRA,
  reference conditioning, and a 529-character script rendered across four
  chunks. Clicking Stop during CUDA inference terminated the worker and left
  no partial take.

Example request:

```json
{"model":"robin","input":"Okay, okay, [laughs] I really didn't think you'd do it!",
 "mode":"design","instruct":"A cheerful woman in her thirties, chatting casually",
 "language":"en","temperature":0.9,"top_k":25,"top_p":0.8,"cfg":3,"steps":32,
 "sampler":"dpm_2","as_written":false,"lora_strength":1,"seed":0,"response_format":"wav"}
```

Use `/api/jobs` for cancellable jobs or `/v1/audio/speech` for a blocking WAV.
Reference conditioning adds `mode: "reference"`, `reference` and optional
`ref_text`. `/v1/audio/voices?model=robin` returns an empty list because Robin's
voices come from descriptions, references and adapters rather than a preset catalog.

Robin is licensed under the **Fixed Seed Open License v1.0** (commercial use above
a revenue threshold needs a separate agreement); its bundled CosyVoice 3 speech
tokenizer is Apache-2.0. Robin source and weights are not redistributed by
Mondovoca. Built with Robin.
