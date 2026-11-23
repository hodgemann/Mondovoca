# API and Open WebUI

[Back to the README](../README.md)

The connection dialog shows the selected model's current parameters. For Open
WebUI's OpenAI TTS engine, use base URL `http://127.0.0.1:8093/v1`, placeholder key
`local`, selected model ID, and the dialog's extra JSON parameters. No existing
Open WebUI configuration is changed. Reference IDs must already exist in this
studio. A Docker installation needs separate networking configuration.

| Endpoint | Behavior |
| --- | --- |
| `GET /api/health` | Version, instance and whether any model's files are available |
| `GET /api/config` | Model catalog, mode availability, languages, speakers, limits, save location |
| `GET/POST /api/model-settings` | Read/save local engine locations |
| `GET/POST /api/output-dir` | Read/change where takes are saved; `{"output_mode":"browser"\|"folder","output_dir":"…"}` |
| `POST /api/output-dir` with `folder_name` | Resolve a browser folder picker's name to a local path |
| `POST /api/references` | Upload `{audio: "base64 PCM WAV"}`; returns reference ID |
| `POST /api/jobs` | Start a generation with model-specific settings and text |
| `GET /api/jobs`, `GET /api/jobs/{id}` | History or job status |
| `POST /api/jobs/{id}/cancel` | Cancel; send `{}` |
| `GET /clips/{id}.wav` | Completed audio |
| `POST /v1/audio/speech` | Blocking WAV response; script may use `input` |
| `GET /v1/audio/models`, `GET /v1/models` | Model discovery |
| `GET /v1/audio/voices?model=qwen3-tts` | Model-specific preset voices; defaults to Tiny Voca |

Model IDs are `tiny-voca` (default for existing recipes), `qwen3-tts`, `auk`,
`fish-drama3` and `robin`. Qwen
settings use `mode: custom|design|clone`, `language`, `seed`, and advanced parameter
names shown in the connection dialog. Custom mode uses `voice` and `instruct`;
design uses `instruct`; clone uses `reference`, `ref_text`, `x_vector_only_mode`.
AuK uses `mode: design|clone`; Fish uses `mode: tts|clone`; Robin uses
`mode: design|reference`, `language: en|ja|mix`, optional `instruct` (an empty
description keeps the base or adapter voice), `temperature`,
`top_k`, `top_p`, `cfg`, `steps`, `sampler`, `as_written`, `lora`, `lora_strength`
and `seed`. Robin reference mode takes `reference` and optional `ref_text`.
Only `response_format: wav` is supported. Invalid settings return 400; missing
installation files return 503; a busy studio returns 409. Clients should handle
409 and retry when idle.

The service binds only to IPv4 loopback and checks Host/Origin. It is for a trusted
local machine, with no remote access, account management or TLS. Model setup can
select local Python executables; only trusted local users should control it.
