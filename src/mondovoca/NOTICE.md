# Attribution

Mondovoca is a local voice studio for separately installed speech engines. Its
first engine is tiny-voca by Muhammed Nazeem
(https://github.com/n4ze3m/tiny-voca), Apache-2.0. Tiny-voca model weights are
also Apache-2.0; its training dataset is CC BY 4.0. Model weights and training
data are not included in the standalone Mondovoca source package.

The studio follows Mondocraft's interface style. `static/tooltips.js` and
`static/themes.css` were copied from Mondocraft
(https://github.com/hodgemann/Mondocraft) on 2026-11-16. The original comments remain. `static/themes.css` was re-synced from
the same source on 2026-11-21 to bring in the green and amber phosphor CRT
skins, keeping the two copies byte-identical so they round-trip. Mondocraft's
source LICENSE is Apache-2.0. These assets and the remaining Mondovoca code are
included under Apache-2.0; this notice preserves their provenance.

`static/favicon.ico` is Mondocraft's goose favicon, copied unchanged on
2026-11-23. `static/goose.png` is a 96-pixel resize of the 256-pixel goose
head in Mondocraft's `packaging/windows/app.ico`, the same crop.

Mondocraft theme re-sync (green and amber phosphor CRT skins): Qwen Code.

Initial Mondovoca implementation: Codex, at hodgemann's direction.

Output-folder history refresh: Qwen Code, with review and verification by Codex.

Folder-to-browser spool fix and regression tests: Qwen Code, with review and
verification by Codex.

Qwen3-TTS is developed by the Qwen team at Alibaba Cloud
(https://github.com/QwenLM/Qwen3-TTS), Apache-2.0. Its optional runtime and
checkpoints are installed separately and are not redistributed in Mondovoca.

Qwen3-TTS integration: Codex, with read-only API and code review by Qwen Code.

AuK is developed by Tencent (https://github.com/Tencent-Hunyuan/AuK), MIT.
Its optional runtime, weights and VAE are separately installed, not redistributed
in Mondovoca. The separately installed Qwen2.5-Omni-3B encoder is subject to the
Qwen RESEARCH LICENSE AGREEMENT (noncommercial research/evaluation; commercial
use requires a separate license). Built with Qwen. License source:
https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/main/LICENSE

AuK integration: Codex.

Fish Audio Drama 3 is developed by Fish Audio. The optional INT8 adapter uses
PedroMarinhoDev/ComfyUI-Drama3 and separately installed ComfyUI libraries.
Fish model weights and derived components: Fish Audio Community License
(https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI).
Runtime licenses and notices: https://github.com/PedroMarinhoDev/ComfyUI-Drama3.
ComfyUI: GPL-3.0 (https://github.com/Comfy-Org/ComfyUI).
No upstream engine source or model weights are bundled. Built with Fish Audio.
Fish adapter implementation: Codex.

Robin is developed by Fixed Seed LLC (https://github.com/Fixed-Seed/Robin) under
the Fixed Seed Open License v1.0, which is based on Apache-2.0 and adds a
commercial-use revenue threshold. The bundled `speech_tokenizer_v3_25hz.onnx`
comes from CosyVoice 3 (https://github.com/FunAudioLLM/CosyVoice) under
Apache-2.0, as recorded in the model repository's NOTICE. Robin source, weights,
the optional voice LoRA and its Python environment are installed separately in
one shared location and are not redistributed in Mondovoca. Built with Robin.

Robin backend integration: Qwen Code, with supervision, review and verification
by Codex.

Robin studio UI and browser fixtures: Qwen Code, with review and verification by
Codex.

Release-prep README pass and notice reconciliation: Strata Scribe (Qwen Code).

README rewrite to the Mondocraft README layout and the documentation pages under
`docs/`: Claude.
