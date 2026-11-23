"""Model catalogs and local installation metadata. Never imports an engine."""
import os
import json
import sys
from pathlib import Path

DEFAULT_LANGUAGE = 'English'
LANGUAGES = ['Auto', 'Chinese', 'English', 'Japanese', 'Korean', 'German', 'French', 'Russian', 'Portuguese', 'Spanish', 'Italian']
SPEAKERS = dict(vivian='Vivian · bright female · Chinese', serena='Serena · warm female · Chinese',
    uncle_fu='Uncle Fu · mellow male · Chinese', dylan='Dylan · clear male · Beijing dialect',
    eric='Eric · lively male · Sichuan dialect', ryan='Ryan · rhythmic male · English',
    aiden='Aiden · clear male · English', ono_anna='Ono Anna · playful female · Japanese',
    sohee='Sohee · warm female · Korean')
MODES = dict(custom='CustomVoice', design='VoiceDesign', clone='Base')
ROBIN_LANGUAGES = dict(en='English', ja='Japanese', mix='English + Japanese')
ROBIN_SAMPLERS = ('dpm_2', 'dpmpp_2m')
ROBIN_WEIGHTS = 'robin_v1.safetensors'
ROBIN_VAE = 'robin_vae.safetensors'
ROBIN_SPEECH = 'speech_tokenizer_v3_25hz.onnx'
ROBIN_LORA = 'robin_character_lora.safetensors'
# Ranges and defaults come from the installed Robin GUI sliders and config.json sampling block.
ROBIN_PARAMETERS = dict(temperature=(.9, 0, 2, False), top_k=(25, 0, 500, True), top_p=(.8, .01, 1, False),
                        cfg=(1, 1, 10, False), steps=(32, 1, 128, True))
PARAMETERS = dict(temperature=(.9, .05, 2, False), top_p=(1, .01, 1, False),
    top_k=(50, 0, 1000, True), repetition_penalty=(1.05, 1, 2, False),
    subtalker_temperature=(.9, .05, 2, False), subtalker_top_p=(1, .01, 1, False),
    subtalker_top_k=(50, 0, 1000, True), max_new_tokens=(8192, 32, 8192, True))


def installation_settings(config=None, checkpoint=None, phonemizer=None):
    values = {}
    if config and config.is_file():
        values = json.loads(config.read_text())
    default_root = Path.home() / '.local/share/mondovoca/models/qwen3-tts'
    defaults = Path.home() / '.local/share/mondovoca/models/tiny-voca'
    return dict(tiny_python=os.environ.get('MONDOVOCA_TINY_PYTHON', values.get('tiny_python', sys.executable)),
                tiny_checkpoint=os.environ.get('MONDOVOCA_CHECKPOINT', values.get('tiny_checkpoint', str(checkpoint or defaults / 'tiny-voca.pt'))),
                tiny_phonemizer=os.environ.get('DIZIONE_DATA', values.get('tiny_phonemizer', str(phonemizer or defaults / 'phonemizer'))),
                qwen_root=os.environ.get('MONDOVOCA_QWEN_ROOT', values.get('qwen_root', str(default_root))),
                qwen_python=os.environ.get('MONDOVOCA_QWEN_PYTHON', values.get('qwen_python', sys.executable)),
                fish_python=os.environ.get('MONDOVOCA_FISH_PYTHON', values.get('fish_python', sys.executable)),
                fish_root=os.environ.get('MONDOVOCA_FISH_ROOT', values.get('fish_root', str(default_root.parent / 'fish-drama3'))),
                fish_runtime=os.environ.get('MONDOVOCA_FISH_RUNTIME', values.get('fish_runtime', str(default_root.parent / 'fish-runtime'))),
                fish_comfy=os.environ.get('MONDOVOCA_FISH_COMFY', values.get('fish_comfy', str(default_root.parent / 'fish-comfy'))),
                auk_python=os.environ.get('MONDOVOCA_AUK_PYTHON', values.get('auk_python', sys.executable)),
                auk_root=os.environ.get('MONDOVOCA_AUK_ROOT', values.get('auk_root', str(default_root.parent / 'auk'))),
                auk_encoder=os.environ.get('MONDOVOCA_AUK_ENCODER', values.get('auk_encoder', str(default_root.parent / 'auk-encoder'))),
                robin_python=os.environ.get('MONDOVOCA_ROBIN_PYTHON', values.get('robin_python', sys.executable)),
                robin_root=os.environ.get('MONDOVOCA_ROBIN_ROOT', values.get('robin_root', str(default_root.parent / 'robin'))),
                robin_lora=os.environ.get('MONDOVOCA_ROBIN_LORA', values.get('robin_lora', '')))


def qwen_installation(config=None):
    values = installation_settings(config)
    root, python = Path(values['qwen_root']).expanduser(), Path(values['qwen_python']).expanduser()
    return python, {mode: root / ('Qwen3-TTS-12Hz-1.7B-' + suffix) for mode, suffix in MODES.items()}


def checkpoint_ready(path, mode):
    """Check a complete supported local layout without importing model code."""
    try:
        config = json.loads((path / 'config.json').read_text())
        expected = dict(custom='custom_voice', design='voice_design', clone='base')[mode]
        if (config.get('tts_model_type') != expected or config.get('tts_model_size') != '1b7'
                or config.get('tokenizer_type') != 'qwen3_tts_tokenizer_12hz'):
            return False
        for folder in (path, path / 'speech_tokenizer'):
            if not (folder / 'config.json').is_file() or not any(folder.glob('*.safetensors')):
                return False
            for index in folder.glob('*.safetensors.index.json'):
                shards = json.loads(index.read_text()).get('weight_map', {}).values()
                if not shards or not all((folder / shard).is_file() for shard in shards):
                    return False
        return all((path / name).is_file() for name in
                   ('generation_config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def catalog(checkpoint, phonemizer, config=None):
    python, paths = qwen_installation(config)
    environment = python.parent.parent
    package = any(environment.glob('lib/python*/site-packages/qwen_tts/__init__.py')) or (environment / 'Lib/site-packages/qwen_tts/__init__.py').is_file()
    modes = {mode: python.is_file() and package and checkpoint_ready(path, mode) for mode, path in paths.items()}
    values = installation_settings(config, checkpoint, phonemizer)
    return [dict(id='tiny-voca', name='Tiny Voca', default_language=DEFAULT_LANGUAGE, ready=Path(values['tiny_checkpoint']).is_file() and Path(values['tiny_phonemizer']).is_dir() and Path(values['tiny_python']).is_file(), device='CPU'),
            dict(id='qwen3-tts', name='Qwen3-TTS · 1.7B', ready=any(modes.values()), device='CUDA',
                 modes=modes, default_language=DEFAULT_LANGUAGE, languages=LANGUAGES, speakers=SPEAKERS, parameters=PARAMETERS), auk_catalog(values), fish_catalog(values),
            robin_catalog(values)]


def auk_catalog(values):
    root = Path(values['auk_root']).expanduser()
    encoder = Path(values['auk_encoder']).expanduser()
    python = Path(values['auk_python']).expanduser()
    missing = [str(root / name) for name in ('config.yaml', 'auk_base.safetensors', 'vae.safetensors')
               if not (root / name).is_file()]
    if not python.is_file():
        missing.append(str(python))
    environment = python.parent.parent
    if not (any(environment.glob('lib/python*/site-packages/auk-*.dist-info/METADATA'))
            or any(environment.glob('Lib/site-packages/auk-*.dist-info/METADATA'))):
        missing.append('AuK package installed in the configured Python environment')
    try:
        config = json.loads((encoder / 'config.json').read_text())
        thinker = config.get('thinker_config', config)
        if (config.get('model_type') not in ('qwen2_5_omni', 'qwen2_5_omni_thinker')
                or thinker['text_config']['hidden_size'] != 2048
                or thinker['text_config']['num_hidden_layers'] != 36
                or 'audio_config' not in thinker):
            raise ValueError('Requires the 3B multimodal Thinker')
        if not any(encoder.glob('*.safetensors')):
            raise ValueError('Missing safetensors')
        for index in encoder.glob('*.safetensors.index.json'):
            shards = json.loads(index.read_text()).get('weight_map', {}).values()
            if not shards or not all((encoder / shard).is_file() for shard in shards):
                raise ValueError('Missing encoder shards')
        for name in ('tokenizer_config.json', 'preprocessor_config.json', 'vocab.json', 'merges.txt'):
            if not (encoder / name).is_file():
                raise ValueError('Missing processor files')
        if not any((encoder / name).is_file() for name in ('chat_template.json', 'chat_template.jinja')):
            raise ValueError('Missing chat template')
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        missing.append(str(encoder) + ' (complete Qwen2.5-Omni-3B Transformers encoder)')
    return dict(id='auk', name='AuK · Base', ready=not missing, missing=missing,
                device='CUDA', modes=dict(design=True, clone=True), default_language=DEFAULT_LANGUAGE,
                license='MIT (AuK); encoder license applies separately',
                capabilities=['voice_design', 'voice_clone', 'instruction_tts'],
                parameters=dict(gen_seconds=(5, .1, 120, False), nfe=(32, 1, 128, True),
                                cfg_strength=(2, 0, 10, False), sway_sampling_coef=(-1, -1, 1, False)))


FISH_WEIGHT = 'drama3_int8experts_bf16.safetensors'
FISH_PARAMETERS = dict(temperature=(1, .1, 2, False), top_p=(.9, .01, 1, False),
                       top_k=(30, 1, 1000, True), max_frames=(1024, 1, 4096, True),
                       expert_cache_gb=(16, 1, 16, False))


def fish_catalog(values):
    root = Path(values['fish_root']).expanduser()
    runtime = Path(values['fish_runtime']).expanduser()
    comfy = Path(values['fish_comfy']).expanduser()
    required = [root / FISH_WEIGHT, root / 'drama3_codec.safetensors',
                Path(values['fish_python']).expanduser(),
                runtime / 'drama3/loader.py', runtime / 'drama3/fish/tokenizer.json',
                runtime / 'drama3/fish/line_head.safetensors',
                comfy / 'comfy/text_encoders/qwen35.py']
    missing = [str(path) for path in required if not path.is_file()]
    return dict(id='fish-drama3', name='Fish Audio · Drama 3 INT8', ready=not missing,
                missing=missing, device='CUDA + system RAM', default_language=DEFAULT_LANGUAGE,
                modes=dict(tts=True, clone=True), parameters=FISH_PARAMETERS,
                license='Fish Audio Community License; ComfyUI GPL-3.0 dependencies',
                capabilities=['instruction_tts', 'voice_clone', 'inline_speaker_tags'])


def robin_catalog(values):
    """Robin readiness: base weights, VAE, ONNX speech tokenizer, config and package. Never imports Robin."""
    root = Path(values['robin_root']).expanduser()
    python = Path(values['robin_python']).expanduser()
    # The loader resolves the VAE and the ONNX tokenizer beside the main checkpoint, so they must stay in one folder.
    missing = [str(root / name) for name in (ROBIN_WEIGHTS, ROBIN_VAE, ROBIN_SPEECH, 'config.json', 'tokenizer.json')
               if not (root / name).is_file()]
    if not python.is_file():
        missing.append(str(python))
    environment = python.parent.parent
    if not (any(environment.glob('lib/python*/site-packages/robin/__init__.py'))
            or (environment / 'Lib/site-packages/robin/__init__.py').is_file()):
        missing.append('Robin package installed in the configured Python environment')
    try:
        config = json.loads((root / 'config.json').read_text())
        # A malformed file can hold any JSON value, so every shape is checked before subscripting.
        if not isinstance(config, dict):
            raise ValueError('Model configuration must be a JSON object')
        if not all(key in config for key in ('model', 'tokens', 'sampling', 'instruction')):
            raise ValueError('Missing model, token, sampling or instruction configuration')
        tokens, sampling = config['tokens'], config['sampling']
        if not isinstance(tokens, dict) or not isinstance(sampling, dict):
            raise ValueError('Token and sampling configuration must be JSON objects')
        # Robin's pipeline reads these ids directly in its constructor, so a config without them
        # cannot render even when every file is present.
        if not all(key in tokens for key in ('eod', 'score_start', 'score_end', 'audio_start', 'audio_end',
                                             'audio_offset', 'audio_codes')):
            raise ValueError('Missing audio token layout')
        if sampling.get('decoder_sampler') not in ROBIN_SAMPLERS:
            raise ValueError('Unknown decoder sampler ' + str(sampling.get('decoder_sampler')))
        if not all(key in sampling for key in ('temperature', 'top_k', 'top_p', 'decoder_steps')):
            raise ValueError('Missing sampling defaults')
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        missing.append(str(root / 'config.json') + ' (complete Robin model configuration)')
    # The supplied character LoRA is an optional voice adapter and never gates base readiness.
    choice = str(values.get('robin_lora') or '').strip()
    adapter = Path(choice).expanduser() if choice else root / ROBIN_LORA
    return dict(id='robin', name='Robin · Fixed Seed', ready=not missing, missing=missing, device='CUDA',
                modes=dict(design=True, reference=True), default_language=DEFAULT_LANGUAGE,
                languages=list(ROBIN_LANGUAGES), language_names=ROBIN_LANGUAGES, samplers=list(ROBIN_SAMPLERS),
                parameters=ROBIN_PARAMETERS, voice_lora=str(adapter) if adapter.is_file() else None,
                license='Fixed Seed Open License v1.0; CosyVoice 3 speech tokenizer Apache-2.0',
                capabilities=['voice_design', 'voice_lora', 'reference_conditioning', 'inline_nonverbal_tags', 'long_text'])
