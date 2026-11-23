"""Validated synthesis controls, shared by the studio and speech API."""
import math
import re
from .models import (DEFAULT_LANGUAGE, LANGUAGES, SPEAKERS, MODES, PARAMETERS, FISH_PARAMETERS,
                     ROBIN_LANGUAGES, ROBIN_SAMPLERS, ROBIN_PARAMETERS)

# Heuristic style anchors, not age-conditioned model capabilities.
# age: speaking rate, pitch offset in semitones, pitch movement
AGE_ANCHORS = [(5, 1.08, 4.0, 1.15), (15, 1.04, 1.5, 1.10),
               (30, 1.0, 0.0, 1.0), (50, .97, -.5, .95),
               (70, .91, -1.0, .90), (100, .83, -1.5, .85)]


def number(value, name, low, high, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    if integer and result != int(result):
        raise ValueError(f"{name} must be a whole number")
    return int(result) if integer else result


def age_style(age):
    age = number(age, "Age", 5, 100, integer=True)
    for left, right in zip(AGE_ANCHORS, AGE_ANCHORS[1:]):
        if age <= right[0]:
            mix = (age - left[0]) / (right[0] - left[0])
            return dict(zip(("speed", "semitones", "pitch_range"),
                            [round(a + (b - a) * mix, 3) for a, b in zip(left[1:], right[1:])]))


def settings(body):
    if not isinstance(body, dict):
        raise ValueError("Expected a JSON object")
    text = body.get("text", body.get("input", ""))
    if not isinstance(text, str) or not text.strip() or len(text) > 5000:
        raise ValueError("Enter between 1 and 5,000 characters of text")
    model = body.get("model", "tiny-voca")
    if model not in ("tiny-voca", "qwen3-tts", "auk", "fish-drama3", "robin"):
        raise ValueError("Choose model tiny-voca, qwen3-tts, auk, fish-drama3 or robin")
    if model == "fish-drama3":
        return fish_settings(body, text.strip())
    if model == "robin":
        return robin_settings(body, text.strip())
    if model == "auk":
        return auk_settings(body, text.strip())
    if model == "qwen3-tts":
        return qwen_settings(body, text.strip())
    voice = body.get("voice", "M2")
    if voice not in ("F2", "M2"):
        raise ValueError("Choose voice F2 or M2")
    age = body.get("age")
    base = age_style(age) if age is not None else dict(speed=1, semitones=0, pitch_range=1)
    age = number(age, "Age", 5, 100, integer=True) if age is not None else None
    return dict(model=model, text=text.strip(), voice=voice, age=age,
                speed=number(body.get("speed", base["speed"]), "Speed", .5, 1.8),
                semitones=number(body.get("semitones", base["semitones"]), "Pitch", -6, 6),
                pitch_range=number(body.get("pitch_range", base["pitch_range"]), "Pitch movement", .5, 1.5),
                seed=number(body.get("seed", 0), "Seed", 0, 2147483647, integer=True))


def qwen_settings(body, text):
    mode = body.get('mode', 'custom')
    if mode not in MODES:
        raise ValueError('Choose custom, design or clone mode')
    language = body.get('language', DEFAULT_LANGUAGE)
    if language not in LANGUAGES:
        raise ValueError('Choose a supported Qwen language')
    result = dict(model='qwen3-tts', mode=mode, language=language, text=text,
                  seed=number(body.get('seed', 0), 'Seed', 0, 2147483647, integer=True))
    for key, default in [('do_sample', True), ('subtalker_dosample', True), ('non_streaming_mode', mode != 'clone')]:
        value = body.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(key + ' must be a boolean')
        result[key] = value
    for key, (default, low, high, integer) in PARAMETERS.items():
        result[key] = number(body.get(key, default), key, low, high, integer)
    if mode == 'custom':
        result['voice'] = body.get('voice', 'ryan')
        if result['voice'] not in SPEAKERS:
            raise ValueError('Choose a supported Qwen speaker')
    if mode in ('custom', 'design'):
        instruct = body.get('instruct', '')
        if not isinstance(instruct, str) or len(instruct) > 2000 or (mode == 'design' and not instruct.strip()):
            raise ValueError('Voice design requires a description; instructions allow at most 2,000 characters')
        result['instruct'] = instruct.strip()
    else:
        reference = body.get('reference', '')
        if not isinstance(reference, str) or not re.fullmatch(r'[a-f0-9]{32}', reference):
            raise ValueError('Upload reference audio or select a completed take')
        transcript = body.get('ref_text', '')
        if not isinstance(transcript, str) or len(transcript) > 5000:
            raise ValueError('Reference transcript allows at most 5,000 characters')
        embedding = body.get('x_vector_only_mode', False)
        if not isinstance(embedding, bool):
            raise ValueError('Embedding-only mode must be a boolean')
        if not embedding and not transcript.strip():
            raise ValueError('Enter the reference transcript or enable embedding-only cloning')
        result.update(reference=reference, ref_text=transcript.strip(), x_vector_only_mode=embedding)
    return result


def auk_settings(body, text):
    mode = body.get('mode', 'design')
    if mode not in ('design', 'clone'):
        raise ValueError('Choose design or clone mode for AuK')
    instruct = body.get('instruct', '')
    if not isinstance(instruct, str) or len(instruct) > 2000 or (mode == 'design' and not instruct.strip()):
        raise ValueError('Describe the AuK voice in at most 2,000 characters')
    # AuK has no native language selector; the language is part of its instruction.
    result = dict(model='auk', mode=mode, text=text, instruct=instruct.strip(),
                  seed=number(body.get('seed', 0), 'Seed', 0, 2147483647, True))
    for key, default, low, high, integer in [('gen_seconds', 5, .1, 120, False),
            ('nfe', 32, 1, 128, True), ('cfg_strength', 2, 0, 10, False),
            ('sway_sampling_coef', -1, -1, 1, False)]:
        result[key] = number(body.get(key, default), key, low, high, integer)
    offload = body.get('cpu_offload', False)
    if not isinstance(offload, bool):
        raise ValueError('cpu_offload must be a boolean')
    result['cpu_offload'] = offload
    if mode == 'clone':
        reference = body.get('reference', '')
        if not isinstance(reference, str) or not re.fullmatch(r'[a-f0-9]{32}', reference):
            raise ValueError('Upload reference audio or select a completed take')
        result['reference'] = reference
    return result


def fish_settings(body, text):
    mode = body.get('mode', 'tts')
    if mode not in ('tts', 'clone'):
        raise ValueError('Choose tts or clone mode for Fish Audio')
    result = dict(model='fish-drama3', mode=mode, text=text,
                  seed=number(body.get('seed', 0), 'Seed', 0, 2147483647, True))
    for key, (default, low, high, integer) in FISH_PARAMETERS.items():
        result[key] = number(body.get(key, default), key, low, high, integer)
    if mode == 'clone':
        reference = body.get('reference', '')
        transcript = body.get('ref_text', '')
        if not isinstance(reference, str) or not re.fullmatch(r'[a-f0-9]{32}', reference):
            raise ValueError('Upload reference audio or select a completed take')
        if not isinstance(transcript, str) or not transcript.strip() or len(transcript) > 5000:
            raise ValueError('Fish cloning requires the exact reference transcript (up to 5,000 characters)')
        result.update(reference=reference, ref_text=transcript.strip())
    return result


def robin_settings(body, text):
    # Robin reads scripts of any length through its own chunking; the 45 s budget is per render, not per request.
    mode = body.get('mode', 'design')
    if mode not in ('design', 'reference'):
        raise ValueError('Choose design (a voice in words) or reference (conditioning from a recording) for Robin')
    language = body.get('language', 'en')
    if language not in ROBIN_LANGUAGES:
        raise ValueError('Choose Robin language en, ja or mix')
    sampler = body.get('sampler', ROBIN_SAMPLERS[0])
    if sampler not in ROBIN_SAMPLERS:
        raise ValueError('Choose Robin sampler dpm_2 or dpmpp_2m')
    # Upstream accepts an empty description: pieces.read turns "" into None and Robin's tags stay
    # language-only, so a voice LoRA or a reference alone can carry the timbre.
    instruct = body.get('instruct', '')
    if not isinstance(instruct, str) or len(instruct) > 2000:
        raise ValueError('Robin voice description must be text of at most 2,000 characters')
    as_written = body.get('as_written', False)
    if not isinstance(as_written, bool):
        raise ValueError('as_written must be a boolean')
    # Absent means the LoRA chosen in Model setup; empty means no voice adapter for this take.
    lora = body.get('lora')
    if 'lora' not in body or lora is None:
        lora = None
    elif isinstance(lora, str) and lora.strip():
        lora = lora.strip()
    elif not isinstance(lora, str):
        raise ValueError('Robin voice LoRA must be a path, an empty string or null')
    else:
        lora = ''
    result = dict(model='robin', mode=mode, language=language, text=text, instruct=instruct.strip(),
                  sampler=sampler, as_written=as_written, lora=lora,
                  lora_strength=number(body.get('lora_strength', 1), 'LoRA strength', 0, 2),
                  seed=number(body.get('seed', 0), 'Seed', 0, 2147483647, True))
    for key, (default, low, high, integer) in ROBIN_PARAMETERS.items():
        result[key] = number(body.get(key, default), key, low, high, integer)
    if mode == 'reference':
        reference = body.get('reference', '')
        if not isinstance(reference, str) or not re.fullmatch(r'[a-f0-9]{32}', reference):
            raise ValueError('Upload reference audio or select a completed take')
        transcript = body.get('ref_text', '')
        if not isinstance(transcript, str) or len(transcript) > 5000:
            raise ValueError('Reference transcript allows at most 5,000 characters')
        result.update(reference=reference, ref_text=transcript.strip())
    return result
