"""Single Qwen job, executed only by the isolated Qwen Python environment."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time


def synthesize(request):
    # Offline local paths only; importing this module never loads a model.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    c = request['controls']
    torch.manual_seed(c['seed'])
    started = time.monotonic()
    model = Qwen3TTSModel.from_pretrained(request['checkpoint'], device_map='cuda',
                                        dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    keys = ('do_sample', 'top_k', 'top_p', 'temperature', 'repetition_penalty',
            'subtalker_dosample', 'subtalker_top_k', 'subtalker_top_p',
            'subtalker_temperature', 'max_new_tokens', 'non_streaming_mode')
    kwargs = {key: c[key] for key in keys}
    kwargs.update(text=c['text'], language=c['language'])
    if c['mode'] == 'custom':
        waves, sr = model.generate_custom_voice(speaker=c['voice'], instruct=c['instruct'], **kwargs)
    elif c['mode'] == 'design':
        waves, sr = model.generate_voice_design(instruct=c['instruct'], **kwargs)
    else:
        audio, rate = sf.read(request['reference_path'], dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        waves, sr = model.generate_voice_clone(ref_audio=(audio, rate), ref_text=c['ref_text'] or None,
            x_vector_only_mode=c['x_vector_only_mode'], **kwargs)
    if not waves or not len(waves[0]):
        raise ValueError('Qwen3-TTS produced no audio')
    buffer = io.BytesIO()
    sf.write(buffer, waves[0], sr, format='WAV', subtype='PCM_16')
    with Path(request['output']).open('xb') as output:
        output.write(buffer.getvalue())
    return dict(seconds=round(len(waves[0]) / sr, 3), elapsed=round(time.monotonic()-started, 3),
                sample_rate=sr, device='CUDA', phonemes=[])


if __name__ == '__main__':
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        result = synthesize(request)
    print(json.dumps(result))
