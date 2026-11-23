"""Single AuK take in its independently installed Python environment."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time


def synthesize(request):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import soundfile as sf
    import torch
    from auk.infer.infer_auk import AukInfer

    c = request['controls']
    root = Path(request['auk_root']).expanduser()
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise ValueError('AuK requires CUDA in this integration')
    engine = AukInfer(config_path=str(root / 'config.yaml'),
                      ckpt_path=str(root / 'auk_base.safetensors'),
                      qwen_path=str(Path(request['auk_encoder']).expanduser()),
                      device='cuda', dtype='bf16', cpu_offload=c['cpu_offload'])
    if c['mode'] == 'clone':
        instruction = 'Say the following in English with the same voice as the reference audio: ' + json.dumps(c['text'], ensure_ascii=False)
    else:
        instruction = 'Generate speech in English with the following voice description: ' + c['instruct']
        instruction += '. Say the following: ' + json.dumps(c['text'], ensure_ascii=False)
    if c['mode'] == 'clone' and c['instruct']:
        instruction += '. ' + c['instruct']
    content = [dict(type='text', text=instruction)]
    audio = request.get('reference_path')
    if audio:
        content.append(dict(type='audio', audio=audio))
    wave, sr = engine.generate([dict(role='user', content=content)], audio=audio,
                              gen_seconds=c['gen_seconds'], nfe=c['nfe'],
                              cfg_strength=c['cfg_strength'], sway_sampling_coef=c['sway_sampling_coef'], seed=c['seed'])
    samples = wave.detach().float().cpu().numpy().reshape(-1)
    if not len(samples):
        raise ValueError('AuK produced no audio')
    buffer = io.BytesIO()
    sf.write(buffer, samples, sr, format='WAV', subtype='PCM_16')
    with Path(request['output']).open('xb') as output:
        output.write(buffer.getvalue())
    return dict(seconds=round(len(samples)/sr, 3), elapsed=round(time.monotonic()-started, 3),
                sample_rate=sr, device='CUDA', phonemes=[])


if __name__ == '__main__':
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        result = synthesize(request)
    print(json.dumps(result))
