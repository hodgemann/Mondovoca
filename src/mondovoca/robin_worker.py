"""One offline Robin take in the shared Robin Python environment."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time

# Robin resolves the VAE and the ONNX speech tokenizer beside the main checkpoint, so these
# files stay together in one flat folder. The ONNX path must stay a full local path: s3tokenizer's
# canonical model name would fetch a copy outside Hugging Face offline mode.
REQUIRED = ('robin_v1.safetensors', 'robin_vae.safetensors', 'speech_tokenizer_v3_25hz.onnx',
            'config.json', 'tokenizer.json')


def synthesize(request):
    # An existing take is refused before Robin is imported or weights are loaded, so a collision
    # never costs a GPU load. The write below still uses exclusive mode against concurrent workers.
    output = Path(request['output']).expanduser()
    if output.exists():
        raise FileExistsError('Robin take already exists: ' + str(output))
    # Robin's loader falls back to a Hugging Face download for an unknown model path, so offline
    # mode is set before the engine is imported and only an absolute local folder is passed.
    os.environ['HF_HUB_OFFLINE'] = '1'
    import numpy as np
    import soundfile as sf
    import torch
    from robin import Robin
    from robin import pieces

    c = request['controls']
    # from_pretrained keeps a relative path relative and its speech property resolves the ONNX
    # tokenizer from the checkpoint's parent, so the folder is made absolute before it is passed.
    root = Path(request['robin_root']).expanduser().resolve()
    missing = [str(root / name) for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise ValueError('Robin files are missing: ' + '; '.join(missing))
    if not torch.cuda.is_available():
        raise ValueError('Robin requires CUDA in this integration')
    started = time.monotonic()
    engine = Robin.from_pretrained(str(root), device='cuda')
    lora = request.get('lora_path')
    if lora:
        # First use hashes the base checkpoint into Robin's own adapter cache; that is local work, not a download.
        engine.load_lora(lora, c['lora_strength'])
    reference = reference_rate = reference_text = None
    if c['mode'] == 'reference':
        samples, rate = sf.read(request['reference_path'], dtype='float32', always_2d=True)
        reference, reference_rate = samples.mean(axis=1), rate
        reference_text = c['ref_text'] or None
    # pieces.read keeps upstream sentence splitting, retakes of bad takes and continuity between
    # chunks, so a long script is never capped by the 45 s budget of a single render.
    audio = pieces.read(engine, c['text'], c['instruct'], reference=reference, reference_rate=reference_rate,
                        reference_text=reference_text, seed=c['seed'], cfg=c['cfg'], lang=c['language'],
                        as_written=c['as_written'], temperature=c['temperature'], top_k=c['top_k'],
                        top_p=c['top_p'], steps=c['steps'], sampler=c['sampler'])
    if not len(audio):
        raise ValueError('Robin produced no audio')
    buffer = io.BytesIO()
    # Robin's join() normalizes level between chunks and can exceed full scale; clip like its own studio.
    sf.write(buffer, np.clip(np.asarray(audio, dtype=np.float32), -1, 1), pieces.RATE, format='WAV', subtype='PCM_16')
    with output.open('xb') as take:
        take.write(buffer.getvalue())
    return dict(seconds=round(len(audio) / pieces.RATE, 3), elapsed=round(time.monotonic() - started, 3),
                sample_rate=pieces.RATE, device='CUDA', phonemes=[])


if __name__ == '__main__':
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        result = synthesize(request)
    print(json.dumps(result))
