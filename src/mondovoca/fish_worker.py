"""One offline Drama 3 take using the separate RAM-backed community runtime."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time


def initialize_memory():
    """Use upstream's dynamic allocator/graphs for this worker's selected GPU."""
    import comfy_aimdo.control as control
    # No process-wide NVML workload monitoring is needed by this isolated worker.
    control.init(nvml_pressure=False)
    import comfy.model_management as memory
    import comfy.model_patcher as patcher
    import comfy.memory_management as allocator
    from comfy.cli_args import args
    device = memory.get_torch_device()
    if not control.init_devices([(device.index, int(args.vram_headroom * 1024 ** 3))]):
        raise RuntimeError('Fish CUDA memory manager could not initialize; check the shared runtime environment')
    patcher.CoreModelPatcher = patcher.ModelPatcherDynamic
    allocator.aimdo_enabled = True
    return memory


def synthesize(request):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    # Import the engine libraries without importing the ComfyUI node package or
    # starting a server. This subprocess owns every model it loads.
    sys.path[:0] = [str(Path(request[key]).expanduser()) for key in ('fish_runtime', 'fish_comfy')]
    import torch
    import soundfile as sf
    if not torch.cuda.is_available():
        raise ValueError('Fish Drama 3 requires CUDA plus system RAM for its experts')
    memory = initialize_memory()
    from drama3.loader import load_drama3, load_codec
    from drama3.generate import generate
    from drama3.prompt import build_prompt
    from drama3.chunking import split_text
    from drama3.codec import SAMPLE_RATE, DECODE_CONTEXT, DECODE_WINDOW, DECODE_FRAME_BYTES
    from drama3.audio import tidy_clip

    started = time.monotonic()
    c = request['controls']
    root = Path(request['fish_root']).expanduser()
    codec = load_codec(str(root / 'drama3_codec.safetensors'))
    references = []
    with torch.inference_mode():
        if c['mode'] == 'clone':
            import torchaudio.functional as audio
            samples, rate = sf.read(request['reference_path'], dtype='float32', always_2d=True)
            wave = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0)
            if rate != SAMPLE_RATE:
                wave = audio.resample(wave, rate, SAMPLE_RATE)
            wave = tidy_clip(wave)
            memory.load_models_gpu([codec.patcher])
            with memory.cuda_device_context(codec.patcher.load_device):
                codes = codec.model.encode(wave.to(codec.patcher.load_device))[0].cpu()
            references = [(c['ref_text'], codes)]
            memory.unload_all_models()
        model = load_drama3(str(root / 'drama3_int8experts_bf16.safetensors'),
                            experts_in_ram=True, expert_cache_gb=c['expert_cache_gb'])
        memory.load_models_gpu(model.patchers())
        history, takes = [], []
        for index, chunk in enumerate(split_text(c['text']) or [c['text']]):
            context = history[:1] + history[1:][-1:]
            prompt = build_prompt(model.tokenizer, model.config, chunk, references, context)
            if len(prompt) + c['max_frames'] > 10240:
                raise ValueError('Fish prompt and generated frames exceed the model context; shorten the reference or script')
            frames = generate(model, prompt, c['max_frames'], c['temperature'], c['top_k'], c['top_p'], c['seed'] + index)
            valid = ((frames[:, 0] >= model.config.semantic_start_token_id)
                     & (frames[:, 0] <= model.config.semantic_end_token_id))
            codes = frames[valid, 1:].T.contiguous()
            if not codes.shape[-1]:
                raise ValueError('Fish Drama 3 generated no speech; try another seed or reference')
            history.append((chunk, frames))
            takes.append(codes)
        memory.unload_all_models()
        # Release the large RAM expert store before decoding the completed take.
        del model
        import gc
        gc.collect()
        waveforms = []
        for codes in takes:
            window = min(codes.shape[-1], DECODE_CONTEXT + DECODE_WINDOW)
            memory.load_models_gpu([codec.patcher], memory_required=window * DECODE_FRAME_BYTES)
            with memory.cuda_device_context(codec.patcher.load_device):
                waveforms.append(codec.model.decode(codes[None].to(codec.patcher.load_device), torch.float32).float().cpu())
        samples = torch.cat(waveforms, dim=-1).reshape(-1).numpy()
        buffer = io.BytesIO()
        sf.write(buffer, samples, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        with Path(request['output']).open('xb') as output:
            output.write(buffer.getvalue())
    return dict(seconds=round(len(samples) / SAMPLE_RATE, 3), elapsed=round(time.monotonic() - started, 3),
                sample_rate=SAMPLE_RATE, device='CUDA + system RAM', phonemes=[])


if __name__ == '__main__':
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        result = synthesize(request)
    print(json.dumps(result))
