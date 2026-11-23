"""One isolated CPU synthesis job. The studio owns and can cancel this process."""
import io
import json
import os
from pathlib import Path
import sys
import time


def main():
    request = json.load(sys.stdin)
    os.environ["DIZIONE_DATA"] = request["phonemizer"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import soundfile as sf
    import torch
    from tiny_voca.infer import load_model, match_frontend, speak_text
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mondovoca.controls import settings
    controls = settings(request["controls"])
    torch.set_num_threads(int(os.environ.get("MONDOVOCA_THREADS", "4")))
    started = time.monotonic()
    match_frontend(request["checkpoint"])
    model = load_model(request["checkpoint"], "cpu")
    wave, phones = speak_text(model, controls["text"], controls["voice"],
                              speed=controls["speed"],
                              pitch_shift=controls["semitones"] * .057762265,
                              pitch_range=controls["pitch_range"], seed=controls["seed"])
    buffer = io.BytesIO()
    sf.write(buffer, wave.numpy(), model.config.sample_rate, format="WAV", subtype="PCM_16")
    # Exclusive creation: never overwrite an existing saved clip, even on an ID collision.
    with Path(request["output"]).open("xb") as output:
        output.write(buffer.getvalue())
    print(json.dumps(dict(seconds=round(len(wave) / model.config.sample_rate, 3),
                          elapsed=round(time.monotonic() - started, 3),
                          sample_rate=model.config.sample_rate,
                          phonemes=[" ".join(p.symbols) for p in phones])))


if __name__ == "__main__":
    main()
