"""Dependency-free local HTTP service; engines are imported only by workers."""
import argparse
import base64
import binascii
import io
import wave
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
from uuid import uuid4

from . import __version__
from .controls import AGE_ANCHORS, settings
from .models import catalog, qwen_installation, installation_settings, SPEAKERS

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "src/mondovoca").is_dir():
    ROOT = Path.home() / ".local/share/mondovoca"

STATIC = Path(__file__).with_name("static")
ID = re.compile(r"^[a-f0-9]{32}$")
OUTPUT_HISTORY = 8
MODEL_SETTINGS = ('qwen_python', 'qwen_root', 'tiny_python', 'tiny_checkpoint', 'tiny_phonemizer', 'auk_python', 'auk_root',
                  'auk_encoder', 'fish_python', 'fish_root', 'fish_runtime', 'fish_comfy', 'robin_python', 'robin_root',
                  'robin_lora')
# A setting may stay empty because the engine works without it; the Robin voice LoRA is optional.
OPTIONAL_MODEL_SETTINGS = ('robin_lora',)
ENVIRONMENT_SETTINGS = [('fish_root', 'MONDOVOCA_FISH_ROOT'), ('fish_python', 'MONDOVOCA_FISH_PYTHON'),
                        ('fish_runtime', 'MONDOVOCA_FISH_RUNTIME'), ('fish_comfy', 'MONDOVOCA_FISH_COMFY'),
                        ('auk_root', 'MONDOVOCA_AUK_ROOT'), ('auk_python', 'MONDOVOCA_AUK_PYTHON'),
                        ('auk_encoder', 'MONDOVOCA_AUK_ENCODER'), ('qwen_root', 'MONDOVOCA_QWEN_ROOT'),
                        ('qwen_python', 'MONDOVOCA_QWEN_PYTHON'), ('tiny_python', 'MONDOVOCA_TINY_PYTHON'),
                        ('tiny_checkpoint', 'MONDOVOCA_CHECKPOINT'), ('tiny_phonemizer', 'DIZIONE_DATA'),
                        ('robin_root', 'MONDOVOCA_ROBIN_ROOT'), ('robin_python', 'MONDOVOCA_ROBIN_PYTHON'),
                        ('robin_lora', 'MONDOVOCA_ROBIN_LORA')]
WORKER_TIMEOUT = 180
# Robin renders a long script chunk by chunk with retakes, so it needs far more than one take's budget.
WORKER_TIMEOUTS = {'qwen3-tts': 900, 'auk': 900, 'fish-drama3': 900, 'robin': 3600}


class Studio:
    def __init__(self, checkpoint, phonemizer, data, model_config=None, settings_path=None, output_locked=False):
        self.checkpoint, self.phonemizer = checkpoint, phonemizer
        self.model_config = model_config or data.parent / "models.json"
        self.settings_path = settings_path or self.model_config.parent / "settings.json"
        self.spool = self.settings_path.parent / "spool"
        # A launch option outranks the saved choice, so the studio cannot move clips out from under it.
        self.output_locked = output_locked
        saved = self.read_settings()
        chosen = saved.get("output_dir")
        chosen = Path(chosen).expanduser() if isinstance(chosen, str) and chosen.strip() else None
        if output_locked:
            self.output_mode = "folder"
        else:
            self.output_mode = "browser" if saved.get("output_mode") == "browser" or not chosen else "folder"
        self.data = self.spool if self.output_mode == "browser" else (data if output_locked else chosen)
        self.data.mkdir(parents=True, exist_ok=True)
        if self.output_mode == "browser":
            self.clear_spool()
        recorded = saved.get("previous_output_dirs")
        self.previous = []
        for value in [*(recorded if isinstance(recorded, list) else []), chosen, data]:
            if not isinstance(value, (str, Path)) or not str(value).strip():
                continue
            path = Path(value).expanduser()
            if path != self.data and path != self.spool and path.is_dir() and path not in self.previous:
                self.previous.append(path)
        self.previous = self.previous[-OUTPUT_HISTORY:]
        self.jobs = {}
        self.lock = threading.RLock()
        self.closed = False
        self.load_history()

    @property
    def roots(self):
        # Completed takes stay in the folder that saved them; older folders stay readable.
        return [self.data, *self.previous]

    def clear_spool(self):
        # In browser-download mode the server file is a working copy, never a second archive.
        shutil.rmtree(self.spool, ignore_errors=True)
        self.spool.mkdir(parents=True, exist_ok=True)

    def read_settings(self):
        try:
            values = json.loads(self.settings_path.read_text())
        except (OSError, ValueError):
            return {}
        return values if isinstance(values, dict) else {}

    def write_settings(self, values):
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.settings_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2))
        temporary.replace(self.settings_path)

    def load_history(self):
        # Interrupted jobs are not listed as completed clips. Only finalized metadata is history.
        entries = []
        for root in self.roots:
            for path in root.glob("*.json"):
                try:
                    entries.append((path.stat().st_mtime, path, root))
                except OSError:
                    continue
        for _, path, root in sorted(entries, key=lambda entry: entry[0], reverse=True)[:100]:
            try:
                job = json.loads(path.read_text())
                if ID.fullmatch(job["id"]) and (root / (job["id"] + ".wav")).is_file():
                    self.jobs.setdefault(job["id"], job)
            except (ValueError, OSError, KeyError, TypeError):
                continue

    def clip_path(self, job_id, suffix=".wav"):
        for root in self.roots:
            path = root / (job_id + suffix)
            if path.is_file():
                return path
        return None

    def models(self):
        return catalog(self.checkpoint, self.phonemizer, self.model_config)

    def configure_models(self, body):
        values = {}
        current = installation_settings(self.model_config, self.checkpoint, self.phonemizer)
        for key in MODEL_SETTINGS:
            value = body.get(key, current.get(key))
            if value is None:
                value = ''
            if not isinstance(value, str) or len(value) > 4096 or (key not in OPTIONAL_MODEL_SETTINGS and not value.strip()):
                raise ValueError('Enter a location for ' + key)
            value = value.strip()
            if value:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    raise ValueError('Model locations must be absolute paths')
                values[key] = str(path)
            else:
                values[key] = ''
        with self.lock:
            if any(j['state'] == 'running' for j in self.jobs.values()):
                raise BlockingIOError('Wait for generation to finish before changing model setup')
            self.model_config.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.model_config.with_suffix('.tmp')
            temporary.write_text(json.dumps(values, indent=2))
            temporary.replace(self.model_config)
        return dict(settings=installation_settings(self.model_config, self.checkpoint, self.phonemizer), models=self.models())

    def output_state(self):
        return dict(output_mode=self.output_mode,
                    output_dir=None if self.output_mode == "browser" else str(self.data),
                    previous_output_dirs=[str(path) for path in self.previous],
                    locked=self.output_locked,
                    lock_source='MONDOVOCA_DATA_DIR' if os.environ.get('MONDOVOCA_DATA_DIR') and self.output_locked
                    else '--data-dir' if self.output_locked else '')

    def configure_output(self, body):
        if self.output_locked:
            raise ValueError('The output folder is fixed by this server launch (' + self.output_state()['lock_source'] + ')')
        mode = body.get('output_mode', 'folder')
        if mode not in ('browser', 'folder'):
            raise ValueError('Choose browser or folder for saved takes')
        value = body.get('output_dir')
        if mode == 'browser' and (value is None or isinstance(value, str) and not value.strip()):
            path = None
        else:
            if not isinstance(value, str) or not value.strip() or len(value) > 4096:
                raise ValueError('Enter an output folder')
            path = Path(value).expanduser()
            if not path.is_absolute():
                raise ValueError('The output folder must be an absolute path')
        with self.lock:
            if any(j['state'] == 'running' for j in self.jobs.values()):
                raise BlockingIOError('Wait for generation to finish before changing the output folder')
            spool = self.spool.resolve()
            if path is not None:
                chosen = path.resolve()
                if chosen == spool or spool in chosen.parents:
                    raise ValueError('Choose a folder outside the studio working spool ' + str(self.spool))
                if self.output_mode == 'folder' and chosen == self.data.resolve():
                    return self.output_state()
                if path.exists() and not path.is_dir():
                    raise ValueError('Choose a folder, not a file: ' + str(path))
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise ValueError('Cannot create the output folder: ' + str(exc)) from None
                if not os.access(path, os.W_OK | os.X_OK):
                    raise ValueError('The studio cannot write to ' + str(path))
            elif self.output_mode == 'browser':
                return self.output_state()
            previous = [p for p in dict.fromkeys([*self.previous, self.data])
                        if p != self.spool and p != path][-OUTPUT_HISTORY:]
            saved = self.read_settings()
            saved.update(output_mode='browser' if path is None else 'folder',
                         output_dir=None if path is None else str(path),
                         previous_output_dirs=[str(p) for p in previous])
            try:
                self.write_settings(saved)
            except OSError as exc:
                raise ValueError('Could not save the output setting: ' + str(exc)) from None
            # The working spool is rebuilt whenever browser mode is entered or left. Leaving it discards
            # those takes because they now live in the browser's download folder; entering it must create
            # the directory before uploads or workers use it and must not resurrect stale leftovers, as a
            # studio that starts in folder mode never creates the spool at all.
            if path is None or self.output_mode == 'browser':
                self.clear_spool()
            self.previous = previous
            self.data = self.spool if path is None else path
            self.output_mode = 'browser' if path is None else 'folder'
            self.jobs = {key: job for key, job in self.jobs.items() if self.clip_path(key)}
            self.load_history()
        return self.output_state()

    def resolve_output(self, name):
        # A browser folder picker reveals only the folder name, so resolve it against nearby folders.
        folder = name.strip() if isinstance(name, str) else ''
        if not folder or len(folder) > 255 or '/' in folder or '\\' in folder or folder in ('.', '..'):
            raise ValueError('Choose a folder in the picker first')
        home = Path.home()
        found = []
        for base in (self.data.parent, self.data, home, home / 'Documents', home / 'Music', home / 'Downloads', *self.previous):
            path = base / folder
            if path.is_dir() and path not in found:
                found.append(path)
        if len(found) == 1:
            return dict(output_dir=str(found[0]))
        if not found:
            raise ValueError('The browser only reveals the name “' + folder + '”. Enter the full path instead.')
        raise ValueError('Several folders are named “' + folder + '”. Enter the full path: ' + ' · '.join(map(str, found)))

    def reference_path(self, reference):
        for root in self.roots:
            path = root / 'references' / (reference + '.wav')
            if path.is_file():
                return path
        if self.jobs.get(reference, {}).get('state') == 'done':
            clip = self.clip_path(reference)
            if clip:
                return clip
        raise ValueError('Reference audio is missing; upload it again')

    def robin_lora_path(self, value):
        # A voice LoRA is an optional adapter file, chosen in Model setup or per request.
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError('Robin voice LoRA must be an absolute path')
        if path.suffix != '.safetensors' or not path.is_file():
            raise ValueError('Robin voice LoRA is missing: ' + str(path))
        return path

    def upload_reference(self, body):
        try:
            audio = base64.b64decode(body.get('audio', ''), validate=True)
            with wave.open(io.BytesIO(audio)) as wav:
                if wav.getframerate() <= 0:
                    raise ValueError('Reference WAV needs a valid sample rate')
                duration = wav.getnframes() / wav.getframerate()
                if not 1 <= duration <= 60 or wav.getnchannels() not in (1, 2):
                    raise ValueError('Reference must be 1–60 seconds, mono or stereo PCM WAV')
                if len(wav.readframes(wav.getnframes())) != wav.getnframes()*wav.getnchannels()*wav.getsampwidth():
                    raise ValueError('Reference WAV is truncated')
        except (wave.Error, EOFError, binascii.Error, TypeError):
            raise ValueError('Upload a valid PCM WAV reference') from None
        reference = uuid4().hex
        folder = self.data / 'references'
        folder.mkdir(exist_ok=True)
        with (folder / (reference + '.wav')).open('xb') as output:
            output.write(audio)
        return dict(reference=reference, seconds=duration)

    def public(self, job):
        return {key: value for key, value in job.items() if key not in ("process", "output_path")}

    def start(self, controls):
        with self.lock:
            if self.closed or any(j["state"] == "running" for j in self.jobs.values()):
                raise BlockingIOError("A voice is being generated. Stop it or wait for it to finish.")
            job_id = uuid4().hex
            while any((self.data / (job_id + suffix)).exists() for suffix in (".wav", ".json", ".pending.wav")):
                job_id = uuid4().hex
            output = self.data / (job_id + ".pending.wav")
            qwen = controls.get('model') == 'qwen3-tts'
            payload = dict(controls=controls, checkpoint=str(self.checkpoint),
                           phonemizer=str(self.phonemizer), output=str(output))
            env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONFAULTHANDLER='1')
            locations = installation_settings(self.model_config, self.checkpoint, self.phonemizer)
            command = [locations['tiny_python'], str(Path(__file__).with_name('worker.py'))]
            payload.update(checkpoint=locations['tiny_checkpoint'], phonemizer=locations['tiny_phonemizer'])
            if qwen:
                python, paths = qwen_installation(self.model_config)
                command = [str(python), str(Path(__file__).with_name('qwen_worker.py'))]
                payload['checkpoint'] = str(paths[controls['mode']])
                if controls['mode'] == 'clone':
                    payload['reference_path'] = str(self.reference_path(controls['reference']))
            elif controls.get('model') == 'fish-drama3':
                command = [locations['fish_python'], str(Path(__file__).with_name('fish_worker.py'))]
                payload.update({key: locations[key] for key in ('fish_root', 'fish_runtime', 'fish_comfy')})
                if controls['mode'] == 'clone':
                    payload['reference_path'] = str(self.reference_path(controls['reference']))
            elif controls.get('model') == 'auk':
                command = [locations['auk_python'], str(Path(__file__).with_name('auk_worker.py'))]
                payload.update(auk_root=locations['auk_root'], auk_encoder=locations['auk_encoder'])
                if controls['mode'] == 'clone':
                    payload['reference_path'] = str(self.reference_path(controls['reference']))
            elif controls.get('model') == 'robin':
                command = [locations['robin_python'], str(Path(__file__).with_name('robin_worker.py'))]
                payload['robin_root'] = locations['robin_root']
                lora = locations['robin_lora'] if controls.get('lora') is None else controls['lora']
                adapter = str(self.robin_lora_path(lora)) if lora else ''
                # The saved recipe records the adapter this take actually used, so a later Model setup
                # change cannot rewrite history. The caller's controls are left untouched.
                controls = dict(controls, lora=adapter)
                payload['controls'] = controls
                if adapter:
                    payload['lora_path'] = adapter
                if controls['mode'] == 'reference':
                    payload['reference_path'] = str(self.reference_path(controls['reference']))
            else:
                env['CUDA_VISIBLE_DEVICES'] = ''
            try:
                process = subprocess.Popen(command,
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           text=True, env=env)
            except OSError as exc:
                raise ValueError('Cannot start model Python. Check Model setup: ' + str(exc)) from None
            job = dict(id=job_id, state="running", controls=controls, created=time.time(), process=process,
                       output_path=output, output_mode=self.output_mode)
            self.jobs[job_id] = job
            threading.Thread(target=self.finish, args=(job, payload), daemon=True).start()
            return self.public(job)

    def finish(self, job, payload):
        process = job["process"]
        try:
            model = payload.get("controls", {}).get("model")
            out, err = process.communicate(json.dumps(payload), timeout=WORKER_TIMEOUTS.get(model, WORKER_TIMEOUT))
            with self.lock:
                if job["state"] == "cancelled":
                    return
                if process.returncode:
                    if err.strip():
                        print(f"Synthesis worker {job['id']} (exit {process.returncode}):\n{err}", file=sys.stderr, flush=True)
                    if process.returncode < 0:
                        try:
                            reason = signal.Signals(-process.returncode).name
                        except ValueError:
                            reason = f"signal {-process.returncode}"
                        error = f"Synthesis worker stopped by {reason}. See the server log for details."
                    else:
                        error = err.strip().splitlines()[-1] if err.strip() else f"Synthesis worker exited with code {process.returncode}"
                    job.update(state="error", error=error)
                else:
                    details = json.loads(out)
                    # Atomic no-replace publication. An existing final WAV is never removed or overwritten.
                    os.link(job["output_path"], self.data / (job["id"] + ".wav"))
                    job.update(state="done", **details)
                    job["url"] = "/clips/" + job["id"] + ".wav"
                    metadata = self.data / (job["id"] + ".json")
                    temporary = metadata.with_suffix(".tmp")
                    temporary.write_text(json.dumps(self.public(job), indent=2))
                    temporary.replace(metadata)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            with self.lock:
                if job["state"] != "cancelled":
                    job.update(state="error", error="Generation exceeded the worker time limit")
        except Exception as exc:
            with self.lock:
                if job["state"] != "cancelled":
                    job.update(state="error", error=str(exc))
        finally:
            with self.lock:
                job.pop("process", None)
                job["output_path"].unlink(missing_ok=True)

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job["state"] == "running":
                job["state"] = "cancelled"
                process = job.get("process")
                if process and process.poll() is None:
                    process.terminate()
            return self.public(job)

    def close(self):
        with self.lock:
            self.closed = True
            processes = [j["process"] for j in self.jobs.values() if "process" in j]
            for job in self.jobs.values():
                if job["state"] == "running":
                    self.cancel(job["id"])
        for process in processes:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self.output_mode == "browser":
            self.clear_spool()


class Handler(BaseHTTPRequestHandler):
    server_version = "Mondovoca/" + __version__

    def send(self, status, data, content_type="application/json"):
        if content_type == "application/json":
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; media-src 'self' blob:; img-src 'self' data:; frame-ancestors 'self'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def valid_host(self):
        # Restrict loopback service and reject browser DNS rebinding/cross-origin writes.
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        return host in allowed

    def do_GET(self):
        if not self.valid_host():
            return self.send(403, {"error": "Loopback host required"})
        path = urlsplit(self.path).path
        studio = self.server.studio
        if path == "/api/health":
            return self.send(200, dict(app="mondovoca", version=__version__, instance=self.server.instance,
                                       ready=any(m["ready"] for m in studio.models())))
        if path == '/api/model-settings':
            return self.send(200, dict(settings=installation_settings(studio.model_config, studio.checkpoint, studio.phonemizer),
                                       overrides=[key for key, env in ENVIRONMENT_SETTINGS if env in os.environ]))
        if path == "/api/config":
            return self.send(200, dict(age_anchors=AGE_ANCHORS, max_characters=5000, models=studio.models(),
                                       output_mode=studio.output_mode,
                                       output_dir=None if studio.output_mode == "browser" else str(studio.data)))
        if path == "/api/output-dir":
            return self.send(200, studio.output_state())
        if path == "/api/jobs":
            with studio.lock:
                return self.send(200, [studio.public(j) for j in sorted(studio.jobs.values(), key=lambda j: j["created"], reverse=True)][:100])
        if path.startswith("/api/jobs/"):
            with studio.lock:
                job = studio.jobs.get(path.rsplit("/", 1)[-1])
                return self.send(200, studio.public(job)) if job else self.send(404, {"error": "Unknown job"})
        if path in ("/v1/models", "/v1/audio/models"):
            models = [dict(**model, object="model", owned_by="local") for model in studio.models()]
            return self.send(200, dict(data=models, models=models, object="list"))
        if path == "/v1/audio/voices":
            model = parse_qs(urlsplit(self.path).query).get('model', ['tiny-voca'])[0]
            if model not in ('tiny-voca', 'qwen3-tts', 'auk', 'fish-drama3', 'robin'):
                return self.send(400, dict(error='Unknown model'))
            voices = {} if model in ('auk', 'fish-drama3', 'robin') else SPEAKERS if model == 'qwen3-tts' else dict(F2='Female · F2', M2='Male · M2')
            return self.send(200, dict(voices=[dict(id=k, name=v) for k, v in voices.items()]))
        if path.startswith("/clips/"):
            name = path.removeprefix("/clips/")
            if not name.endswith(".wav") or not ID.fullmatch(name[:-4]):
                return self.send(404, {"error": "Unknown clip"})
            with studio.lock:
                job = studio.jobs.get(name[:-4])
                clip = studio.clip_path(name[:-4]) if job and job["state"] == "done" else None
            if clip is None:
                return self.send(404, {"error": "Unknown clip"})
            return self.send(200, clip.read_bytes(), "audio/wav")
        assets = {"/": "index.html", "/app.js": "app.js", "/studio.css": "studio.css",
                  "/themes.css": "themes.css", "/tooltips.js": "tooltips.js",
                  "/goose.png": "goose.png", "/favicon.ico": "favicon.ico"}
        if path in assets:
            file = STATIC / assets[path]
            return self.send(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
        return self.send(404, {"error": "Not found"})

    def do_POST(self):
        if not self.valid_host():
            return self.send(403, {"error": "Loopback host required"})
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers['Host']}":
            return self.send(403, {"error": "Same-origin requests required"})
        path = urlsplit(self.path).path
        studio = self.server.studio
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= (12 * 1024 * 1024 if path == '/api/references' else 32768):
                return self.send(413, {"error": "Request exceeds the JSON size limit"})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError('Expected a JSON object')
            if path == '/api/model-settings':
                return self.send(200, studio.configure_models(body))
            if path == '/api/output-dir':
                if 'folder_name' in body and 'output_dir' not in body and 'output_mode' not in body:
                    return self.send(200, studio.resolve_output(body['folder_name']))
                return self.send(200, studio.configure_output(body))
            if path == '/api/references':
                return self.send(201, studio.upload_reference(body))
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                return self.send(200, studio.cancel(path.split("/")[-2]))
            if path in ("/api/jobs", "/v1/audio/speech"):
                controls = settings(body)
                if path == "/v1/audio/speech":
                    if body.get("response_format", "wav") != "wav":
                        raise ValueError("Mondovoca supports response_format wav")
                model = next(m for m in studio.models() if m['id'] == controls['model'])
                if not model['ready'] or (controls['model'] == 'qwen3-tts' and not model['modes'][controls['mode']]):
                    return self.send(503, {"error": "Missing model files or Python environment for this mode. " + "; ".join(model.get("missing", [])) + " See README.md."})
                job = studio.start(controls)
                if path == "/api/jobs":
                    return self.send(202, job)
                while job["state"] == "running":
                    time.sleep(.1)
                    with studio.lock:
                        job = studio.public(studio.jobs[job["id"]])
                if job["state"] != "done":
                    return self.send(422, {"error": job.get("error", "Generation cancelled")})
                clip = studio.clip_path(job["id"])
                if clip is None:
                    return self.send(422, {"error": "The generated audio file is missing"})
                return self.send(200, clip.read_bytes(), "audio/wav")
            return self.send(404, {"error": "Not found"})
        except BlockingIOError as exc:
            self.send(409, {"error": str(exc)})
        except (ValueError, TypeError) as exc:
            self.send(400, {"error": str(exc)})
        except KeyError:
            self.send(404, {"error": "Unknown job"})


def main():
    parser = argparse.ArgumentParser(description="Mondovoca local voice studio")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MONDOVOCA_PORT", "8093")))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("MONDOVOCA_CHECKPOINT", Path.home() / ".local/share/mondovoca/models/tiny-voca/tiny-voca.pt")))
    parser.add_argument("--phonemizer", type=Path, default=Path(os.environ.get("DIZIONE_DATA", Path.home() / ".local/share/mondovoca/models/tiny-voca/phonemizer")))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=Path(os.environ.get("MONDOVOCA_MODEL_CONFIG", ROOT / ".mondovoca/models.json")))
    parser.add_argument("--instance", default=uuid4().hex)
    args = parser.parse_args()
    environment_data = os.environ.get("MONDOVOCA_DATA_DIR")
    data = (args.data_dir or Path(environment_data or ROOT / ".mondovoca/clips")).resolve()
    # Bind the port first: a second studio must fail before it can clear the shared working spool.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    studio = Studio(args.checkpoint.resolve(), args.phonemizer.resolve(), data, args.model_config.resolve(),
                    output_locked=args.data_dir is not None or bool(environment_data))
    server.studio, server.instance = studio, args.instance
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"Mondovoca {__version__} · http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        studio.close()
        server.server_close()


if __name__ == "__main__":
    main()
