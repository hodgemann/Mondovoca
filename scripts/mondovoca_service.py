"""Own only this project's studio process; never kill a service by port/name."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".mondovoca"
STATE = RUNTIME / "service.json"


def identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        # Parenthesized command names may contain spaces.
        fields = raw[raw.rfind(")") + 2:].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def owned(state):
    return state and identity(state["pid"]) == state["birth"]


def health(port):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/api/health", timeout=.5) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return {}


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action not in ("start", "stop", "status"):
        raise SystemExit("Usage: mondovoca_service.py start|stop|status")
    RUNTIME.mkdir(exist_ok=True)
    with (RUNTIME / "service.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = json.loads(STATE.read_text())
        except (OSError, ValueError):
            state = None
        if action == "status":
            if owned(state):
                print(f"Mondovoca is running: http://127.0.0.1:{state['port']}")
            else:
                print("Mondovoca is stopped")
            return
        if action == "stop":
            if owned(state):
                os.kill(state["pid"], signal.SIGTERM)
                for _ in range(100):
                    if not owned(state):
                        break
                    time.sleep(.1)
                else:
                    raise SystemExit("Studio is still stopping; see .mondovoca/server.log")
            STATE.unlink(missing_ok=True)
            print("Mondovoca stopped")
            return
        if owned(state):
            print(f"Mondovoca is already running: http://127.0.0.1:{state['port']}")
            return
        port = int(os.environ.get("MONDOVOCA_PORT", "8093"))
        instance = uuid4().hex
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        with (RUNTIME / "server.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-m", "mondovoca.server", "--port", str(port), "--instance", instance],
                                       cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, start_new_session=True)
        state = dict(pid=process.pid, birth=identity(process.pid), port=port, instance=instance)
        for _ in range(100):
            if process.poll() is not None:
                raise SystemExit("Mondovoca could not start. See .mondovoca/server.log (the port may be occupied).")
            if health(port).get("instance") == instance:
                STATE.write_text(json.dumps(state))
                print(f"Mondovoca is ready: http://127.0.0.1:{port}")
                return
            time.sleep(.1)
        process.terminate()
        process.wait(timeout=10)
        raise SystemExit("Mondovoca startup timed out. See .mondovoca/server.log")


if __name__ == "__main__":
    main()
