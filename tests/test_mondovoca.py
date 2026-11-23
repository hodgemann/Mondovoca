"""Checks for optional studio controls and real HTTP boundary behavior."""
import base64
import io
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import wave
from unittest.mock import Mock
from http.server import ThreadingHTTPServer

from mondovoca import __file__ as package_init
from mondovoca.controls import age_style, settings
from mondovoca.server import Handler, Studio


class ControlsTests(unittest.TestCase):
    def test_all_ages_interpolate_and_manual_controls_override(self):
        for age in range(5, 101):
            style = age_style(age)
            actual = settings(dict(text="Hello", age=age))
            self.assertEqual(actual["speed"], style["speed"])
            self.assertTrue(.5 <= actual["pitch_range"] <= 1.5)
        self.assertEqual(age_style(30), dict(speed=1, semitones=0, pitch_range=1))
        self.assertEqual(settings(dict(text="Hello", age=5, speed=.8))["speed"], .8)

    def test_bad_controls_cannot_reach_the_model(self):
        for field, value in [("age", 4), ("age", 30.5), ("age", True), ("speed", 0),
                             ("speed", float("nan")), ("semitones", 7), ("pitch_range", 0),
                             ("voice", "alloy"), ("seed", -1), ("seed", 1.5), ("text", " "),
                             ("text", "a" * 5001)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                settings({"text": "Hello", field: value})


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.studio = Studio(root / "missing.pt", root / "missing", root / "clips")
        cls.server.instance = "test"
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def request(self, path, data=None, headers=None):
        request = urllib.request.Request(self.url + path, data=json.dumps(data).encode() if data is not None else None,
                                         headers=headers or {})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as response:
            return response.code, response.read()

    def test_missing_weights_is_actionable_and_invalid_settings_are_rejected(self):
        self.assertEqual(self.request("/api/jobs", {"text": "Hello"})[0], 503)
        self.assertEqual(self.request("/api/jobs", {"text": "Hello", "age": 400})[0], 400)

    def test_openwebui_can_discover_model_and_both_voices(self):
        status, raw = self.request("/v1/audio/voices")
        self.assertEqual(status, 200)
        self.assertEqual([v["id"] for v in json.loads(raw)["voices"]], ["F2", "M2"])
        self.assertEqual(json.loads(self.request("/v1/audio/models")[1])["models"][0]["id"], "tiny-voca")

    def test_goose_logo_and_favicon_are_served(self):
        for path, magic in (("/goose.png", b"\x89PNG"), ("/favicon.ico", b"\x00\x00\x01\x00")):
            status, raw = self.request(path)
            self.assertEqual(status, 200)
            self.assertTrue(raw.startswith(magic), path)

    def test_qwen_discovery_does_not_start_a_worker(self):
        models = json.loads(self.request('/api/config')[1])['models']
        self.assertEqual([m['id'] for m in models], ['tiny-voca', 'qwen3-tts', 'auk', 'fish-drama3', 'robin'])
        voices = json.loads(self.request('/v1/audio/voices?model=qwen3-tts')[1])['voices']
        self.assertEqual(len(voices), 9)
        self.assertIn('ryan', [v['id'] for v in voices])
        self.assertEqual(self.request('/v1/audio/voices?model=bad')[0], 400)
        self.assertEqual(self.request('/api/jobs', dict(model='qwen3-tts', mode='design', text='Hi'))[0], 400)
        self.assertEqual(self.request('/api/references', dict(audio='bad'))[0], 400)

    def test_auk_discovery_and_missing_installation(self):
        status, raw = self.request('/v1/audio/voices?model=auk')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)['voices'], [])
        status, raw = self.request('/api/jobs', dict(model='auk', text='Hello', instruct='Warm voice'))
        self.assertEqual(status, 503)
        self.assertIn('vae.safetensors', json.loads(raw)['error'])

    def test_robin_discovery_and_missing_installation(self):
        status, raw = self.request('/v1/audio/voices?model=robin')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)['voices'], [])
        model = next(m for m in json.loads(self.request('/api/config')[1])['models'] if m['id'] == 'robin')
        self.assertEqual(model['default_language'], 'English')
        self.assertEqual(model['languages'], ['en', 'ja', 'mix'])
        self.assertEqual(sorted(model['modes']), ['design', 'reference'])
        status, raw = self.request('/api/jobs', dict(model='robin', text='Hello', instruct='A warm, low voice'))
        self.assertEqual(status, 503)
        self.assertIn('robin_v1.safetensors', json.loads(raw)['error'])
        self.assertEqual(self.request('/api/jobs', dict(model='robin', text='Hello', mode='convert',
                                                        instruct='A warm, low voice'))[0], 400)

    def test_loopback_origin_and_file_boundaries(self):
        self.assertEqual(self.request("/api/jobs", {"text": "Hello"}, {"Origin": "https://elsewhere.example"})[0], 403)
        self.assertEqual(self.request("/api/health", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.request("/clips/../../models/tiny-voca.pt")[0], 404)
        self.assertEqual(self.request("/models/tiny-voca.pt")[0], 404)

    def test_output_folder_is_shown_and_validated(self):
        status, raw = self.request("/api/output-dir")
        self.assertEqual(status, 200)
        state = json.loads(raw)
        self.assertEqual(state["output_mode"], "browser")
        self.assertIsNone(state["output_dir"])
        config = json.loads(self.request("/api/config")[1])
        self.assertEqual(config["output_mode"], "browser")
        self.assertIsNone(config["output_dir"])
        for body in ({}, {"output_dir": ""}, {"output_dir": "takes"}, {"output_dir": 7}, {"output_mode": "nowhere"}):
            with self.subTest(body=body):
                self.assertEqual(self.request("/api/output-dir", body)[0], 400)
        self.assertEqual(json.loads(self.request("/api/output-dir")[1])["output_mode"], "browser")
        picked = Path(self.temp.name) / "picked"
        picked.mkdir(exist_ok=True)
        self.assertEqual(json.loads(self.request("/api/output-dir", {"folder_name": "picked"})[1])["output_dir"], str(picked))
        self.assertEqual(self.request("/api/output-dir", {"folder_name": "../.."})[0], 400)


class OutputFolderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.studio = Studio(self.root / "model", self.root / "phones", self.root / "clips", self.root / "models.json")

    def tearDown(self):
        self.studio.close()
        self.temp.cleanup()

    def saved_take(self, folder, job_id):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (job_id + ".wav")).write_bytes(b"saved audio")
        (folder / (job_id + ".json")).write_text(json.dumps(dict(id=job_id, state="done", controls={}, created=1,
                                                                seconds=1, elapsed=1, sample_rate=24000)))

    def publish(self, studio, folder, job_id, audio):
        pending = folder / (job_id + ".pending.wav")
        pending.write_bytes(audio)
        process = Mock(returncode=0)
        process.communicate.return_value = (json.dumps(dict(seconds=1, elapsed=1, sample_rate=24000)), "")
        job = dict(id=job_id, state="running", process=process, output_path=pending)
        studio.finish(job, {})
        return job

    def reference_wav(self, seconds=1, rate=8000):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(b"\0\0" * int(rate * seconds))
        return base64.b64encode(buffer.getvalue()).decode()

    def saved_folder_studio(self, name):
        # A studio restored from saved folder settings. Only browser-mode startup builds the working
        # spool, so this is exactly the tree a folder-mode shutdown leaves behind.
        root = self.root / name
        takes = root / "takes"
        takes.mkdir(parents=True)
        (root / "settings.json").write_text(json.dumps({"output_mode": "folder", "output_dir": str(takes)}))
        return Studio(root / "model", root / "phones", root / "clips", root / "models.json"), root, takes

    def test_browser_mode_keeps_one_working_copy_while_the_studio_runs(self):
        take = "c" * 32
        self.assertEqual(self.studio.output_mode, "browser")
        self.assertEqual(self.studio.data, self.root / "spool")
        self.assertEqual(self.publish(self.studio, self.root / "spool", take, b"new audio")["state"], "done")
        self.assertEqual((self.root / "spool" / (take + ".wav")).read_bytes(), b"new audio")
        self.studio.close()
        self.assertFalse((self.root / "spool" / (take + ".wav")).exists())
        self.assertFalse((self.root / "spool" / (take + ".json")).exists())

    def test_a_folder_saves_later_takes_and_keeps_earlier_history(self):
        first, second = self.root / "first", self.root / "takes"
        old, new = "a" * 32, "b" * 32
        self.studio.configure_output({"output_mode": "folder", "output_dir": str(first)})
        self.saved_take(first, old)
        state = self.studio.configure_output({"output_mode": "folder", "output_dir": str(second)})
        self.assertEqual(state["output_dir"], str(second))
        self.assertEqual(state["previous_output_dirs"], [str(first)])
        self.assertEqual(self.studio.clip_path(old).read_bytes(), b"saved audio")
        self.assertIn(old, self.studio.jobs)
        self.assertEqual(self.publish(self.studio, second, new, b"new audio")["state"], "done")
        self.assertEqual((second / (new + ".wav")).read_bytes(), b"new audio")
        self.assertEqual(json.loads((self.root / "settings.json").read_text())["output_dir"], str(second))
        reopened = Studio(self.root / "model", self.root / "phones", self.root / "clips", self.root / "models.json")
        try:
            self.assertEqual(reopened.output_mode, "folder")
            self.assertEqual(reopened.data, second)
            self.assertEqual(set(reopened.jobs), {old, new})
            self.assertEqual(reopened.clip_path(old).read_bytes(), b"saved audio")
        finally:
            reopened.close()

    def test_choosing_browser_downloads_again_returns_to_the_working_spool(self):
        folder = self.root / "takes"
        self.studio.configure_output({"output_mode": "folder", "output_dir": str(folder)})
        state = self.studio.configure_output({"output_mode": "browser", "output_dir": ""})
        self.assertEqual(state["output_mode"], "browser")
        self.assertIsNone(state["output_dir"])
        self.assertEqual(self.studio.data, self.root / "spool")
        self.assertEqual(json.loads((self.root / "settings.json").read_text())["output_mode"], "browser")
        self.assertEqual(self.publish(self.studio, self.root / "spool", "d" * 32, b"audio")["state"], "done")

    def test_a_saved_folder_startup_switching_to_browser_downloads_creates_the_spool(self):
        studio, root, takes = self.saved_folder_studio("absent")
        saved, fresh = "a" * 32, "b" * 32
        self.saved_take(takes, saved)
        self.assertEqual(studio.output_mode, "folder")
        self.assertEqual(studio.data, takes)
        self.assertFalse((root / "spool").exists())
        try:
            state = studio.configure_output({"output_mode": "browser", "output_dir": ""})
            self.assertEqual(state["output_mode"], "browser")
            self.assertIsNone(state["output_dir"])
            self.assertEqual(state["previous_output_dirs"], [str(takes)])
            self.assertEqual(studio.data, root / "spool")
            # Asserted before close(): shutdown builds the spool too and would hide this defect.
            self.assertTrue((root / "spool").is_dir())
            audio = self.reference_wav()
            reference = studio.upload_reference({"audio": audio})["reference"]
            self.assertEqual((root / "spool" / "references" / (reference + ".wav")).read_bytes(),
                             base64.b64decode(audio))
            self.assertEqual(studio.reference_path(reference).read_bytes(), base64.b64decode(audio))
            self.assertEqual(self.publish(studio, root / "spool", fresh, b"new audio")["state"], "done")
            self.assertEqual(studio.clip_path(fresh).read_bytes(), b"new audio")
            self.assertEqual(studio.clip_path(saved).read_bytes(), b"saved audio")
            self.assertIn(saved, studio.jobs)
            self.assertEqual(json.loads((root / "settings.json").read_text())["output_mode"], "browser")
        finally:
            studio.close()
        self.assertEqual(list((root / "spool").glob("*")), [])
        self.assertEqual((takes / (saved + ".wav")).read_bytes(), b"saved audio")
        self.assertTrue((takes / (saved + ".json")).is_file())

    def test_choosing_browser_downloads_discards_stale_spool_entries_and_keeps_saved_takes(self):
        studio, root, takes = self.saved_folder_studio("stale")
        saved, stale, interrupted, old_reference = "c" * 32, "d" * 32, "e" * 32, "f" * 32
        self.saved_take(takes, saved)
        self.saved_take(root / "spool", stale)
        (root / "spool" / (interrupted + ".pending.wav")).write_bytes(b"interrupted take")
        (root / "spool" / "references").mkdir()
        (root / "spool" / "references" / (old_reference + ".wav")).write_bytes(b"old reference")
        try:
            studio.configure_output({"output_mode": "browser", "output_dir": ""})
            self.assertTrue((root / "spool").is_dir())
            self.assertEqual(sorted(path.name for path in (root / "spool").iterdir()), [])
            self.assertNotIn(stale, studio.jobs)
            self.assertIsNone(studio.clip_path(stale))
            self.assertIsNone(studio.clip_path(interrupted))
            with self.assertRaises(ValueError):
                studio.reference_path(old_reference)
            self.assertEqual(studio.clip_path(saved).read_bytes(), b"saved audio")
            self.assertIn(saved, studio.jobs)
        finally:
            studio.close()
        self.assertEqual((takes / (saved + ".wav")).read_bytes(), b"saved audio")

    def test_a_saved_folder_stays_readable_after_browser_downloads_and_a_newer_folder(self):
        studio, root, first = self.saved_folder_studio("roundtrip")
        saved, browser_take = "a" * 32, "b" * 32
        self.saved_take(first, saved)
        try:
            studio.configure_output({"output_mode": "browser", "output_dir": ""})
            self.assertTrue((root / "spool").is_dir())
            self.assertEqual(self.publish(studio, root / "spool", browser_take, b"new audio")["state"], "done")
            second = root / "later"
            state = studio.configure_output({"output_mode": "folder", "output_dir": str(second)})
            self.assertEqual(state["output_dir"], str(second))
            self.assertEqual(state["previous_output_dirs"], [str(first)])
            self.assertFalse((root / "spool" / (browser_take + ".wav")).exists())
            self.assertNotIn(browser_take, studio.jobs)
            self.assertEqual(studio.clip_path(saved).read_bytes(), b"saved audio")
            self.assertEqual(self.publish(studio, second, "9" * 32, b"newest audio")["state"], "done")
            self.assertEqual(studio.clip_path("9" * 32).read_bytes(), b"newest audio")
        finally:
            studio.close()
        self.assertEqual((first / (saved + ".wav")).read_bytes(), b"saved audio")

    def test_output_folders_are_validated_before_the_switch(self):
        not_a_folder = self.root / "afile"
        not_a_folder.write_text("not a folder")
        for value in ("", "   ", None, 7, "takes", str(not_a_folder)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.studio.configure_output({"output_dir": value})
        self.assertEqual(self.studio.data, self.root / "spool")
        self.assertFalse((self.root / "settings.json").exists())

    def test_a_missing_writable_folder_is_created(self):
        target = self.root / "new" / "takes"
        self.assertEqual(self.studio.configure_output({"output_dir": str(target)})["output_dir"], str(target))
        self.assertTrue(target.is_dir())

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "the superuser ignores directory permissions")
    def test_a_folder_without_write_access_is_rejected(self):
        unwritable = self.root / "read-only"
        unwritable.mkdir()
        unwritable.chmod(0o500)
        try:
            with self.assertRaises(ValueError):
                self.studio.configure_output({"output_dir": str(unwritable)})
        finally:
            unwritable.chmod(0o700)

    def test_takes_saved_by_an_earlier_version_stay_listed_in_browser_mode(self):
        legacy, take = self.root / "clips", "e" * 32
        self.saved_take(legacy, take)
        studio = Studio(self.root / "model", self.root / "phones", legacy, self.root / "models.json")
        try:
            self.assertEqual(studio.output_mode, "browser")
            self.assertIn(take, studio.jobs)
            self.assertEqual(studio.clip_path(take).read_bytes(), b"saved audio")
        finally:
            studio.close()

    def test_the_working_spool_cannot_become_a_permanent_folder(self):
        for value in (str(self.root / "spool"), str(self.root / "spool" / "inside")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.studio.configure_output({"output_dir": value})
        self.assertEqual(self.studio.output_mode, "browser")

    def test_repeated_switches_do_not_duplicate_history_slots(self):
        first, second = self.root / "first", self.root / "second"
        for _ in range(3):
            self.studio.configure_output({"output_mode": "folder", "output_dir": str(first)})
            self.studio.configure_output({"output_mode": "browser", "output_dir": ""})
        state = self.studio.configure_output({"output_mode": "folder", "output_dir": str(second)})
        self.assertEqual(state["previous_output_dirs"], [str(first)])

    def test_a_launch_override_keeps_the_saved_folder_readable(self):
        saved, take = self.root / "saved", "f" * 32
        self.studio.configure_output({"output_mode": "folder", "output_dir": str(saved)})
        self.saved_take(saved, take)
        override = Studio(self.root / "model", self.root / "phones", self.root / "launch",
                          self.root / "models.json", output_locked=True)
        try:
            self.assertEqual(override.data, self.root / "launch")
            self.assertEqual(override.clip_path(take).read_bytes(), b"saved audio")
            self.assertIn(take, override.jobs)
        finally:
            override.close()

    def test_a_picked_folder_name_resolves_only_when_it_is_unique(self):
        picked = self.root / "Voices"
        picked.mkdir()
        self.assertEqual(self.studio.resolve_output("Voices"), dict(output_dir=str(picked)))
        (self.studio.data / "Voices").mkdir()
        for value in ("Nowhere", "Voices", "", "   ", "../secret", "a/b", None, 7):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.studio.resolve_output(value)

    def test_a_running_take_holds_the_output_folder(self):
        self.studio.jobs["running"] = dict(id="running", state="running", created=1)
        with self.assertRaises(BlockingIOError):
            self.studio.configure_output({"output_dir": str(self.root / "takes")})

    def test_a_launch_option_fixes_the_output_folder(self):
        locked = Studio(self.root / "model", self.root / "phones", self.root / "clips", self.root / "models.json",
                        output_locked=True)
        try:
            state = locked.output_state()
            self.assertTrue(state["locked"])
            self.assertEqual(state["output_mode"], "folder")
            self.assertEqual(state["output_dir"], str(self.root / "clips"))
            with self.assertRaises(ValueError):
                locked.configure_output({"output_mode": "browser", "output_dir": ""})
        finally:
            locked.close()


class OutputSafetyTests(unittest.TestCase):
    def test_a_filename_collision_preserves_the_saved_wav(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            studio = Studio(root / "model", root / "phones", root)
            job_id = "a" * 32
            final = studio.data / (job_id + ".wav")
            pending = studio.data / (job_id + ".pending.wav")
            final.write_bytes(b"previous saved audio")
            pending.write_bytes(b"new generated audio")
            process = Mock(returncode=0)
            process.communicate.return_value = (json.dumps(dict(seconds=1, elapsed=1, phonemes=[], sample_rate=24000)), "")
            job = dict(id=job_id, state="running", process=process, output_path=pending)
            studio.finish(job, {})
            self.assertEqual(final.read_bytes(), b"previous saved audio")
            self.assertEqual(job["state"], "error")
            self.assertFalse(pending.exists())


class StudioThemeTests(unittest.TestCase):
    """The Colour theme dropdown and the copied Mondocraft palette sheet stay in step."""

    @classmethod
    def setUpClass(cls):
        static = Path(package_init).resolve().parent / "static"
        page = (static / "index.html").read_text()
        dropdown = re.search(r'<select id="theme">(.*?)</select>', page, re.S).group(1)
        cls.choices = re.findall(r'<option value="([^"]*)"', dropdown)
        cls.palettes = (static / "themes.css").read_text()

    def test_theme_choices_match_palettes(self):
        palettes = re.findall(r'^html\[data-theme="([^"]+)"\]\s*\{[^}]*--bg:', self.palettes, re.M)
        # The studio's own :root defaults provide the teal palette.
        self.assertCountEqual(self.choices, ["teal", *palettes])

    def test_the_retro_crt_skins_are_offered(self):
        self.assertIn("crtgreen", self.choices)
        self.assertIn("crtamber", self.choices)


if __name__ == "__main__":
    unittest.main()
