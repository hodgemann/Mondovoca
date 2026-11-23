"""Robin contracts: offline discovery, control ranges, LoRA optionality, worker args and dispatch."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mondovoca.controls import settings
from mondovoca.models import installation_settings, robin_catalog
from mondovoca.robin_worker import synthesize
from mondovoca.server import Studio, WORKER_TIMEOUT, WORKER_TIMEOUTS

CONFIG = dict(model=dict(hidden_size=2048), instruction='Generate speech',
              tokens=dict(eod=1, score_start=2, score_end=3, audio_start=4, audio_end=5, audio_offset=6, audio_codes=7),
              sampling=dict(temperature=.9, top_k=25, top_p=.8, pause_guard=40, guard_distinct=4,
                            decoder_steps=32, decoder_sampler='dpm_2'),
              audio=dict(sample_rate=48000, frame_rate=25))
ASSETS = ('robin_v1.safetensors', 'robin_vae.safetensors', 'speech_tokenizer_v3_25hz.onnx', 'config.json', 'tokenizer.json')


class RobinControlTests(unittest.TestCase):
    def test_defaults_follow_upstream_and_english_is_the_default_language(self):
        c = settings(dict(model='robin', input='Hello [laughs]', instruct='A warm, low voice'))
        self.assertEqual((c['mode'], c['language'], c['sampler']), ('design', 'en', 'dpm_2'))
        self.assertEqual((c['temperature'], c['top_k'], c['top_p'], c['cfg'], c['steps']), (.9, 25, .8, 1, 32))
        self.assertFalse(c['as_written'])
        self.assertIsNone(c['lora'])
        self.assertEqual(c['lora_strength'], 1)
        self.assertNotIn('voice', c)
        self.assertNotIn('age', c)
        self.assertNotIn('gen_seconds', c)
        for language in ('ja', 'mix'):
            self.assertEqual(settings(dict(model='robin', text='Hi', instruct='Warm', language=language))['language'], language)
        clone = settings(dict(model='robin', text='Hi', mode='reference', reference='a' * 32))
        self.assertEqual((clone['ref_text'], clone['instruct']), ('', ''))
        # Upstream keeps a language-only prompt for an empty description, so a voice LoRA or a
        # reference alone can carry the timbre. Only the type and length are enforced.
        blank = settings(dict(model='robin', text='Hi', instruct=''))
        self.assertEqual((blank['instruct'], blank['lora']), ('', None))
        adapter = settings(dict(model='robin', text='Hi', instruct='   ', lora='/shared/voice.safetensors'))
        self.assertEqual((adapter['instruct'], adapter['lora']), ('', '/shared/voice.safetensors'))
        self.assertEqual(settings(dict(model='robin', text='Hi'))['instruct'], '')
        self.assertEqual(settings(dict(model='robin', text='Hi', mode='reference', reference='a' * 32,
                                       instruct=''))['instruct'], '')
        self.assertEqual(settings(dict(model='robin', text='Hi', instruct='Warm', lora=''))['lora'], '')
        self.assertEqual(settings(dict(model='robin', text='Hi', instruct='Warm', lora='/shared/voice.safetensors'))['lora'],
                         '/shared/voice.safetensors')
        self.assertEqual(settings(dict(model='robin', text='Hi', mode='reference', reference='a' * 32,
                                       ref_text='Spoken words', sampler='dpmpp_2m', as_written=True,
                                       lora_strength=2, temperature=0, top_k=0, top_p=1, cfg=10, steps=128,
                                       language='mix'))['sampler'], 'dpmpp_2m')

    def test_invalid_modes_languages_ranges_and_references_are_rejected(self):
        base = dict(model='robin', text='Hello', instruct='A warm, low voice')
        for override in (dict(instruct=7), dict(instruct='x' * 2001), dict(mode='convert'), dict(mode='clone'),
                         dict(language='de'), dict(language='English'), dict(sampler='euler'), dict(sampler=1),
                         dict(temperature=2.5), dict(temperature=-1), dict(top_k=-1), dict(top_k=501), dict(top_k=1.5),
                         dict(top_p=0), dict(cfg=0), dict(cfg=11), dict(steps=0), dict(steps=129), dict(steps=1.5),
                         dict(as_written='yes'), dict(lora_strength=2.5), dict(lora_strength=-1), dict(seed=-1),
                         dict(seed=1.5), dict(lora=7), dict(mode='reference'),
                         dict(mode='reference', reference='../escape'),
                         dict(mode='reference', reference='a' * 32, ref_text='x' * 5001)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                settings(dict(base, **override))


class RobinCatalogTests(unittest.TestCase):
    def install(self, root, lora=False, config=CONFIG, package=True):
        values = dict(robin_python=str(root/'env/bin/python'), robin_root=str(root/'models'), robin_lora='')
        python = root/'env/bin/python'
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        if package:
            site = root/'env/lib/python3.12/site-packages/robin'
            site.mkdir(parents=True, exist_ok=True)
            (site/'__init__.py').touch()
        models = root/'models'
        models.mkdir(parents=True, exist_ok=True)
        for name in ASSETS:
            (models/name).touch()
        (models/'config.json').write_text(json.dumps(config))
        if lora:
            (models/'robin_character_lora.safetensors').touch()
        return values

    def test_readiness_needs_base_vae_onnx_config_tokenizer_and_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertFalse(robin_catalog(dict(robin_python='', robin_root=str(root), robin_lora=''))['ready'])
            values = self.install(root)
            self.assertTrue(robin_catalog(values)['ready'])
            self.assertIsNone(robin_catalog(values)['voice_lora'])
            for name in ASSETS:
                broken = root/'models'/name
                broken.unlink()
                self.assertFalse(robin_catalog(values)['ready'], name)
                broken.touch()
            (root/'models'/'config.json').write_text(json.dumps(CONFIG))
            self.assertTrue(robin_catalog(values)['ready'])
            (root/'env/bin/python').unlink()
            self.assertFalse(robin_catalog(values)['ready'])

    def test_the_optional_voice_lora_never_gates_base_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            values = self.install(root)
            self.assertTrue(robin_catalog(values)['ready'])
            self.assertIsNone(robin_catalog(values)['voice_lora'])
            adapter = root/'models'/'robin_character_lora.safetensors'
            adapter.touch()
            found = robin_catalog(values)
            self.assertTrue(found['ready'])
            self.assertEqual(found['voice_lora'], str(adapter))
            values['robin_lora'] = str(root/'chosen.safetensors')
            self.assertIsNone(robin_catalog(values)['voice_lora'])
            (root/'chosen.safetensors').touch()
            self.assertEqual(robin_catalog(values)['voice_lora'], str(root/'chosen.safetensors'))
            self.assertTrue(robin_catalog(values)['ready'])

    def test_a_partial_or_unknown_model_config_is_reported_as_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            values = self.install(root)
            without = lambda key: {k: v for k, v in CONFIG['tokens'].items() if k != key}
            for config in (dict(CONFIG, sampling=dict(CONFIG['sampling'], decoder_sampler='euler')),
                           {k: v for k, v in CONFIG.items() if k != 'sampling'},
                           dict(CONFIG, tokens=without('audio_offset')),
                           dict(CONFIG, tokens=without('score_start')),
                           dict(CONFIG, tokens=without('score_end')),
                           dict(CONFIG, tokens=list(CONFIG['tokens'])),
                           dict(CONFIG, sampling=list(CONFIG['sampling'])),
                           ['tokens'], None, 'not json'):
                with self.subTest(config=str(config)[:40]):
                    (root/'models'/'config.json').write_text(json.dumps(config) if not isinstance(config, str) else config)
                    result = robin_catalog(values)
                    self.assertFalse(result['ready'])
                    self.assertIn('config.json', ' '.join(result['missing']))
            # A malformed file never becomes an AttributeError from the catalog.
            (root/'models'/'config.json').write_text('{"tokens": [], "sampling": []}')
            self.assertIn('config.json', ' '.join(robin_catalog(values)['missing']))
            (root/'models'/'config.json').write_text(json.dumps(CONFIG))
            self.assertTrue(robin_catalog(values)['ready'])

    def test_setup_paths_are_private_portable_and_overridable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            saved = root/'models.json'
            saved.write_text(json.dumps(dict(robin_python=str(root/'env/bin/python'), robin_root=str(root/'models'),
                                             robin_lora='')))
            values = installation_settings(saved)
            self.assertEqual(values['robin_root'], str(root/'models'))
            self.assertEqual(values['robin_lora'], '')
            self.assertTrue(values['robin_python'])
            defaults = installation_settings(root/'absent.json')
            self.assertEqual(defaults['robin_root'], str(Path.home()/'.local/share/mondovoca/models/robin'))
            self.assertEqual(defaults['robin_lora'], '')
            with patch.dict(os.environ, {'MONDOVOCA_ROBIN_ROOT': str(root/'shared'), 'MONDOVOCA_ROBIN_LORA': str(root/'v.safetensors')}):
                overridden = installation_settings(saved)
                self.assertEqual(overridden['robin_root'], str(root/'shared'))
                self.assertEqual(overridden['robin_lora'], str(root/'v.safetensors'))


class RobinDispatchTests(unittest.TestCase):
    def test_native_crashes_report_the_signal_and_keep_diagnostics_in_the_log(self):
        import io
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            studio = Studio(root/'tiny.pt', root/'phones', root/'clips', root/'models.json')
            try:
                for code, stderr, expected in ((-11, '', 'SIGSEGV'),
                                                (-11, 'Fatal Python error: Segmentation fault\nNative stack', 'SIGSEGV'),
                                                (-9, '', 'SIGKILL'), (7, '', 'code 7'),
                                                (1, 'Traceback\nValueError: missing files', 'ValueError: missing files')):
                    with self.subTest(code=code, stderr=stderr):
                        process = Mock(returncode=code)
                        process.communicate.return_value = ('', stderr)
                        job = dict(id='failed', state='running', process=process, output_path=root/'pending.wav')
                        log = io.StringIO()
                        with redirect_stderr(log):
                            studio.finish(job, dict(controls=dict(model='robin')))
                        self.assertEqual(job['state'], 'error')
                        self.assertIn(expected, job['error'])
                        if stderr:
                            self.assertIn(stderr, log.getvalue())
            finally:
                studio.close()

    def studio(self, root, lora=True):
        studio = Studio(root/'tiny.pt', root/'phones', root/'clips', root/'models.json')
        adapter = root/'voice.safetensors'
        if lora:
            adapter.touch()
        studio.configure_models(dict(tiny_python=str(root/'python'), tiny_checkpoint=str(root/'tiny.pt'),
                                     tiny_phonemizer=str(root/'phones'), qwen_python=str(root/'python'),
                                     qwen_root=str(root), auk_python=str(root/'python'), auk_root=str(root/'auk'),
                                     auk_encoder=str(root/'encoder'), fish_python=str(root/'python'),
                                     fish_root=str(root/'fish'), fish_runtime=str(root/'runtime'),
                                     fish_comfy=str(root/'comfy'), robin_python=str(root/'robin-python'),
                                     robin_root=str(root/'models'), robin_lora=str(adapter) if lora else ''))
        return studio

    def test_setup_accepts_an_empty_optional_lora_and_rejects_relative_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            studio = self.studio(root, lora=False)
            try:
                saved = json.loads((root/'models.json').read_text())
                self.assertEqual(saved['robin_lora'], '')
                self.assertEqual(saved['robin_python'], str(root/'robin-python'))
                for body in (dict(robin_root='models'), dict(robin_python=''), dict(robin_python=7)):
                    with self.subTest(body=body):
                        with self.assertRaises(ValueError):
                            studio.configure_models(body)
                self.assertEqual(json.loads((root/'models.json').read_text())['robin_python'], str(root/'robin-python'))
            finally:
                studio.close()

    def test_dispatch_uses_the_shared_python_cuda_lora_and_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            studio = self.studio(root)
            reference = 'a' * 32
            (studio.data/'references').mkdir()
            (studio.data/'references'/(reference+'.wav')).touch()
            c = settings(dict(model='robin', text='Hello', instruct='A warm, low voice', mode='reference',
                              reference=reference, ref_text='Hello', lora_strength=.5, seed=19))
            try:
                with patch('mondovoca.server.subprocess.Popen') as popen, patch('mondovoca.server.threading.Thread') as thread:
                    job = studio.start(c)
                    self.assertTrue(popen.call_args.args[0][1].endswith('robin_worker.py'))
                    self.assertEqual(popen.call_args.args[0][0], str(root/'robin-python'))
                    self.assertNotEqual(popen.call_args.kwargs['env'].get('CUDA_VISIBLE_DEVICES'), '')
                    self.assertEqual(popen.call_args.kwargs['env']['PYTHONFAULTHANDLER'], '1')
                    payload = thread.call_args.kwargs['args'][1]
                    self.assertEqual(payload['robin_root'], str(root/'models'))
                    self.assertEqual(payload['lora_path'], str(root/'voice.safetensors'))
                    self.assertEqual(payload['reference_path'], str(studio.reference_path(reference)))
                    self.assertEqual(payload['controls']['lora_strength'], .5)
                    # The setup default is snapshotted into the job, so the saved recipe still says
                    # which adapter this take used after Model setup changes. The request is untouched.
                    self.assertIsNone(c['lora'])
                    self.assertEqual(job['controls']['lora'], str(root/'voice.safetensors'))
                    self.assertEqual(thread.call_args.kwargs['args'][0]['controls'], payload['controls'])
                    self.assertEqual(payload['controls']['lora'], str(root/'voice.safetensors'))
                    popen.return_value.poll.return_value = None
                    studio.cancel(job['id'])
                    popen.return_value.terminate.assert_called_once()
                # An explicit empty choice opts out of the configured adapter; a missing file is refused.
                without = settings(dict(model='robin', text='Hello', instruct='Warm', lora=''))
                with patch('mondovoca.server.subprocess.Popen') as popen, patch('mondovoca.server.threading.Thread') as thread:
                    second = studio.start(without)
                    payload = thread.call_args.kwargs['args'][1]
                    self.assertNotIn('lora_path', payload)
                    self.assertEqual(payload['controls']['lora'], '')
                    self.assertEqual(second['controls']['lora'], '')
                    self.assertEqual(without['lora'], '')
                    popen.return_value.poll.return_value = None
                    studio.cancel(second['id'])
                with self.assertRaises(ValueError):
                    studio.start(settings(dict(model='robin', text='Hello', instruct='Warm', lora=str(root/'gone.safetensors'))))
                with self.assertRaises(ValueError):
                    studio.start(settings(dict(model='robin', text='Hello', mode='reference', reference='b' * 32)))
            finally:
                studio.close()

    def test_long_scripts_get_their_own_timeout_and_other_models_keep_theirs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            studio = Studio(root/'tiny.pt', root/'phones', root/'clips', root/'models.json')
            try:
                for model, expected in (('robin', 3600), ('fish-drama3', 900), ('tiny-voca', WORKER_TIMEOUT)):
                    with self.subTest(model=model):
                        pending = studio.data/('job-'+model+'.pending.wav')
                        pending.write_bytes(b'audio')
                        process = Mock(returncode=1)
                        process.communicate.return_value = ('', 'stopped')
                        studio.finish(dict(id='job-'+model, state='running', process=process, output_path=pending),
                                     dict(controls=dict(model=model)))
                        self.assertEqual(process.communicate.call_args.kwargs['timeout'], expected)
                self.assertGreater(WORKER_TIMEOUTS['robin'], WORKER_TIMEOUTS['fish-drama3'])
            finally:
                studio.close()


class RobinWorkerTests(unittest.TestCase):
    def fakes(self, cuda=True):
        torch = Mock()
        torch.cuda.is_available.return_value = cuda
        robin = Mock()
        robin.pieces.RATE = 48000
        robin.pieces.read.return_value = [0.0] * 48000
        soundfile = Mock()
        soundfile.read.return_value = (Mock(), 24000)
        soundfile.write.side_effect = lambda buffer, *args, **kwargs: buffer.write(b'fixture WAV')
        return dict(torch=torch, robin=robin, soundfile=soundfile, numpy=Mock()), robin

    def request(self, root, name='take.wav', **overrides):
        controls = settings(dict(model='robin', text='Hello [laughs]', instruct='A warm, low voice', **overrides))
        request = dict(controls=controls, robin_root=str(root/'models'), output=str(root/name))
        if controls['mode'] == 'reference':
            request['reference_path'] = str(root/'reference.wav')
        return request

    def installed(self, root):
        for name in ASSETS:
            (root/'models'/name).parent.mkdir(parents=True, exist_ok=True)
            (root/'models'/name).touch()

    def test_worker_arguments_offline_mode_and_exclusive_48k_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.installed(root)
            modules, robin = self.fakes()
            engine = robin.Robin.from_pretrained.return_value
            request = self.request(root, seed=19, cfg=3, temperature=.7, top_k=0, top_p=1, steps=8,
                                   sampler='dpmpp_2m', as_written=True, lora='/shared/voice.safetensors', lora_strength=.5)
            request['lora_path'] = '/shared/voice.safetensors'
            with patch.dict('sys.modules', modules), patch.dict(os.environ, {}, clear=False):
                result = synthesize(request)
                self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
                with self.assertRaises(FileExistsError):
                    synthesize(request)
            # The existing take is refused before Robin is imported, so no weights are reloaded.
            self.assertEqual((robin.Robin.from_pretrained.call_count, robin.pieces.read.call_count), (1, 1))
            self.assertEqual((robin.Robin.from_pretrained.call_args.args, robin.Robin.from_pretrained.call_args.kwargs),
                             ((str(root/'models'),), dict(device='cuda')))
            engine.load_lora.assert_called_once_with('/shared/voice.safetensors', .5)
            args, kwargs = robin.pieces.read.call_args
            self.assertEqual(args[1:3], ('Hello [laughs]', 'A warm, low voice'))
            self.assertEqual(kwargs['lang'], 'en')
            self.assertEqual(kwargs['seed'], 19)
            self.assertEqual(kwargs['cfg'], 3)
            self.assertEqual((kwargs['temperature'], kwargs['top_k'], kwargs['top_p']), (.7, 0, 1))
            self.assertEqual((kwargs['steps'], kwargs['sampler'], kwargs['as_written']), (8, 'dpmpp_2m', True))
            self.assertIsNone(kwargs['reference'])
            self.assertEqual((result['sample_rate'], result['seconds'], result['device']), (48000, 1.0, 'CUDA'))
            self.assertEqual((root/'take.wav').read_bytes(), b'fixture WAV')

    def test_a_relative_model_folder_is_resolved_before_the_engine_sees_it(self):
        # Robin keeps a relative path relative and builds the ONNX tokenizer path from the
        # checkpoint's parent, so an environment override may not reach the loader unresolved.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.installed(root)
            modules, robin = self.fakes()
            request = self.request(root)
            request['robin_root'] = os.path.relpath(root/'models')
            with patch.dict('sys.modules', modules), patch.dict(os.environ, {}, clear=False):
                synthesize(request)
            self.assertEqual(robin.Robin.from_pretrained.call_args.args[0], str((root/'models').resolve()))
            (root/'models'/'speech_tokenizer_v3_25hz.onnx').unlink()
            request['output'] = str(root/'second.wav')
            with patch.dict('sys.modules', modules), self.assertRaises(ValueError) as caught:
                synthesize(request)
            self.assertIn(str((root/'models'/'speech_tokenizer_v3_25hz.onnx').resolve()), str(caught.exception))
            self.assertEqual(robin.Robin.from_pretrained.call_count, 1)

    def test_reference_conditioning_passes_audio_rate_and_optional_transcript(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.installed(root)
            modules, robin = self.fakes()
            engine = robin.Robin.from_pretrained.return_value
            request = self.request(root, mode='reference', reference='a' * 32, ref_text='Spoken words')
            with patch.dict('sys.modules', modules), patch.dict(os.environ, {}, clear=False):
                synthesize(request)
            engine.load_lora.assert_not_called()
            _, kwargs = robin.pieces.read.call_args
            self.assertEqual(kwargs['reference_rate'], 24000)
            self.assertEqual(kwargs['reference_text'], 'Spoken words')
            self.assertIsNotNone(kwargs['reference'])
            request = self.request(root, name='second.wav', mode='reference', reference='a' * 32)
            with patch.dict('sys.modules', modules), patch.dict(os.environ, {}, clear=False):
                synthesize(request)
            self.assertIsNone(robin.pieces.read.call_args.kwargs['reference_text'])

    def test_missing_files_and_missing_cuda_fail_before_any_weight_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            modules, robin = self.fakes()
            load = robin.Robin.from_pretrained
            with patch.dict('sys.modules', modules), self.assertRaises(ValueError) as caught:
                synthesize(self.request(root))
            self.assertIn('robin_v1.safetensors', str(caught.exception))
            load.assert_not_called()
            self.installed(root)
            (root/'models'/'speech_tokenizer_v3_25hz.onnx').unlink()
            with patch.dict('sys.modules', modules), self.assertRaises(ValueError) as caught:
                synthesize(self.request(root))
            self.assertIn('speech_tokenizer_v3_25hz.onnx', str(caught.exception))
            load.assert_not_called()
            self.installed(root)
            modules['torch'].cuda.is_available.return_value = False
            with patch.dict('sys.modules', modules), self.assertRaises(ValueError):
                synthesize(self.request(root))
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
