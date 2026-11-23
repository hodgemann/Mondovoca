"""Qwen integration checks with fake engines only: never import real torch/qwen."""
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

from mondovoca.controls import settings
from mondovoca.models import PARAMETERS
from mondovoca.server import Studio
from mondovoca.qwen_worker import synthesize


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        wav.writeframes(b'\0\0' * 24000)
    return output.getvalue()


class QwenTests(unittest.TestCase):
    def test_modes_validation_and_legacy_recipes(self):
        self.assertEqual(settings(dict(text='Hello'))['model'], 'tiny-voca')
        self.assertEqual(settings(dict(model='qwen3-tts', text='Hello'))['language'], 'English')
        self.assertEqual(settings(dict(model='qwen3-tts', text='Hello', language='Auto'))['language'], 'Auto')
        for mode in ('custom', 'design', 'clone'):
            c = settings(dict(model='qwen3-tts', text='Bonjour', mode=mode, language='French',
                instruct='Warm and slow', reference='a'*32, ref_text='Hello'))
            self.assertEqual(c['mode'], mode)
            self.assertNotIn('semitones', c)
        for override in (dict(mode='bad'), dict(language='bad'), dict(voice='bad'),
                         dict(temperature=float('nan')), dict(top_k=1.2), dict(do_sample='false'),
                         dict(mode='design'), dict(mode='clone', reference='../etc'),
                         dict(mode='clone', reference='a'*32), dict(seed=True)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                settings(dict(model='qwen3-tts', text='Hi', **override))
        c = settings(dict(model='qwen3-tts', text='Hi', mode='clone', reference='a'*32, x_vector_only_mode=True))
        self.assertEqual(c['ref_text'], '')

    def test_portable_setup_persists_and_environment_overrides(self):
        from mondovoca.models import installation_settings
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root/'private/settings.json'
            studio = Studio(root/'tiny', root/'phones', root/'clips', config)
            paths = dict(tiny_python=str(root/'tiny-env/bin/python'), tiny_checkpoint=str(root/'weights/tiny.pt'),
                         tiny_phonemizer=str(root/'weights/phones'), qwen_python=str(root/'qwen-env/bin/python'),
                         qwen_root=str(root/'qwen-weights'), auk_python=str(root/'auk-env/bin/python'),
                         auk_root=str(root/'auk'), auk_encoder=str(root/'encoder'),
                         fish_python=str(root/'fish-env/bin/python'), fish_root=str(root/'fish'),
                         fish_runtime=str(root/'fish-runtime'), fish_comfy=str(root/'fish-comfy'),
                         robin_python=str(root/'robin-env/bin/python'), robin_root=str(root/'robin-models'),
                         robin_lora='')
            studio.configure_models(paths)
            self.assertEqual(installation_settings(config), paths)
            self.assertFalse(any(model['ready'] for model in studio.models()))
            with patch.dict('os.environ', MONDOVOCA_QWEN_ROOT=str(root/'override')):
                self.assertEqual(installation_settings(config)['qwen_root'], str(root/'override'))
            with self.assertRaises(ValueError):
                studio.configure_models(dict(paths, tiny_checkpoint='relative.pt'))
            studio.jobs['a'*32] = dict(state='running')
            with self.assertRaises(BlockingIOError):
                studio.configure_models(paths)

    def test_checkpoint_layout_rejects_partial_and_wrong_models(self):
        from mondovoca.models import checkpoint_ready
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = dict(tts_model_type='custom_voice', tts_model_size='1b7', tokenizer_type='qwen3_tts_tokenizer_12hz')
            (root/'config.json').write_text(json.dumps(config))
            (root/'model.safetensors').touch()
            self.assertFalse(checkpoint_ready(root, 'custom'))
            tokenizer = root/'speech_tokenizer';tokenizer.mkdir()
            (tokenizer/'config.json').write_text('{}');(tokenizer/'model.safetensors').touch()
            for name in ('generation_config.json','tokenizer_config.json','vocab.json','merges.txt'):
                (root/name).write_text('{}')
            self.assertTrue(checkpoint_ready(root, 'custom'))
            self.assertFalse(checkpoint_ready(root, 'clone'))
            (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=dict(a='missing.safetensors'))))
            self.assertFalse(checkpoint_ready(root, 'custom'))
            (root/'missing.safetensors').touch()
            self.assertTrue(checkpoint_ready(root, 'custom'))
        self.assertTrue(settings(dict(model='qwen3-tts', text='Hi'))['non_streaming_mode'])
        self.assertFalse(settings(dict(model='qwen3-tts', text='Hi', mode='clone', reference='a'*32, x_vector_only_mode=True))['non_streaming_mode'])

    def test_reference_upload_and_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            studio = Studio(root/'model', root/'phones', root)
            raw = wav_bytes()
            result = studio.upload_reference(dict(audio=base64.b64encode(raw).decode()))
            self.assertEqual(studio.reference_path(result['reference']).read_bytes(), raw)
            with self.assertRaises(ValueError):
                studio.upload_reference(dict(audio='broken'))
            with self.assertRaises(ValueError):
                studio.upload_reference(dict(audio=base64.b64encode(raw[:-20]).decode()))
            with self.assertRaises(ValueError):
                studio.reference_path('b'*32)

    def test_subprocess_isolation_and_cancel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            studio = Studio(root/'model', root/'phones', root)
            with patch('mondovoca.server.subprocess.Popen') as popen, patch('mondovoca.server.threading.Thread') as thread, patch.dict('os.environ', MONDOVOCA_QWEN_PYTHON='/fake/qwen/bin/python'):
                job = studio.start(settings(dict(model='qwen3-tts', text='Hello')))
                args, kwargs = popen.call_args
                self.assertEqual(args[0][0], '/fake/qwen/bin/python')
                self.assertTrue(args[0][1].endswith('qwen_worker.py'))
                payload = thread.call_args.kwargs['args'][1]
                self.assertTrue(payload['checkpoint'].endswith('CustomVoice'))
                self.assertEqual(payload['controls']['voice'], 'ryan')
                popen.return_value.poll.return_value = None
                studio.cancel(job['id'])
                popen.return_value.terminate.assert_called_once()
                studio.jobs.clear()
                studio.start(settings(dict(text='Hello')))
                self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')

    def test_native_api_contract_for_each_mode_without_model_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Mock modules rather than import GPU libraries or touch real weights.
            torch, sf, qwen, np = Mock(), Mock(), Mock(), Mock()
            model = qwen.Qwen3TTSModel.from_pretrained.return_value
            audio = Mock(ndim=1)
            sf.read.return_value = (audio, 24000)
            sf.write.side_effect = lambda buffer, *args, **kwargs: buffer.write(wav_bytes())
            for name in ('generate_custom_voice', 'generate_voice_design', 'generate_voice_clone'):
                getattr(model, name).return_value = ([list(range(24000))], 24000)
            modules = dict(torch=torch, soundfile=sf, qwen_tts=qwen, numpy=np)
            for mode, method in [('custom','generate_custom_voice'), ('design','generate_voice_design'), ('clone','generate_voice_clone')]:
                c = settings(dict(model='qwen3-tts', text='Hi', mode=mode, seed=123, language='Japanese',
                     instruct='Quiet', reference='a'*32, ref_text='Hello'))
                with patch.dict('sys.modules', modules):
                    result = synthesize(dict(controls=c, checkpoint='/fake/local/model',
                        reference_path='/fake/reference.wav', output=str(root/(mode+'.wav'))))
                kwargs = getattr(model, method).call_args.kwargs
                self.assertEqual(kwargs['language'], 'Japanese')
                for key in PARAMETERS:
                    self.assertEqual(kwargs[key], c[key])
                self.assertNotIn('seed', kwargs)  # seed belongs to torch, not generate kwargs
                torch.manual_seed.assert_called_with(123)
                self.assertEqual(result['seconds'], 1)
                self.assertEqual((root/(mode+'.wav')).read_bytes()[:4], b'RIFF')
                self.assertTrue(qwen.Qwen3TTSModel.from_pretrained.call_args.kwargs['local_files_only'])
            self.assertEqual(model.generate_custom_voice.call_args.kwargs['speaker'], 'ryan')
            self.assertEqual(model.generate_voice_design.call_args.kwargs['instruct'], 'Quiet')
            self.assertFalse(model.generate_voice_clone.call_args.kwargs['x_vector_only_mode'])
            self.assertEqual(model.generate_voice_clone.call_args.kwargs['ref_text'], 'Hello')


if __name__ == '__main__':
    unittest.main()
