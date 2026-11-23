"""Fish contracts: offline discovery, clone validation and owned-worker dispatch."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from mondovoca.controls import settings
from mondovoca.models import fish_catalog
from mondovoca.server import Studio


class FishTests(unittest.TestCase):
    def test_controls_and_memory_budget(self):
        c = settings(dict(model='fish-drama3', input='Hello [laughs]'))
        self.assertEqual(c['expert_cache_gb'], 16)
        self.assertEqual(c['mode'], 'tts')
        self.assertNotIn('cpu_offload', c)
        for override in (dict(expert_cache_gb=0), dict(expert_cache_gb=24),
                         dict(expert_cache_gb=float('nan')), dict(max_frames=1.5),
                         dict(mode='clone'), dict(mode='clone', reference='a'*32),
                         dict(mode='clone', reference='../etc', ref_text='Hello')):
            with self.subTest(override=override), self.assertRaises(ValueError):
                settings(dict(model='fish-drama3', input='Hello', **override))
        c = settings(dict(model='fish-drama3', input='Hello', mode='clone', reference='a'*32, ref_text='A reference'))
        self.assertEqual(c['ref_text'], 'A reference')

    def test_discovery_requires_both_checkpoints_and_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            values = {key: str(root/key) for key in ('fish_root', 'fish_python', 'fish_runtime', 'fish_comfy')}
            self.assertFalse(fish_catalog(values)['ready'])
            files = [root/'fish_python', root/'fish_root/drama3_int8experts_bf16.safetensors',
                     root/'fish_root/drama3_codec.safetensors', root/'fish_runtime/drama3/loader.py',
                     root/'fish_runtime/drama3/fish/tokenizer.json',
                     root/'fish_runtime/drama3/fish/line_head.safetensors',
                     root/'fish_comfy/comfy/text_encoders/qwen35.py']
            for file in files:
                file.parent.mkdir(parents=True, exist_ok=True); file.touch()
            self.assertTrue(fish_catalog(values)['ready'])
            files[2].unlink()
            self.assertFalse(fish_catalog(values)['ready'])

    def test_worker_dispatch_keeps_cuda_and_reference_and_can_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            studio=Studio(root/'tiny',root/'phones',root/'clips',root/'models.json')
            studio.configure_models({key:str(root/key) for key in ('fish_python','fish_root','fish_runtime','fish_comfy')})
            ref='a'*32
            (studio.data/'references').mkdir()
            (studio.data/'references'/(ref+'.wav')).touch()
            c=settings(dict(model='fish-drama3',text='Hi',mode='clone',reference=ref,ref_text='Hello'))
            with patch('mondovoca.server.subprocess.Popen') as popen, patch('mondovoca.server.threading.Thread') as thread:
                job=studio.start(c)
                self.assertTrue(popen.call_args.args[0][1].endswith('fish_worker.py'))
                self.assertNotEqual(popen.call_args.kwargs['env'].get('CUDA_VISIBLE_DEVICES'), '')
                payload=thread.call_args.kwargs['args'][1]
                self.assertEqual(payload['fish_root'], str(root/'fish_root'))
                self.assertEqual(payload['reference_path'], str(studio.reference_path(ref)))
                self.assertEqual(payload['controls']['expert_cache_gb'],16)
                popen.return_value.poll.return_value=None
                studio.cancel(job['id']);popen.return_value.terminate.assert_called_once()
