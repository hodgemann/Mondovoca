"""AuK contracts with fixture files and fake modules; no real model loading."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mondovoca.controls import settings
from mondovoca.models import auk_catalog
from mondovoca.server import Studio
from mondovoca.auk_worker import synthesize


class AukTests(unittest.TestCase):
    def test_validation(self):
        c = settings(dict(model='auk', text='Hello', instruct='Warm English narrator'))
        self.assertEqual(c['mode'], 'design')
        self.assertEqual(c['gen_seconds'], 5)
        self.assertFalse(c['cpu_offload'])
        self.assertNotIn('voice', c)
        self.assertNotIn('semitones', c)
        clone = settings(dict(model='auk', text='Hello', mode='clone', reference='a'*32))
        self.assertNotIn('ref_text', clone)
        for override in (dict(instruct=''), dict(mode='custom'), dict(gen_seconds=0),
                         dict(gen_seconds=float('nan')), dict(nfe=1.5), dict(cpu_offload='yes'),
                         dict(seed=True), dict(mode='clone', reference='../escape')):
            with self.subTest(override=override), self.assertRaises(ValueError):
                settings(dict(model='auk', text='Hello', **{'instruct':'Warm', **override}))

    def test_readiness_rejects_missing_files_shards_and_wrong_encoder(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); weights=root/'weights';weights.mkdir()
            encoder=root/'encoder';encoder.mkdir()
            python=root/'env/bin/python';python.parent.mkdir(parents=True);python.touch()
            metadata=root/'env/lib/python3.10/site-packages/auk-0.1.0.dist-info/METADATA'
            metadata.parent.mkdir(parents=True);metadata.touch()
            values=dict(auk_python=str(python),auk_root=str(weights),auk_encoder=str(encoder))
            self.assertFalse(auk_catalog(values)['ready'])
            for name in ('config.yaml','auk_base.safetensors','vae.safetensors'):(weights/name).touch()
            cfg=dict(model_type='qwen2_5_omni',thinker_config=dict(text_config=dict(hidden_size=2048,num_hidden_layers=36),audio_config={}))
            (encoder/'config.json').write_text(json.dumps(cfg))
            for name in ('model.safetensors','tokenizer_config.json','preprocessor_config.json','vocab.json','merges.txt','chat_template.json'):(encoder/name).touch()
            self.assertTrue(auk_catalog(values)['ready'])
            index=encoder/'model.safetensors.index.json'
            index.write_text(json.dumps(dict(weight_map=dict(weight='missing.safetensors'))))
            self.assertFalse(auk_catalog(values)['ready'])
            (encoder/'missing.safetensors').touch()
            self.assertTrue(auk_catalog(values)['ready'])
            cfg['thinker_config']['text_config']['hidden_size']=3584
            (encoder/'config.json').write_text(json.dumps(cfg))
            self.assertFalse(auk_catalog(values)['ready'])
            (encoder/'config.json').write_text('[]')
            self.assertFalse(auk_catalog(values)['ready'])

    def test_worker_dispatch_reference_and_cancellation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);studio=Studio(root/'tiny',root/'phones',root/'clips')
            reference='a'*32
            (studio.data/'references').mkdir()
            (studio.data/'references'/(reference+'.wav')).touch()
            c=settings(dict(model='auk',text='Hello',mode='clone',reference=reference))
            with patch('mondovoca.server.subprocess.Popen') as popen, patch('mondovoca.server.threading.Thread') as thread:
                job=studio.start(c)
                self.assertTrue(popen.call_args.args[0][1].endswith('auk_worker.py'))
                self.assertNotEqual(popen.call_args.kwargs['env'].get('CUDA_VISIBLE_DEVICES'), '')
                payload=thread.call_args.kwargs['args'][1]
                self.assertEqual(payload['reference_path'],str(studio.reference_path(reference)))
                self.assertIn('auk_encoder',payload)
                popen.return_value.poll.return_value=None
                studio.cancel(job['id'])
                popen.return_value.terminate.assert_called_once()

    def test_native_api_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);torch=Mock();sf=Mock();auk=Mock()
            wave=Mock()
            wave.detach.return_value.float.return_value.cpu.return_value.numpy.return_value.reshape.return_value=[0]*24000
            engine=auk.AukInfer.return_value
            engine.generate.return_value=(wave,24000)
            sf.write.side_effect=lambda buffer,*args,**kwargs:buffer.write(b'fixture WAV')
            modules={'torch':torch,'soundfile':sf,'auk.infer.infer_auk':auk}
            for mode in ('design','clone'):
                c=settings(dict(model='auk',text='Hello',mode=mode,instruct='Warm narrator',reference='a'*32,seed=19))
                request=dict(controls=c,auk_root='/local/weights',auk_encoder='/local/encoder',output=str(root/(mode+'.wav')))
                if mode=='clone':request['reference_path']='/local/reference.wav'
                with patch.dict('sys.modules',modules):
                    result=synthesize(request)
                    with self.assertRaises(FileExistsError):synthesize(request)
                kwargs=auk.AukInfer.call_args.kwargs
                self.assertEqual(kwargs['qwen_path'],'/local/encoder')
                self.assertEqual(kwargs['device'],'cuda')
                self.assertFalse(kwargs['cpu_offload'])
                args,kw=engine.generate.call_args
                self.assertEqual(kw['seed'],19)
                self.assertEqual(kw['nfe'],32)
                self.assertEqual(kw['gen_seconds'],5)
                self.assertIn('Hello',args[0][0]['content'][0]['text'])
                self.assertEqual(kw['audio'],'/local/reference.wav' if mode=='clone' else None)
                self.assertEqual(result['seconds'],1)
