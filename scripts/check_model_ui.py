"""Browser checks with fixture audio only. Never starts an inference worker."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from uuid import uuid4
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mondovoca.models import installation_settings
from mondovoca.server import Handler, Studio
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chromium', required=True)
    parser.add_argument('--shots', default='/tmp/mondovoca-model-ui')
    args = parser.parse_args()
    shots = Path(args.shots); shots.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        studio = Studio(root/'tiny.pt', root/'phones', root/'clips', root/'models.json')
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as audio:
            audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            audio.writeframes(b'\0\0'*24000)
        wav = buffer.getvalue()
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as audio:
            audio.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
            audio.writeframes(b'\0\0'*48000)
        wav48 = buffer.getvalue()
        requests = []
        # Robin's own config layout: model, token ids, sampling defaults and instruction settings.
        robin_config = dict(model={}, tokens=dict(eod=1, score_start=2, score_end=3, audio_start=4,
            audio_end=5, audio_offset=6, audio_codes=7), sampling=dict(decoder_sampler='dpm_2',
            temperature=.9, top_k=25, top_p=.8, decoder_steps=32), instruction={})

        def fake_start(controls):
            requests.append(controls)
            take, sample_rate = wav, 24000
            if controls.get('model') == 'robin':
                # Robin renders 48 kHz, and the adapter it really used is snapshotted into the saved
                # take exactly as the server does, so a later Model setup change cannot rewrite history.
                take, sample_rate = wav48, 48000
                chosen = (installation_settings(studio.model_config)['robin_lora']
                          if controls.get('lora') is None else controls['lora'])
                controls = dict(controls, lora=str(studio.robin_lora_path(chosen)) if chosen else '')
            job_id = uuid4().hex
            (studio.data/(job_id+'.wav')).write_bytes(take)
            job = dict(id=job_id, state='done', controls=controls, created=time.time(),
                       seconds=1, elapsed=.01, sample_rate=sample_rate, phonemes=[], url='/clips/'+job_id+'.wav',
                       output_mode=studio.output_mode)
            studio.jobs[job_id] = job
            return job

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.studio, server.instance = studio, 'fixture'
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with patch.object(studio, 'start', side_effect=fake_start), sync_playwright() as p:
                browser = p.chromium.launch(executable_path=args.chromium, headless=True)
                page = browser.new_page(viewport=dict(width=1440, height=1000))
                errors = []; page.on('pageerror', lambda e: errors.append(str(e)))
                page.goto(f'http://127.0.0.1:{server.server_port}')
                page.wait_for_function("() => !document.getElementById('generate').disabled")
                assert page.locator('#model option').count() == 5
                robin_option = page.locator('#model option[value=robin]')
                assert not robin_option.is_disabled()
                assert robin_option.get_attribute('title') is None
                assert page.locator('[name=voice]:checked').count() == 1
                assert not requests
                # Setup both engines at arbitrary temporary locations; no real installation paths.
                page.locator('#model-setup').click()
                (root/'tiny.pt').touch(); (root/'phones').mkdir()
                python = root/'env/bin/python'; python.parent.mkdir(parents=True); python.touch()
                package = root/'env/lib/python3.12/site-packages/qwen_tts'; package.mkdir(parents=True)
                (package/'__init__.py').touch()
                metadata=package.parent/'auk-0.1.0.dist-info';metadata.mkdir();(metadata/'METADATA').touch()
                # Robin is checked as an installed package in the same environment, never imported here.
                robin_package=package.parent/'robin';robin_package.mkdir();(robin_package/'__init__.py').touch()
                for suffix in ('CustomVoice','VoiceDesign','Base'):
                    model = root/('Qwen3-TTS-12Hz-1.7B-'+suffix); model.mkdir()
                    (model/'config.json').write_text(json.dumps(dict(tts_model_type={'CustomVoice':'custom_voice','VoiceDesign':'voice_design','Base':'base'}[suffix], tts_model_size='1b7',tokenizer_type='qwen3_tts_tokenizer_12hz')))
                    (model/'model.safetensors').touch()
                    for name in ('generation_config.json','tokenizer_config.json','vocab.json','merges.txt'):
                        (model/name).write_text('{}')
                    tokenizer=model/'speech_tokenizer';tokenizer.mkdir()
                    (tokenizer/'config.json').write_text('{}');(tokenizer/'model.safetensors').touch()
                # Robin keeps every asset in one flat folder; the optional character LoRA arrives later.
                robin=root/'robin'; robin.mkdir()
                for name in ('robin_v1.safetensors','robin_vae.safetensors','speech_tokenizer_v3_25hz.onnx','tokenizer.json'):
                    (robin/name).touch()
                (robin/'config.json').write_text(json.dumps(robin_config))
                adapter=str(robin/'robin_character_lora.safetensors')
                paths = dict(tiny_python=str(python), tiny_checkpoint=str(root/'tiny.pt'),
                    tiny_phonemizer=str(root/'phones'), qwen_python=str(python), qwen_root=str(root), auk_python=str(python),
                    auk_root=str(root/'auk'), auk_encoder=str(root/'encoder'), fish_python=str(python),
                    fish_root=str(root/'fish'), fish_runtime=str(root/'fish-runtime'), fish_comfy=str(root/'fish-comfy'),
                    robin_python=str(python), robin_root=str(robin), robin_lora='')
                for key,value in paths.items(): page.locator('#setup-'+key).fill(value)
                # The Robin voice LoRA is optional, so its field stays empty and still saves.
                assert not page.evaluate("() => document.getElementById('setup-robin_lora').required")
                page.get_by_role('button', name='Save model locations').click()
                page.wait_for_function("() => document.getElementById('setup-status').textContent.includes('files found')")
                assert 'Robin · Fixed Seed: files found' in page.locator('#setup-status').inner_text()
                assert json.loads((root/'models.json').read_text()) == paths
                page.locator('#close-setup').click()
                page.locator('#model').select_option('qwen3-tts')
                assert page.locator('#tiny-controls').is_hidden()
                assert page.locator('#robin-controls').is_hidden()
                assert page.locator('#compare').is_hidden()
                assert page.locator('#language option').count() == 11
                assert page.locator('#speaker option').count() == 9
                page.locator('#text').fill('こんにちは')
                page.locator('#autoplay').uncheck()
                page.locator('#language').select_option('Japanese')
                page.locator('#speaker').select_option('ono_anna')
                page.locator('#instruct').fill('Softly and with warmth')
                # Browser-download mode is the default: each finished take downloads once.
                assert page.locator('#output-location').inner_text() == 'My browser’s download folder'
                with page.expect_download() as first_download:
                    page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===1")
                assert first_download.value.suggested_filename.startswith('mondovoca-')
                assert list((root/'spool').glob('*.wav'))
                assert requests[-1]['voice'] == 'ono_anna'
                assert requests[-1]['instruct'] == 'Softly and with warmth'
                assert 'speed' not in requests[-1]
                page.locator('#qwen-mode').select_option('design')
                page.locator('#instruct').fill('A low, calm adult voice')
                page.locator('#preset-name').fill('Qwen narrator')
                page.locator('#save-preset').click()
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===2")
                assert requests[-1]['mode'] == 'design' and 'voice' not in requests[-1]
                page.get_by_role('button', name='Use as clone reference').first.click()
                assert page.locator('#qwen-mode').input_value() == 'clone'
                assert page.locator('#ref-text').input_value() == 'こんにちは'
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===3")
                assert requests[-1]['mode'] == 'clone' and not requests[-1]['x_vector_only_mode']
                page.locator('#reference-file').set_input_files(dict(name='sample.wav', mimeType='audio/wav', buffer=wav))
                page.wait_for_function("() => document.getElementById('status').textContent.includes('Reference uploaded')")
                page.locator('#embedding-only').check()
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===4")
                assert requests[-1]['x_vector_only_mode']
                page.get_by_role('button', name='Reuse settings').first.click()
                assert page.locator('#embedding-only').is_checked()
                page.locator('#integration').click()
                params = json.loads(page.locator('#api-params').inner_text())
                assert params['model'] == 'qwen3-tts' and params['mode'] == 'clone'
                page.keyboard.press('Escape')
                page.locator('#model').select_option('tiny-voca')
                assert page.locator('#qwen-controls').is_hidden()
                assert page.locator('#compare').is_visible()
                page.locator('#presets').select_option('Qwen narrator')
                assert page.locator('#model').input_value() == 'qwen3-tts'
                assert page.locator('#qwen-mode').input_value() == 'design'
                assert page.locator('#instruct').input_value() == 'A low, calm adult voice'
                # AuK design and clone use the same history/API with native controls.
                model = root/'auk';model.mkdir()
                for name in ('config.yaml','auk_base.safetensors','vae.safetensors'):(model/name).touch()
                encoder=root/'encoder';encoder.mkdir()
                (encoder/'config.json').write_text(json.dumps(dict(model_type='qwen2_5_omni',thinker_config=dict(text_config=dict(hidden_size=2048,num_hidden_layers=36),audio_config={}))))
                for name in ('model.safetensors','tokenizer_config.json','preprocessor_config.json','vocab.json','merges.txt','chat_template.json'):(encoder/name).touch()
                page.locator('#model-setup').click()
                page.get_by_role('button', name='Save model locations').click()
                page.locator('#close-setup').click()
                page.locator('#model').select_option('auk')
                assert page.locator('#qwen-controls').is_hidden()
                assert page.locator('#tiny-controls').is_hidden()
                assert page.locator('#auk-controls').is_visible()
                page.locator('#auk-instruct').fill('Warm English narrator')
                page.locator('#auk-gen_seconds').fill('7')
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===5")
                assert requests[-1]['model']=='auk' and requests[-1]['gen_seconds']==7
                page.get_by_role('button', name='Use as clone reference').first.click()
                assert page.locator('#model').input_value()=='auk'
                assert page.locator('#auk-mode').input_value()=='clone'
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===6")
                assert requests[-1]['mode']=='clone' and requests[-1]['reference']
                page.get_by_role('button', name='Reuse settings').first.click()
                assert page.locator('#auk-mode').input_value()=='clone'
                page.locator('#integration').click()
                assert json.loads(page.locator('#api-params').inner_text())['model']=='auk'
                page.keyboard.press('Escape')
                # Fish uses a RAM-backed engine, supplied transcript and saved memory controls.
                for name in ('fish/drama3_int8experts_bf16.safetensors','fish/drama3_codec.safetensors',
                             'fish-runtime/drama3/loader.py','fish-runtime/drama3/fish/tokenizer.json',
                             'fish-runtime/drama3/fish/line_head.safetensors','fish-comfy/comfy/text_encoders/qwen35.py'):
                    file=root/name;file.parent.mkdir(parents=True,exist_ok=True);file.touch()
                page.locator('#model-setup').click()
                page.get_by_role('button',name='Save model locations').click()
                page.locator('#close-setup').click()
                page.locator('#model').select_option('fish-drama3')
                assert page.locator('#fish-controls').is_visible()
                assert page.locator('#auk-controls').is_hidden()
                assert page.locator('#tiny-controls').is_hidden()
                page.locator('#text').fill('[with warmth] Welcome to the studio.')
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===7")
                assert requests[-1]['model']=='fish-drama3' and requests[-1]['expert_cache_gb']==16
                assert requests[-1]['temperature']==1 and requests[-1]['mode']=='tts'
                page.get_by_role('button',name='Use as clone reference').first.click()
                assert page.locator('#fish-mode').input_value()=='clone'
                assert page.locator('#fish-ref_text').input_value()=='[with warmth] Welcome to the studio.'
                page.locator('#fish-ref_text').fill('Welcome to the studio.')
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===8")
                assert requests[-1]['mode']=='clone' and requests[-1]['ref_text']=='Welcome to the studio.'
                page.get_by_role('button',name='Reuse settings').first.click()
                assert page.locator('#fish-mode').input_value()=='clone'
                page.locator('#integration').click()
                params=json.loads(page.locator('#api-params').inner_text())
                assert params['model']=='fish-drama3' and params['expert_cache_gb']==16
                page.keyboard.press('Escape')
                # Robin: native controls, LoRA choices, reference conditioning, recipes and presets.
                opened=len(requests)
                page.locator('#model').select_option('robin')
                assert page.locator('#robin-controls').is_visible()
                for other in ('#tiny-controls','#qwen-controls','#auk-controls','#fish-controls'):
                    assert page.locator(other).is_hidden(), other+' stays open beside Robin'
                assert page.locator('#compare').is_hidden() and page.locator('#reset').is_hidden()
                assert len(requests) == opened, 'Choosing Robin must not start a generation.'
                assert page.locator('#robin-language option').count() == 3
                assert page.locator('#robin-language option').first.text_content() == 'English'
                assert page.locator('#robin-language option[value=mix]').text_content() == 'English + Japanese'
                assert page.locator('#robin-language').input_value() == 'en'
                # No adapter is installed yet, so only the setup default, None and a custom path are offered.
                assert page.locator('#robin-lora option').count() == 3
                assert page.locator('#robin-lora option[value=%s]' % json.dumps(adapter)).count() == 0
                assert page.locator('#robin-lora-custom').is_hidden()
                assert page.evaluate("() => document.getElementById('robin-lora-path').disabled")
                assert page.evaluate("() => !document.getElementById('robin-instruct').required")
                # Hints use the shared tooltip on native controls, never a browser title bubble.
                for selector in ('#robin-mode','#robin-language','#robin-instruct','#robin-lora','#robin-lora_strength',
                                 '#robin-lora-strength-range','#robin-temperature','#robin-top_k','#robin-top_p',
                                 '#robin-cfg','#robin-steps','#robin-sampler','#robin-as_written','#robin-reference'):
                    assert page.locator(selector).get_attribute('title') is None, selector
                assert page.locator('#robin-temperature').get_attribute('data-tooltip')
                assert page.locator('#robin-lora option[value=default]').get_attribute('data-tooltip')
                page.locator('#text').fill("Okay, okay, [laughs] I really didn't think you'd do it!")
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===9")
                assert (requests[-1]['temperature'],requests[-1]['top_k'],requests[-1]['top_p'],requests[-1]['cfg'],
                        requests[-1]['steps']) == (.9,25,.8,1,32)
                assert requests[-1]['sampler']=='dpm_2' and not requests[-1]['as_written']
                assert requests[-1]['lora'] is None and requests[-1]['lora_strength']==1
                assert '48000 Hz' in page.locator('.take details pre').first.text_content()
                assert 'CUDA' in page.locator('.take details pre').first.text_content()
                page.locator('#robin-controls summary').click()
                page.locator('#robin-temperature').fill('1.4')
                page.locator('#robin-top_k').fill('0')
                page.locator('#robin-top_p').fill('0.95')
                page.locator('#robin-cfg').fill('3')
                page.locator('#robin-steps').fill('64')
                page.locator('#robin-sampler').select_option('dpmpp_2m')
                page.locator('#robin-as_written').check()
                page.locator('#seed').fill('7')
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===10")
                assert (requests[-1]['temperature'],requests[-1]['top_k'],requests[-1]['top_p'],requests[-1]['cfg'],
                        requests[-1]['steps'],requests[-1]['sampler'],requests[-1]['as_written'],
                        requests[-1]['seed']) == (1.4,0,0.95,3,64,'dpmpp_2m',True,7)
                # A custom adapter is a typed absolute path; a missing one fails before any take exists.
                page.locator('#robin-lora').select_option('custom')
                assert not page.locator('#robin-lora-custom').is_hidden()
                assert page.evaluate("() => document.getElementById('robin-lora-path').required")
                before_empty = len(requests)
                page.locator('#robin-lora-path').fill('')
                page.locator('#generate').click()
                assert not page.locator('#robin-lora-path').evaluate('(e) => e.checkValidity()')
                assert len(requests) == before_empty, 'An empty custom path must not become a no-adapter request.'
                page.locator('#robin-lora-path').fill(str(root/'missing.safetensors'))
                page.locator('#generate').click()
                page.wait_for_function("() => document.getElementById('status').textContent.includes('Robin voice LoRA is missing')")
                assert page.locator('.take').count() == 10, 'A missing adapter must not create a take.'
                (root/'voice.safetensors').touch()
                page.locator('#robin-lora-path').fill(str(root/'voice.safetensors'))
                page.locator('#robin-lora_strength').fill('1.6')
                assert page.locator('#robin-lora-strength-range').input_value() == '1.6'
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===11")
                assert requests[-1]['lora']==str(root/'voice.safetensors') and requests[-1]['lora_strength']==1.6
                # None opts out of any adapter for this take, whatever Model setup chooses.
                page.locator('#robin-lora').select_option('none')
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===12")
                assert requests[-1]['lora']==''
                # Installing the supplied adapter and saving setup offers it without loading a model,
                # and keeps the custom path the user typed.
                page.locator('#robin-lora').select_option('custom')
                (robin/'robin_character_lora.safetensors').touch()
                started=len(requests)
                page.locator('#model-setup').click()
                page.locator('#setup-robin_lora').fill(adapter)
                # Clear the previous result first, then wait for the refreshed LoRA choices: they are
                # rebuilt from the saved settings and new catalog, so this also proves the save landed.
                page.evaluate("() => document.getElementById('setup-status').textContent = ''")
                page.get_by_role('button',name='Save model locations').click()
                page.wait_for_function("() => [...document.getElementById('robin-lora').options].some(o => o.value === %s)" % json.dumps(adapter))
                page.locator('#close-setup').click()
                assert json.loads((root/'models.json').read_text())['robin_lora'] == adapter
                assert 'Robin · Fixed Seed: files found' in page.locator('#setup-status').inner_text()
                assert len(requests) == started, 'Saving model setup must not start a generation.'
                assert page.locator('#robin-lora').input_value() == 'custom'
                assert page.locator('#robin-lora-path').input_value() == str(root/'voice.safetensors')
                # The setup default sends null; the saved take keeps the adapter it actually used.
                page.locator('#robin-lora').select_option('default')
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===13")
                assert requests[-1]['lora'] is None
                with page.expect_download() as recipe_download:
                    page.get_by_role('button', name='Export recipe').first.click()
                recipe=json.loads(Path(recipe_download.value.path()).read_text())
                assert recipe['app']=='mondovoca' and recipe['model']=='robin' and recipe['lora']==adapter
                page.get_by_role('button', name='Reuse settings').first.click()
                assert page.locator('#robin-lora').input_value() == adapter
                # An empty description is allowed: the adapter alone carries the timbre.
                page.locator('#robin-instruct').fill('')
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===14")
                assert requests[-1]['instruct']=='' and requests[-1]['lora']==adapter
                assert 'No description' in page.locator('.take').first.inner_text()
                page.locator('#robin-mode').select_option('reference')
                assert not page.locator('#robin-clone').is_hidden()
                assert page.evaluate("() => document.getElementById('robin-reference').required")
                assert not page.evaluate("() => document.getElementById('robin-ref_text').required")
                page.locator('#robin-reference-file').set_input_files(dict(name='sample.wav', mimeType='audio/wav', buffer=wav))
                page.wait_for_function("() => document.getElementById('status').textContent.includes('Reference uploaded for Robin')")
                assert page.locator('#robin-reference').input_value()
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===15")
                assert requests[-1]['mode']=='reference' and requests[-1]['reference'] and requests[-1]['ref_text']==''
                # A completed take becomes a Robin reference carrying its script as the optional transcript.
                page.get_by_role('button', name='Use as clone reference').first.click()
                assert page.locator('#robin-mode').input_value()=='reference'
                assert page.locator('#robin-reference').input_value()
                assert page.locator('#robin-ref_text').input_value()=="Okay, okay, [laughs] I really didn't think you'd do it!"
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===16")
                assert requests[-1]['mode']=='reference' and requests[-1]['ref_text'].startswith('Okay')
                previous_reference = page.locator('#robin-reference').input_value()
                page.locator('#robin-reference-file').set_input_files(dict(name='another.wav', mimeType='audio/wav', buffer=wav))
                page.wait_for_function("old => document.getElementById('robin-reference').value !== old", arg=previous_reference)
                assert page.locator('#robin-ref_text').input_value() == '', 'A new upload must not inherit another clip’s transcript.'
                page.locator('#robin-reference').select_option(previous_reference)
                assert page.locator('#robin-ref_text').input_value().startswith('Okay')
                page.locator('#robin-mode').select_option('design')
                page.locator('#robin-language').select_option('ja')
                page.locator('#text').fill('こんにちは')
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===17")
                assert requests[-1]['language']=='ja'
                page.locator('#robin-language').select_option('mix')
                page.locator('#robin-instruct').fill('A cheerful woman in her thirties')
                page.locator('#integration').click()
                params=json.loads(page.locator('#api-params').inner_text())
                assert (params['model'],params['mode'],params['language'],params['lora'],params['response_format']
                        ) == ('robin','design','mix',adapter,'wav')
                page.keyboard.press('Escape')
                page.locator('#preset-name').fill('Robin narrator')
                page.locator('#save-preset').click()
                page.locator('#model').select_option('tiny-voca')
                assert page.locator('#robin-controls').is_hidden()
                assert page.locator('#tiny-controls').is_visible()
                page.locator('#presets').select_option('Robin narrator')
                assert page.locator('#model').input_value()=='robin'
                assert page.locator('#robin-mode').input_value()=='design'
                assert page.locator('#robin-language').input_value()=='mix'
                assert page.locator('#robin-instruct').input_value()=='A cheerful woman in her thirties'
                assert page.locator('#robin-temperature').input_value()=='1.4'
                assert page.locator('#robin-sampler').input_value()=='dpmpp_2m'
                assert page.locator('#robin-as_written').is_checked()
                assert page.locator('#robin-lora').input_value()==adapter
                assert page.locator('#robin-lora_strength').input_value()=='1.6'
                # A changed setup adapter must not silently replace an explicit selection.
                replacement = str(root/'replacement.safetensors')
                Path(replacement).touch()
                before_setup = len(requests)
                page.locator('#model-setup').click()
                page.locator('#setup-robin_lora').fill(replacement)
                page.get_by_role('button', name='Save model locations').click()
                page.wait_for_function("() => [...document.getElementById('robin-lora').options].some(o => o.value === %s)" % json.dumps(replacement))
                page.locator('#close-setup').click()
                assert len(requests) == before_setup
                assert page.locator('#robin-lora').input_value() == 'custom'
                assert page.locator('#robin-lora-path').input_value() == adapter
                assert not page.locator('#robin-lora option').filter(has_text='replacement.safetensors').inner_text().startswith('Robin character')
                page.locator('#presets').select_option('Robin narrator')
                assert page.locator('#robin-lora').input_value() == 'custom'
                assert page.locator('#robin-lora-path').input_value() == adapter
                # Robin's controls stay visible and unclipped in the narrow layout in both themes.
                page.set_viewport_size(dict(width=390,height=844))
                for theme in ('teal','purple'):
                    page.locator('#theme').select_option(theme)
                    assert page.locator('#robin-controls').is_visible()
                    assert page.evaluate("() => document.documentElement.scrollWidth <= innerWidth"), theme
                    assert page.evaluate("() => {const f=document.getElementById('robin-controls');"
                                         "return f.scrollWidth <= f.clientWidth+1;}"), theme
                    page.screenshot(path=str(shots/('robin-%s-mobile.png'%theme)),full_page=True)
                page.set_viewport_size(dict(width=1440,height=1000))
                page.locator('#theme').select_option('teal')
                # Saved-takes settings: browser downloads by default, then a chosen folder keeps history readable.
                page.locator('#output-setup').click()
                page.wait_for_function("() => document.getElementById('output').open")
                assert page.locator('#output-mode').input_value() == 'browser'
                assert page.locator('#output-current').input_value() == 'My browser’s download folder'
                assert page.evaluate("() => document.getElementById('output-dir').readOnly")
                assert page.evaluate("() => typeof showDirectoryPicker === 'function'")
                # A browser folder picker reveals only a name, so the studio resolves it beside the current folder.
                (root/'picked').mkdir()
                resolved = page.request.post(f'http://127.0.0.1:{server.server_port}/api/output-dir', data={'folder_name': 'picked'})
                assert resolved.json()['output_dir'] == str(root/'picked')
                page.locator('#output-mode').select_option('folder')
                assert not page.evaluate("() => document.getElementById('output-dir').readOnly")
                page.locator('#output-dir').fill(str(root/'takes'))
                page.screenshot(path=str(shots/'output-dialog.png'))
                page.locator('#output-save').click()
                page.wait_for_function("() => document.getElementById('output-status').textContent.includes('saved in')")
                page.wait_for_function("() => document.querySelectorAll('.take').length===0")
                page.locator('#close-output').click()
                assert page.locator('#output-location').inner_text() == str(root/'takes')
                assert json.loads((root/'settings.json').read_text())['output_mode'] == 'folder'
                page.locator('#text').fill('A take saved in the chosen folder.')
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===1")
                assert len(list((root/'takes').glob('*.wav'))) == 1
                assert not list((root/'spool').glob('*.wav'))
                # A second output folder keeps takes saved earlier listed and playable.
                page.locator('#output-setup').click()
                page.locator('#output-dir').fill(str(root/'more'))
                page.locator('#output-save').click()
                page.wait_for_function("() => document.getElementById('output-status').textContent.includes(%s)" % json.dumps(str(root/'more')))
                page.locator('#close-output').click()
                page.locator('#text').fill('The newest take.')
                page.locator('#generate').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===2")
                assert len(list((root/'more').glob('*.wav'))) == 1
                assert len(list((root/'takes').glob('*.wav'))) == 1
                earlier_clip = page.locator('.take audio').nth(1).get_attribute('src')
                assert page.evaluate("async src => (await fetch(src)).status", earlier_clip) == 200
                # A folder chosen for the first time can already hold finalized takes from an earlier studio session.
                archive = root/'archive'; archive.mkdir()
                stored_id = uuid4().hex
                stored = dict(id=stored_id, state='done', controls=requests[-1], created=time.time()-3600,
                              seconds=1, elapsed=.01, sample_rate=48000 if requests[-1]['model']=='robin' else 24000,
                              phonemes=[], url='/clips/'+stored_id+'.wav', output_mode='browser')
                (archive/(stored_id+'.json')).write_text(json.dumps(stored, indent=2))
                (archive/(stored_id+'.wav')).write_bytes(wav48 if requests[-1]['model']=='robin' else wav)
                shown_before = page.evaluate("() => [...document.querySelectorAll('.take')].map(card => card.id)")
                assert len(shown_before) == 2
                # A take whose card is relocated has its audio element restarted, so keep the live nodes and watch for moves.
                page.evaluate("""() => {
                  window.__kept = {};
                  for (const card of document.querySelectorAll('.take')) window.__kept[card.id] = [card, card.querySelector('audio')];
                  window.__moved = [];
                  new MutationObserver(records => records.forEach(record => record.removedNodes.forEach(node => {
                    if (node.nodeType === 1 && window.__kept[node.id]) window.__moved.push(node.id);
                  }))).observe(document.getElementById('takes'), {childList: true});
                }""")
                downloads = []
                page.on('download', lambda download: downloads.append(download))
                page.locator('#output-setup').click()
                page.locator('#output-dir').fill(str(archive))
                page.locator('#output-save').click()
                page.wait_for_function("() => document.querySelectorAll('.take').length===3")
                assert not downloads, 'Restoring discovered takes must not download them.'
                assert page.locator('#count').inner_text() == '3 takes'
                shown = page.evaluate("() => [...document.querySelectorAll('.take')].map(card => card.id)")
                assert shown[:2] == shown_before, 'Takes already listed were replaced or duplicated.'
                assert shown[2] == f'take-{stored_id}'
                assert shown.count(f'take-{stored_id}') == 1
                assert page.evaluate("() => window.__moved") == [], 'Takes already listed were moved in the DOM.'
                assert page.evaluate("""() => Object.entries(window.__kept).every(([id, [card, audio]]) =>
                  document.getElementById(id) === card && document.querySelector('#' + id + ' audio') === audio)"""
                ), 'Takes already listed lost their card or audio element.'
                stored_clip = page.locator(f'#take-{stored_id} audio').get_attribute('src')
                assert page.evaluate("async src => (await fetch(src)).status", stored_clip) == 200
                assert page.evaluate("() => outputMode") == 'folder'
                page.locator('#close-output').click()
                page.reload()
                page.wait_for_function("() => document.querySelectorAll('.take').length===3")
                assert page.locator('#output-location').inner_text() == str(archive)
                # Robin takes survive a reload with their own heading, sample rate and settings.
                assert page.locator('.take h3').first.inner_text().startswith('Robin')
                assert '48000 Hz' in page.locator('.take details pre').first.text_content()
                page.screenshot(path=str(shots/'desktop.png'))
                page.set_viewport_size(dict(width=390,height=844))
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                page.screenshot(path=str(shots/'mobile.png'),full_page=True)
                page.reload()
                page.wait_for_function("() => document.querySelectorAll('.take').length===3")
                assert not errors, errors
                browser.close()
            print(json.dumps(dict(result='passed', fixture_jobs=len(requests), screenshots=str(shots))))
        finally:
            server.shutdown(); server.server_close(); thread.join(); studio.close()


if __name__ == '__main__':
    main()
