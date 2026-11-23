/* Mondovoca 0.9.2. Model work stays in isolated local workers. */
'use strict';
const $ = id => document.getElementById(id);
const read = (key, fallback) => { try { return JSON.parse(localStorage.getItem(`mondovoca-${key}`)) ?? fallback; } catch { return fallback; } };
const write = (key, value) => { try { localStorage.setItem(`mondovoca-${key}`, JSON.stringify(value)); } catch { /* Private browsing may disable storage. */ } };
let anchors = [], busy = false, stopped = false, activeJob = null, custom = false, version = '0.9.2';
let models = [], outputMode = 'browser';
let presets = read('presets', {});
if (!presets || typeof presets !== 'object' || Array.isArray(presets)) presets = {};
const status = (text, error=false) => { $('status').textContent = text; $('status').classList.toggle('error', error); };
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
const fishKeys=['temperature','top_p','top_k','max_frames','expert_cache_gb'];
const robinKeys=['temperature','top_k','top_p','cfg','steps'];
function robinChoice() {
  // The select keeps the three API meanings: null = Model setup choice, '' = no adapter, text = explicit path.
  const choice=$('robin-lora').value;
  return choice==='default'?null:choice==='none'?'':choice==='custom'?$('robin-lora-path').value.trim():choice;
}
function controls() {
  if($('model').value==='fish-drama3') {
    const c={model:'fish-drama3',mode:$('fish-mode').value,seed:Number($('seed').value)};
    for(const key of fishKeys)c[key]=Number($('fish-'+key).value);
    if(c.mode==='clone')Object.assign(c,{reference:$('fish-reference').value,ref_text:$('fish-ref_text').value});
    return c;
  }
  if ($('model').value === 'auk') {
    const c={model:'auk',mode:$('auk-mode').value,instruct:$('auk-instruct').value,seed:Number($('seed').value),cpu_offload:$('auk-cpu_offload').checked};
    for(const key of ['gen_seconds','nfe','cfg_strength','sway_sampling_coef']) c[key]=Number($('auk-'+key).value);
    if(c.mode==='clone') c.reference=$('auk-reference').value;
    return c;
  }
  if ($('model').value === 'robin') {
    const c={model:'robin',mode:$('robin-mode').value,language:$('robin-language').value,instruct:$('robin-instruct').value,
      sampler:$('robin-sampler').value,as_written:$('robin-as_written').checked,lora:robinChoice(),
      lora_strength:Number($('robin-lora_strength').value),seed:Number($('seed').value)};
    for(const key of robinKeys) c[key]=Number($('robin-'+key).value);
    if(c.mode==='reference') Object.assign(c,{reference:$('robin-reference').value,ref_text:$('robin-ref_text').value});
    return c;
  }
  if ($('model').value === 'qwen3-tts') {
    const mode=$('qwen-mode').value;
    const value={model:'qwen3-tts',mode,language:$('language').value,seed:Number($('seed').value)};
    for(const key of Object.keys(models.find(m=>m.id==='qwen3-tts').parameters)) value[key]=Number($(key).value);
    for(const key of ['do_sample','subtalker_dosample','non_streaming_mode']) value[key]=$(key).checked;
    if(mode==='custom') value.voice=$('speaker').value;
    if(mode!=='clone') value.instruct=$('instruct').value;
    else Object.assign(value,{reference:$('reference').value,ref_text:$('ref-text').value,x_vector_only_mode:$('embedding-only').checked});
    return value;
  }
  return {model:'tiny-voca',voice:document.querySelector('[name=voice]:checked').value,
    age:$('use-age').checked ? Number($('age').value) : null,
    speed:Number($('speed').value), semitones:Number($('pitch').value),
    pitch_range:Number($('movement').value), seed:Number($('seed').value)};
}
function labels() {
  $('speed-value').textContent = `${Number($('speed').value).toFixed(2)}×`;
  $('pitch-value').textContent = `${Number($('pitch').value).toFixed(1)} st`;
  $('movement-value').textContent = `${Number($('movement').value).toFixed(2)}×`;
  $('tuning-note').textContent = `${$('use-age').checked ? `Age ${$('age').value} preset` : 'Original voice'}${custom ? ' · custom tuning' : ''}`;
}
function tune(values) {
  $('speed').value = values.speed; $('pitch').value = values.semitones; $('movement').value = values.pitch_range; labels();
}
function agePreset() {
  const age = Number($('age').value);
  $('quick-age').disabled = $('age-range').disabled = $('age').disabled = !$('use-age').checked;
  if (!$('age').checkValidity() || !anchors.length) return;
  $('age-range').value = age;
  $('quick-age').value = age % 5 === 0 ? String(age) : '';
  $('quick-age').disabled = $('age-range').disabled = $('age').disabled = !$('use-age').checked;
  let values = {speed:1,semitones:0,pitch_range:1};
  if ($('use-age').checked) {
    for (let i=1;i<anchors.length;i++) if (age <= anchors[i][0]) {
      const a=anchors[i-1], b=anchors[i], t=(age-a[0])/(b[0]-a[0]);
      const result=a.slice(1).map((v,j)=>v+(b[j+1]-v)*t);
      values={speed:result[0],semitones:result[1],pitch_range:result[2]}; break;
    }
  }
  custom=false; $('delivery').value='natural'; tune(values);
}
function restore(value) {
  if(value?.model==='fish-drama3') {
    $('model').value=value.model;$('fish-mode').value=value.mode;$('seed').value=value.seed;
    for(const key of fishKeys)$('fish-'+key).value=value[key];
    if(value.reference){ensureReference(value.reference,'Saved reference');$('fish-reference').value=value.reference;}
    $('fish-ref_text').value=value.ref_text || '';modelChanged();return;
  }
  if(value?.model==='auk') {
    $('model').value='auk';$('auk-mode').value=value.mode;
    $('auk-instruct').value=value.instruct || '';$('seed').value=value.seed;
    $('auk-cpu_offload').checked=!!value.cpu_offload;
    for(const key of ['gen_seconds','nfe','cfg_strength','sway_sampling_coef']) $('auk-'+key).value=value[key];
    if(value.reference){ensureReference(value.reference,'Saved reference');$('auk-reference').value=value.reference;}
    modelChanged();return;
  }
  if(value?.model==='robin') {
    $('model').value='robin';modelChanged();
    $('robin-mode').value=value.mode;$('robin-language').value=value.language;$('seed').value=value.seed;
    $('robin-instruct').value=value.instruct || '';
    for(const key of robinKeys) $('robin-'+key).value=value[key];
    $('robin-sampler').value=value.sampler || 'dpm_2';
    $('robin-as_written').checked=!!value.as_written;
    $('robin-lora_strength').value=value.lora_strength ?? 1; robinStrength();
    // A saved take records the adapter it used, so restore the matching choice rather than the current setup default.
    setRobinLora(value.lora);
    if(value.reference){ensureReference(value.reference,'Saved reference');$('robin-reference').value=value.reference;}
    $('robin-ref_text').value=value.ref_text || '';
    robinModeChanged();return;
  }
  if(value?.model==='qwen3-tts') {
    $('model').value=value.model;$('qwen-mode').value=value.mode;modelChanged();
    $('language').value=value.language;$('seed').value=value.seed;
    if(value.voice) $('speaker').value=value.voice;
    $('instruct').value=value.instruct || '';
    if(value.reference){ensureReference(value.reference,'Saved reference');$('reference').value=value.reference;}
    $('ref-text').value=value.ref_text || '';$('embedding-only').checked=!!value.x_vector_only_mode;
    for(const key of Object.keys(models.find(m=>m.id==='qwen3-tts').parameters)) $(key).value=value[key];
    for(const key of ['do_sample','subtalker_dosample','non_streaming_mode']) $(key).checked=value[key];
    modeChanged();return;
  }
  if (!value || !['F2','M2'].includes(value.voice)) throw new Error('This is not a Mondovoca voice preset');
  $('model').value='tiny-voca';modelChanged();
  document.querySelector(`[name=voice][value=${value.voice}]`).checked=true;
  $('use-age').checked=value.age !== null;
  $('age').value=value.age ?? 30; agePreset();
  $('seed').value=value.seed ?? 0; custom=true; tune(value);
}
function listPresets() {
  $('presets').replaceChildren(new Option('Choose a saved voice…',''));
  for (const name of Object.keys(presets)) $('presets').add(new Option(name,name));
}
function download(data, name) {
  const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  const a=document.createElement('a'); a.href=url; a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function silence() { document.querySelectorAll('audio').forEach(audio=>audio.pause()); }
function setBusy(value) {
  busy=value; $('generate').disabled=$('compare').disabled=value;
  $('model').disabled=value; $('compare').disabled=value || $('model').value!=='tiny-voca';
  $('stop').disabled=!value; $('light').classList.toggle('busy',value);
}
function node(tag, text, className) {
  const element=document.createElement(tag); if(text !== undefined) element.textContent=text;
  if(className) element.className=className; return element;
}
async function waveform(canvas, url) {
  let context;
  try {
    const AudioContext=window.AudioContext || window.webkitAudioContext;
    if(!AudioContext) return;
    context=new AudioContext();
    const response=await fetch(url); const buffer=await context.decodeAudioData(await response.arrayBuffer());
    const samples=buffer.getChannelData(0), ctx=canvas.getContext('2d');
    canvas.width=800;canvas.height=60;ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
    ctx.beginPath();
    for(let x=0;x<800;x++) {
      const start=Math.floor(x*samples.length/800), end=Math.max(start+1,Math.floor((x+1)*samples.length/800));
      let peak=0;for(let i=start;i<end;i++) peak=Math.max(peak,Math.abs(samples[i] || 0));
      ctx.moveTo(x,30-peak*28);ctx.lineTo(x,30+peak*28);
    }
    ctx.stroke();
  } catch { canvas.hidden=true; } finally { if(context) await context.close(); }
}
function addTake(job, autoplay=false, fresh=false) {
  if(document.getElementById(`take-${job.id}`)) return;
  if(job.output_mode) outputMode=job.output_mode;
  const c=job.controls, qwen=c.model==='qwen3-tts', auk=c.model==='auk', fish=c.model==='fish-drama3', robin=c.model==='robin', cuda=qwen||auk||fish||robin, card=node('article',undefined,'take');card.id=`take-${job.id}`;
  const header=node('header');header.append(node('h3',robin?`Robin · ${c.mode}`:fish?`Fish Audio · Drama 3 · ${c.mode}`:auk?`AuK · ${c.mode}`:qwen?`Qwen3-TTS · ${c.mode} · ${c.voice || c.language}`:`${c.voice==='F2'?'Female':'Male'} · ${c.voice}`),node('span',`${job.seconds.toFixed(2)}s`,'badge'));card.append(header);
  const language=models.find(m=>m.id==='robin')?.language_names?.[c.language] || c.language;
  const description=robin?`${c.instruct || 'No description'} · ${language} · ${c.lora?`LoRA ${c.lora.split('/').pop()}`:'no voice LoRA'}`:fish?`${c.mode==='clone'?'Reference clone':'Generated voice'} · ${c.expert_cache_gb} GiB expert cache`:auk?`${c.instruct || 'Reference voice'} · ${c.gen_seconds}s target · ${c.nfe} steps`:qwen?`${c.language} · ${c.instruct || (c.x_vector_only_mode?'Speaker embedding clone':'Transcript-guided clone')}`:`${c.age === null?'Original voice':`Age ${c.age} styling`} · ${c.speed.toFixed(2)}× speed · ${c.semitones.toFixed(1)} st · ${c.pitch_range.toFixed(2)}× movement`;
  card.append(node('p',description,'hint'),node('p',c.text,'script-preview'));
  const canvas=node('canvas');canvas.setAttribute('aria-label','Generated audio waveform');canvas.setAttribute('role','img');card.append(canvas);
  const audio=node('audio');audio.controls=true;audio.preload='metadata';audio.volume=0.4;audio.src=job.url;
  audio.addEventListener('play',()=>document.querySelectorAll('audio').forEach(other=>{if(other!==audio)other.pause();}));card.append(audio);
  const tools=node('div',undefined,'tools'), wav=node('a','Download WAV','download mini');wav.href=job.url;wav.download=`mondovoca-${c.voice || c.mode}-${job.id}.wav`;
  const reuse=node('button','Reuse settings','mini');reuse.type='button';reuse.onclick=()=>{restore(c);$('text').value=c.text;characters();status('Take settings restored.');};
  const recipe=node('button','Export recipe','mini');recipe.type='button';recipe.onclick=()=>download({app:'mondovoca',version,...c},`mondovoca-${job.id}.json`);
  const clone=node('button','Use as clone reference','mini');clone.type='button';clone.onclick=()=>{
    if($('model').value==='robin') {
      $('robin-mode').value='reference';robinModeChanged();
      ensureReference(job.id,`Take · ${c.text.slice(0,40)}`);
      $('robin-reference').value=job.id;$('robin-ref_text').value=c.text;
      status('Robin reference selected. Enter a new script.');return;
    }
    if($('model').value==='fish-drama3') {
      $('fish-mode').value='clone';ensureReference(job.id,`Take · ${c.text.slice(0,40)}`);
      $('fish-reference').value=job.id;$('fish-ref_text').value=c.text;modelChanged();
      status('Fish reference selected. Enter a new script.');return;
    }
    if($('model').value==='auk') {
      $('auk-mode').value='clone';modelChanged();
      ensureReference(job.id,`Take · ${c.text.slice(0,40)}`);$('auk-reference').value=job.id;
      status('Reference selected. Enter a new script to clone this voice.');return;
    }
    $('model').value='qwen3-tts';$('qwen-mode').value='clone';$('non_streaming_mode').checked=false;modelChanged();
    ensureReference(job.id,`Take · ${c.voice || c.mode} · ${c.text.slice(0,40)}`);
    $('reference').value=job.id;$('ref-text').value=c.text;$('embedding-only').checked=false;modeChanged();
    status('Reference selected. Enter a new script to clone this voice.');
  };
  ensureReference(job.id,`Take · ${c.voice || c.mode} · ${c.text.slice(0,40)}`);
  tools.append(wav,reuse,recipe,clone);card.append(tools);
  const details=node('details'), summary=node('summary','Synthesis details');
  details.append(summary,node('pre',`${new Date(job.created*1000).toLocaleString()}\n${cuda?'CUDA':'CPU'} · ${job.sample_rate} Hz · seed ${c.seed}\nGenerated in ${job.elapsed.toFixed(2)}s\n\n${cuda?JSON.stringify(c,null,2):(job.phonemes || []).join('\n')}`));card.append(details);
  $('takes').prepend(card);$('empty').hidden=true;
  const count=$('takes').children.length;$('count').textContent=`${count} ${count===1?'take':'takes'}`;
  waveform(canvas,job.url);
  if(fresh && job.output_mode==='browser') { const save=node('a'); save.href=job.url; save.download=wav.download; save.click(); }
  if(autoplay){silence();audio.play().catch(()=>status('Take ready. Press play to listen.'));}
}
async function run(compare=false) {
  if(busy) return;
  if(!$('voice-form').reportValidity()) return;
  const text=$('text').value.trim();if(!text){status('Enter a script first.',true);$('text').focus();return;}
  const snapshot={...controls(),text}, voices=compare?['F2','M2']:[snapshot.voice];
  stopped=false;setBusy(true);silence();
  try {
    for(const voice of voices) {
      if(stopped) break;
      status(`Generating ${snapshot.model==='fish-drama3'?'Fish Drama 3 with experts in system RAM':snapshot.model==='auk'?`AuK ${snapshot.mode} on CUDA`:snapshot.model==='robin'?`Robin ${snapshot.mode} on CUDA`:snapshot.model==='qwen3-tts'?`Qwen3-TTS ${snapshot.mode} on CUDA`: `${voice} on CPU`}…`);
      let job=await api('/api/jobs',voice?{...snapshot,voice}:snapshot);activeJob=job.id;
      if(stopped) job=await api(`/api/jobs/${job.id}/cancel`,{});
      while(job.state==='running') {
        await new Promise(resolve=>setTimeout(resolve,300));
        job=await api(`/api/jobs/${job.id}`);
      }
      if(job.state==='done') addTake(job,$('autoplay').checked && !stopped && !compare,true);
      else if(job.state==='error') throw new Error(job.error);
      activeJob=null;
    }
    status(stopped?'Stopped.':compare?'Both voices are ready. Play each take to compare.':outputMode==='browser'?'Take ready. It downloads to your browser’s download folder.':'Take ready.');
  } catch(error) {status(error.message,true);} finally {activeJob=null;setBusy(false);}
}
function characters() {$('characters').textContent=`${$('text').value.length.toLocaleString()} / 5,000`;}
for(let age=5;age<=100;age+=5) $('quick-age').add(new Option(`${age} years`,age));
$('quick-age').add(new Option('Custom',''),0);
$('age').oninput=agePreset;$('age-range').oninput=()=>{$('age').value=$('age-range').value;agePreset();};
$('quick-age').onchange=()=>{if($('quick-age').value){$('age').value=$('quick-age').value;agePreset();}};
$('use-age').onchange=agePreset;$('reset').onclick=()=>{$('seed').value=0;agePreset();};
['speed','pitch','movement'].forEach(id=>$(id).oninput=()=>{custom=true;labels();});
$('delivery').onchange=()=>{
  const style=$('delivery').value;agePreset();$('delivery').value=style;
  const c=controls(), factors={natural:[1,0,1],narration:[.92,-.3,.95],lively:[1.08,.7,1.15],calm:[.88,-.5,.82]}[style];
  custom=style!=='natural';tune({speed:c.speed*factors[0],semitones:c.semitones+factors[1],pitch_range:c.pitch_range*factors[2]});
};
$('save-preset').onclick=()=>{const name=$('preset-name').value.trim();if(!name){status('Give your voice preset a name.',true);return;}
  if(!$('voice-form').reportValidity())return;presets[name]=controls();write('presets',presets);listPresets();$('presets').value=name;status(`Saved voice “${name}” in this browser.`);};
$('presets').onchange=()=>{if($('presets').value)restore(presets[$('presets').value]);};listPresets();
$('theme').value=read('theme','teal');document.documentElement.dataset.theme=$('theme').value || 'teal';
$('theme').onchange=()=>{document.documentElement.dataset.theme=$('theme').value;write('theme',$('theme').value);};
$('text').oninput=characters;characters();
let example=0;const examples=['Welcome to the workshop. Today, we turn a small idea into something you can hear.','Wait a moment. Did you hear that? I think our next adventure has already begun.','The evening settled softly over the hills. In the distance, a single window glowed.'];
$('example').onclick=()=>{$('text').value=examples[example++%examples.length];characters();};
$('text-file').onchange=async()=>{const file=$('text-file').files[0];if(!file)return;
  if(file.size>30000){status('Choose a text file containing at most 5,000 characters.',true);return;}
  try{const text=await file.text();if(text.length>5000)throw new Error('The text file exceeds 5,000 characters.');$('text').value=text;characters();status('Text imported.');}catch(error){status(error.message,true);}finally{$('text-file').value='';}};
$('generate').onclick=()=>run();$('compare').onclick=()=>run(true);
$('stop').onclick=async()=>{stopped=true;silence();$('stop').disabled=true;status('Stopping…');try{if(activeJob)await api(`/api/jobs/${activeJob}/cancel`,{});}catch(error){status(error.message,true);}};
$('voice-form').onsubmit=event=>event.preventDefault();
document.addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key==='Enter'){event.preventDefault();run();}});
const dialog=$('connection');$('integration').onclick=()=>{
  $('api-base').textContent=`${location.origin}/v1`;
  const c=controls();$('api-model').textContent=c.model;$('api-voice').textContent=c.voice || c.mode;
  $('api-params').textContent=JSON.stringify({...c,response_format:'wav'},null,2);
  dialog.showModal();};$('close-connection').onclick=()=>dialog.close();
let backdrop=false;dialog.addEventListener('pointerdown',e=>{const r=dialog.getBoundingClientRect();backdrop=e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom;});
dialog.addEventListener('click',e=>{const r=dialog.getBoundingClientRect();if(backdrop&&(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom))dialog.close();backdrop=false;});
$('copy-connection').onclick=async()=>{try{await navigator.clipboard.writeText($('api-params').textContent);$('copy-connection').textContent='Copied';setTimeout(()=>$('copy-connection').textContent='Copy parameters',1500);}catch{status('Select and copy the parameters from the connection dialog.');}};
async function init() {
  $('generate').disabled=$('compare').disabled=true;
  try {
    const [config,health,jobs]=await Promise.all([api('/api/config'),api('/api/health'),api('/api/jobs')]);
    models=config.models;setupModels();anchors=config.age_anchors;version=health.version;$('version').textContent=version;agePreset();
    outputMode=config.output_mode;
    $('output-location').textContent=$('output-location').title=outputLabel(config);
    [...jobs].reverse().filter(job=>job.state==='done').forEach(job=>addTake(job));
    if(!health.ready)status('Model files are missing. See docs/MONDOVOCA.md.',true);
    const running=jobs.find(job=>job.state==='running');
    if(running){activeJob=running.id;stopped=false;setBusy(true);status('A previous generation is still running…');
      let job=running;while(job.state==='running'){await new Promise(resolve=>setTimeout(resolve,300));job=await api(`/api/jobs/${job.id}`);}
      if(job.state==='done')addTake(job,false,true);status(job.state==='error'?job.error:job.state==='cancelled'?'Stopped.':'Take ready.',job.state==='error');activeJob=null;setBusy(false);}
  } catch(error){status(`Cannot connect to Mondovoca: ${error.message}`,true);} finally {if(!busy){$('generate').disabled=false;$('compare').disabled=$('model').value!=='tiny-voca';}}
}
init();

function ensureReference(id, label) {
  for(const key of ['reference','auk-reference','fish-reference','robin-reference']) if(![...$(key).options].some(o=>o.value===id)) $(key).add(new Option(label,id));
}
function setupModels() {
  $('model').replaceChildren(...models.map(m=>new Option(m.name,m.id)));
  const qwen=models.find(m=>m.id==='qwen3-tts');
  $('language').replaceChildren(...qwen.languages.map(l=>new Option(l,l)));
  $('language').value=qwen.default_language;
  $('speaker').replaceChildren(...Object.entries(qwen.speakers).map(([id,label])=>new Option(label,id)));
  $('speaker').value='ryan';
  const robin=models.find(m=>m.id==='robin');
  $('robin-language').replaceChildren(...Object.entries(robin.language_names).map(([code,label])=>new Option(label,code)));
  $('robin-language').value=robin.languages[0];
  for(const [key,[value,min,max,integer]] of Object.entries(qwen.parameters)) {
    const label=node('label',key.replaceAll('_',' ')), input=node('input');
    input.id=key;input.type='number';input.min=min;input.max=max;input.step=integer?'1':'0.01';input.value=value;input.required=true;
    label.append(input);$('qwen-parameters').append(label);
  }
  modelChanged();
}
function robinLoras() {
  // Rebuilt whenever the catalog changes so a Model setup save offers the adapter it just found,
  // while the previous choice and any typed custom path survive the refresh.
  const previous=$('robin-lora').value || 'default', adapter=models.find(m=>m.id==='robin')?.voice_lora;
  const options=[new Option('Model setup default','default'),new Option('None (base voice)','none')];
  if(adapter) options.push(new Option(`${adapter.endsWith('/robin_character_lora.safetensors')?'Robin character':'Voice LoRA'} · ${adapter.split('/').pop()}`,adapter));
  options.push(new Option('Custom path…','custom'));
  options[0].dataset.tooltip='Use the voice LoRA chosen in Model setup, or no adapter when none is chosen there.';
  options[1].dataset.tooltip='Generate this take with the base voice and no adapter at all.';
  if(adapter) options[2].dataset.tooltip=adapter;
  options[options.length-1].dataset.tooltip='Type the absolute .safetensors path of another adapter on this computer.';
  $('robin-lora').replaceChildren(...options);
  if([...$('robin-lora').options].some(o=>o.value===previous)) $('robin-lora').value=previous;
  else { $('robin-lora-path').value=previous; $('robin-lora').value='custom'; }
  robinLoraChanged();
}
function robinLoraChanged() {
  const typed=$('robin-lora').value==='custom';
  $('robin-lora-custom').hidden=!typed;
  $('robin-lora-path').disabled=!typed;
  $('robin-lora-path').required=typed;
}
function setRobinLora(value) {
  const select=$('robin-lora');
  if(value===null || value===undefined) select.value='default';
  else if(value==='') select.value='none';
  else if([...select.options].some(option=>option.value===value)) select.value=value;
  else { $('robin-lora-path').value=value; select.value='custom'; }
  robinLoraChanged();
}
function robinStrength() {
  const value=Number($('robin-lora_strength').value);
  if($('robin-lora_strength').checkValidity()) $('robin-lora-strength-range').value=value;
}
function robinModeChanged() {
  const active=$('model').value==='robin', reference=$('robin-mode').value==='reference';
  $('robin-clone').hidden=!reference;
  $('robin-reference').required=reference;
  $('robin-reference').disabled=!active || !reference;
  $('robin-reference-file').disabled=!active || !reference;
  $('robin-ref_text').disabled=!active || !reference;
}
function modelChanged() {
  const qwen=$('model').value==='qwen3-tts', auk=$('model').value==='auk', fish=$('model').value==='fish-drama3', robin=$('model').value==='robin', advanced=qwen || auk || fish || robin;
  $('tiny-controls').hidden=$('tiny-controls').disabled=advanced;
  $('auk-controls').hidden=$('auk-controls').disabled=!auk;
  aukModeChanged();
  $('fish-controls').hidden=$('fish-controls').disabled=!fish;fishModeChanged();
  $('qwen-controls').hidden=$('qwen-controls').disabled=!qwen;
  $('robin-controls').hidden=$('robin-controls').disabled=!robin;
  if(robin) { robinLoras(); robinModeChanged(); }
  $('reset').hidden=advanced;$('compare').hidden=advanced;$('compare').disabled=busy || advanced;
  const model=models.find(m=>m.id===$('model').value);
  $('model-note').textContent=fish?'Fish Drama 3 INT8 · CUDA with RAM-backed experts · inline directions and voice cloning.':auk?'AuK Base · CUDA · voice descriptions and reference cloning.':robin?'Robin · CUDA · voice descriptions, an optional voice LoRA and reference conditioning. Loads only when you generate.':qwen?'1.7B · CUDA · ten languages. Loads only when you generate.':'Two English voices · CPU';
  $('engine-footer').textContent=fish?'Fish Audio · Drama 3 · CUDA + RAM':auk?'AuK · Base · CUDA':robin?'Robin · Fixed Seed · CUDA':qwen?'Qwen3-TTS · CUDA · multilingual':'Tiny Voca · CPU · English';
  if(qwen) modeChanged();
  if(model && !model.ready) status(model.missing?.length ? 'Missing: '+model.missing.join(' · ') : 'This model needs local files or its Python environment. See README.md.',true);
}
function modeChanged() {
  const mode=$('qwen-mode').value, clone=mode==='clone', custom=mode==='custom';
  $('speaker-field').hidden=!custom;$('speaker').disabled=!custom || $('qwen-controls').disabled;
  $('instruct-field').hidden=clone;$('instruct').disabled=clone || $('qwen-controls').disabled;$('instruct').required=mode==='design';
  $('clone-fields').hidden=!clone;
  $('ref-text').disabled=!clone || $('embedding-only').checked || $('qwen-controls').disabled;
  $('reference').required=clone;$('reference').disabled=!clone || $('qwen-controls').disabled;
  $('ref-text').required=clone && !$('embedding-only').checked;
  const installed=models.find(m=>m.id==='qwen3-tts')?.modes[mode];
  $('mode-note').textContent=installed?(clone?'Base cloning follows the reference voice. Style instructions apply to preset and design modes.':custom?'Keep a preset timbre and direct its emotion, accent and delivery in words.':'Describe a new voice in words. Use the resulting take as a clone reference to keep its identity.'):'Local checkpoint or Qwen Python is missing for this mode. See README.md.';
}
$('model').onchange=modelChanged;$('qwen-mode').onchange=()=>{$('non_streaming_mode').checked=$('qwen-mode').value!=='clone';modeChanged();};$('embedding-only').onchange=modeChanged;
$('reference').onchange=()=>{
  const take=document.getElementById(`take-${$('reference').value}`);
  if(take) $('ref-text').value=take.querySelector('.script-preview').textContent;
};
$('reference-file').onchange=async()=>{
  const file=$('reference-file').files[0];if(!file)return;
  if(file.size>8*1024*1024){status('Choose a WAV smaller than 8 MB.',true);$('reference-file').value='';return;}
  try {
    const audio=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=reject;reader.readAsDataURL(file);});
    const result=await api('/api/references',{audio});ensureReference(result.reference,`${file.name} · ${result.seconds.toFixed(1)}s`);
    $('reference').value=result.reference;$('ref-text').value='';status('Reference uploaded. Enter its exact transcript or choose embedding-only mode.');
  } catch(error){status(error.message,true);} finally {$('reference-file').value='';}
};

const setupKeys=['tiny_python','tiny_checkpoint','tiny_phonemizer','qwen_python','qwen_root','auk_python','auk_root','auk_encoder','fish_python','fish_root','fish_runtime','fish_comfy','robin_python','robin_root','robin_lora'];
$('model-setup').onclick=async()=>{
  try {
    const result=await api('/api/model-settings');
    for(const key of setupKeys){$('setup-'+key).value=result.settings[key];$('setup-'+key).readOnly=result.overrides.includes(key);}
    $('setup-status').textContent=result.overrides.length?'Some paths are set by environment variables and are read-only here.':'Paths are saved privately on this machine. Missing files appear as unavailable until installed.';
    $('setup').showModal();
  } catch(error){status(error.message,true);}
};
$('close-setup').onclick=()=>$('setup').close();
$('setup-form').onsubmit=async event=>{
  event.preventDefault();
  try {
    const values=Object.fromEntries(setupKeys.map(key=>[key,$('setup-'+key).value.trim()]));
    const result=await api('/api/model-settings',values);models=result.models;
    $('setup-status').textContent=models.map(m=>`${m.name}: ${m.ready?'files found':'files or environment missing'}`).join(' · ');
    modelChanged();status('Model locations saved. No models were loaded.');
  } catch(error){$('setup-status').textContent=error.message;}
};

let outputSettings = null;
const outputLabel = state => state.output_mode === 'browser' ? 'My browser’s download folder' : state.output_dir;
function outputEditable() {
  const folder = $('output-mode').value === 'folder';
  $('output-dir').readOnly = $('output-choose').disabled = !folder || !!outputSettings?.locked;
}
function showOutput(state) {
  outputSettings = state; outputMode = state.output_mode;
  $('output-location').textContent=$('output-location').title=outputLabel(state);
  $('output-current').value=outputLabel(state);
  $('output-mode').value=state.output_mode;
  $('output-mode').disabled=state.locked;
  $('output-dir').value=state.output_dir || '';
  $('output-save').hidden=state.locked;
  outputEditable();
  $('output-note').textContent=(state.locked?`This server fixes the save location with ${state.lock_source}. `:'')
    +'Your browser’s own download folder and save-dialog settings decide where downloaded takes land.'
    +(state.previous_output_dirs.length?` Takes saved earlier remain in ${state.previous_output_dirs.join(', ')}.`:'');
}
$('output-setup').onclick=async()=>{
  try { $('output-status').textContent=''; showOutput(await api('/api/output-dir')); $('output').showModal(); }
  catch(error){ status(error.message,true); }
};
$('close-output').onclick=()=>$('output').close();
$('output-mode').onchange=()=>{
  if($('output-mode').value==='folder' && !$('output-dir').value) $('output-dir').value=outputSettings?.output_dir || '';
  outputEditable();
};
$('output-choose').onclick=async()=>{
  if(!window.showDirectoryPicker) { $('output-status').textContent='This browser has no folder picker. Enter the full path above.'; return; }
  try {
    const picked = await window.showDirectoryPicker({id:'mondovoca-output'});
    const found = await api('/api/output-dir',{folder_name:picked.name});
    $('output-dir').value=found.output_dir; $('output-status').textContent='';
  } catch(error) { if(error.name!=='AbortError') $('output-status').textContent=error.message; }
};
$('output-form').onsubmit=async event=>{
  event.preventDefault();
  try {
    const mode=$('output-mode').value;
    showOutput(await api('/api/output-dir',{output_mode:mode,output_dir:mode==='folder'?$('output-dir').value.trim():''}));
    const jobs=await api('/api/jobs');
    const live=new Set(jobs.map(job=>job.id));
    // Takes that no longer have a file on this machine leave the list; their copies live in the downloads folder.
    document.querySelectorAll('.take').forEach(card=>{ if(!live.has(card.id.slice(5))) card.remove(); });
    // Takes the server found in a folder this page never opened get cards at once: no autoplay, no new download.
    const completed=jobs.filter(job=>job.state==='done');
    const found=new Set(completed.filter(job=>!document.getElementById(`take-${job.id}`)).map(job=>job.id));
    completed.forEach(job=>addTake(job));
    outputMode=outputSettings.output_mode;
    if(found.size) {
      // Only new cards are relocated into newest-first order; moving a listed take would restart its audio element.
      for(let i=completed.length-1;i>=0;i--) {
        if(!found.has(completed[i].id)) continue;
        const next=i+1<completed.length?document.getElementById(`take-${completed[i+1].id}`):null;
        $('takes').insertBefore(document.getElementById(`take-${completed[i].id}`),next);
      }
    }
    const shown=$('takes').children.length;
    $('count').textContent=`${shown} ${shown===1?'take':'takes'}`;
    $('empty').hidden=shown>0;
    $('output-status').textContent=mode==='browser'?'Takes will download to your browser’s download folder.'
      :`Takes will be saved in ${outputSettings.output_dir}.`;
  } catch(error){ $('output-status').textContent=error.message; }
};

function aukModeChanged() {
  const active=$('model').value==='auk', clone=$('auk-mode').value==='clone';
  $('auk-clone').hidden=!clone;
  $('auk-instruct').required=!clone;
  $('auk-reference').required=clone;
  $('auk-reference').disabled=!active || !clone;
  $('auk-reference-file').disabled=!active || !clone;
}
$('auk-mode').onchange=aukModeChanged;
$('auk-reference-file').onchange=async()=>{
  const file=$('auk-reference-file').files[0];if(!file)return;
  try {
    if(file.size>8*1024*1024)throw new Error('Choose a WAV smaller than 8 MB.');
    const audio=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=reject;reader.readAsDataURL(file);});
    const result=await api('/api/references',{audio});ensureReference(result.reference,`${file.name} · ${result.seconds.toFixed(1)}s`);
    $('auk-reference').value=result.reference;status('Reference uploaded for AuK.');
  } catch(error){status(error.message,true);} finally {$('auk-reference-file').value='';}
};

function fishModeChanged() {
  const active=$('model').value==='fish-drama3', clone=$('fish-mode').value==='clone';
  $('fish-clone').hidden=!clone;
  for(const key of ['fish-reference','fish-reference-file','fish-ref_text'])$(key).disabled=!active || !clone;
  $('fish-reference').required=clone;$('fish-ref_text').required=clone;
}
$('fish-mode').onchange=fishModeChanged;
$('fish-reference-file').onchange=async()=>{
  const file=$('fish-reference-file').files[0];if(!file)return;
  try {
    if(file.size>8*1024*1024)throw new Error('Choose a WAV smaller than 8 MB.');
    const audio=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=reject;reader.readAsDataURL(file);});
    const result=await api('/api/references',{audio});ensureReference(result.reference,`${file.name} · ${result.seconds.toFixed(1)}s`);
    $('fish-reference').value=result.reference;status('Reference uploaded. Enter its exact transcript.');
  } catch(error){status(error.message,true);} finally {$('fish-reference-file').value='';}
};

$('robin-mode').onchange=robinModeChanged;
$('robin-reference').onchange=()=>{
  const take=document.getElementById(`take-${$('robin-reference').value}`);
  $('robin-ref_text').value=take?.querySelector('.script-preview').textContent || '';
};
$('robin-lora').onchange=robinLoraChanged;
$('robin-lora_strength').oninput=robinStrength;
$('robin-lora-strength-range').oninput=()=>{$('robin-lora_strength').value=$('robin-lora-strength-range').value;robinStrength();};
$('robin-reference-file').onchange=async()=>{
  const file=$('robin-reference-file').files[0];if(!file)return;
  try {
    if(file.size>8*1024*1024)throw new Error('Choose a WAV smaller than 8 MB.');
    const audio=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=reject;reader.readAsDataURL(file);});
    const result=await api('/api/references',{audio});ensureReference(result.reference,`${file.name} · ${result.seconds.toFixed(1)}s`);
    $('robin-reference').value=result.reference;$('robin-ref_text').value='';status('Reference uploaded for Robin. Its transcript is optional.');
  } catch(error){status(error.message,true);} finally {$('robin-reference-file').value='';}
};
