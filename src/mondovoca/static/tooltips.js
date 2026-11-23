/* Shared Mondocraft tooltips. data-tooltip keeps browser title bubbles out of both documents.
   Use the result prompt popup's appearance; no wrappers or control layout changes. */
(() => {
  'use strict';
  const style = document.createElement('style');
  style.textContent = `
    .mc-tooltip{position:fixed;z-index:2147483647;width:max-content;max-width:min(600px,calc(100vw - 32px));
      max-height:60vh;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;box-sizing:border-box;
      background:var(--panel2,#182d34);color:var(--text,#e7f1f3);border:1px solid var(--line,#2b4852);
      border-radius:6px;padding:12px;box-shadow:0 8px 28px #000c;font:400 14px/1.5 system-ui,sans-serif;
      text-align:start;letter-spacing:normal}
    .mc-tooltip[hidden]{display:none}
  `;
  document.head.append(style);
  const tip = document.createElement('div');
  tip.className = 'mc-tooltip'; tip.id = 'mc-tooltip'; tip.setAttribute('role', 'tooltip'); tip.hidden = true;
  let owner = null, hovered = null, dismissed = null, described = [], showTimer, hideTimer, expiryTimer, keyboard = false, dragging = false;
  const target = el => el instanceof Element ? el.closest('[data-tooltip],select') : null;
  const message = el => el?.dataset.tooltip || (el?.matches('select') ? el.selectedOptions[0]?.dataset.tooltip : '') || '';
  function hide() {
    clearTimeout(showTimer); clearTimeout(hideTimer); clearTimeout(expiryTimer);
    for (const el of described) {
      const ids = (el.getAttribute('aria-describedby') || '').split(/\s+/).filter(id => id && id !== tip.id);
      if (ids.length) el.setAttribute('aria-describedby', ids.join(' ')); else el.removeAttribute('aria-describedby');
    }
    described = []; owner = null; tip.hidden = true;
  }
  function dismiss() { dismissed = owner || hovered; hide(); }
  function show(el) {
    clearTimeout(showTimer); clearTimeout(hideTimer);
    if (!el || dragging || el === dismissed || !el.isConnected || !el.getClientRects().length || !message(el)) { hide(); return; }
    const fresh = owner !== el;
    if (fresh) hide();
    owner = el;
    // A native modal dialog sits above body content. Put its tooltip in that same layer.
    const host = el.closest('dialog[open]') || document.body;
    if (tip.parentNode !== host) host.append(tip);
    tip.textContent = message(el); tip.dir = el.closest('[dir]')?.dir || document.documentElement.dir || 'auto';
    described = [...new Set([owner, ...(owner.contains(document.activeElement) ? [document.activeElement] : [])])];
    for (const el of described) el.setAttribute('aria-describedby', [...new Set((el.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean).concat(tip.id))].join(' '));
    tip.hidden = false;
    tip.style.left = '16px'; tip.style.top = '16px';
    const r = el.getBoundingClientRect(), box = tip.getBoundingClientRect();
    const below = r.bottom + 6;
    tip.style.left = `${Math.max(16, Math.min(r.left, innerWidth - box.width - 16))}px`;
    tip.style.top = `${Math.max(16, Math.min(below + box.height <= innerHeight - 16 ? below : r.top - box.height - 6, innerHeight - box.height - 16))}px`;
    if (fresh) expiryTimer = setTimeout(dismiss, 6000);
  }
  document.addEventListener('pointerover', e => {
    if (tip.contains(e.target)) { clearTimeout(hideTimer); return; }
    const el = target(e.target);
    if (el !== hovered) {
      hovered = el; dismissed = null;
      if (el) { hide(); showTimer = setTimeout(() => show(el), 600); }
    }
  });
  document.addEventListener('pointerout', e => {
    if (hovered?.contains(e.relatedTarget) || tip.contains(e.relatedTarget)) return;
    if ((hovered && target(e.target) === hovered) || tip.contains(e.target)) {
      hovered = null; clearTimeout(showTimer); clearTimeout(hideTimer);
      hideTimer = setTimeout(hide, 100);
    }
  });
  document.addEventListener('pointermove', e => {
    if (owner?.contains(e.target)) dismiss();
  });
  // A dragged chip must not raise its hint over the drop target. The browser cancels the pointer at
  // dragstart and may send pointerover again mid-drag, so hints stay off until the drag ends.
  document.addEventListener('dragstart', () => { dragging = true; hovered = null; hide(); }, true);
  for (const type of ['dragend', 'drop']) document.addEventListener(type, () => { dragging = false; }, true);
  // Clicks and the app's focus() calls must not summon hints. Tab navigation still exposes them.
  document.addEventListener('focusin', e => { dismissed = null; if (keyboard) show(target(e.target)); else hide(); });
  document.addEventListener('focusout', hide);
  document.addEventListener('keydown', e => {
    if (e.key === 'Tab') keyboard = true;
    if (e.key === 'Escape') { dismissed = owner || hovered; hide(); }
  }, true);
  document.addEventListener('pointerdown', e => {
    keyboard = false;
    if (!tip.contains(e.target)) { dismissed = owner || hovered; hide(); }
  }, true);
  document.addEventListener('input', e => { if (e.target.matches('input,textarea,[contenteditable]')) hide(); });
  document.addEventListener('change', e => { if (owner === e.target) show(owner); });
  document.addEventListener('scroll', e => { if (!tip.contains(e.target)) hide(); }, true);
  window.addEventListener('resize', hide);
  window.addEventListener('blur', hide);
  new MutationObserver(() => {
    if (owner && (!owner.isConnected || !owner.getClientRects().length)) hide();
    else if (owner && tip.textContent !== message(owner)) show(owner);
  }).observe(document.documentElement, {subtree:true, childList:true, attributes:true, attributeFilter:['data-tooltip','hidden','open']});
})();
