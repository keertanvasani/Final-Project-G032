/* "Test your own image" panel for the static demo.

   The hosted page has no models, so an uploaded image can only be analysed by a live
   backend. This panel takes that backend's address (the link printed by
   deploy/run_live_demo.ps1), checks that it answers /health, and then reloads the page
   with ?api=<address>, which js/api.js already understands. It shows the connection state
   and lets the visitor disconnect. If the backend later stops answering, api.js falls
   back to recorded mode and this panel updates. */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const panel = $('backendPanel');
  if (!panel) return;

  const el = {
    title: $('backendTitle'), msg: $('backendMsg'), form: $('backendForm'),
    url: $('backendUrl'), btn: $('backendBtn'),
  };
  let busy = false;

  function origin(raw) {
    try {
      const u = new URL(raw.trim());
      const local = u.hostname === 'localhost' || u.hostname === '127.0.0.1';
      return (u.protocol === 'https:' || (local && u.protocol === 'http:')) ? u.origin : null;
    } catch { return null; }
  }

  function render() {
    const live = window.DEEPSEA_MODE === 'live';
    panel.classList.toggle('is-live', live);
    if (live) {
      let host = '';
      try { host = new URL(window.DEEPSEA_API_BASE).host; } catch { /* leave blank */ }
      el.title.textContent = 'Live backend connected';
      el.msg.textContent = `Your images are analysed on ${host || 'the live backend'}. Upload one below.`;
      el.url.hidden = true;
      el.btn.textContent = 'Disconnect';
      el.btn.classList.remove('btn--primary');
    } else {
      el.title.textContent = 'Testing your own image needs a live backend';
      el.msg.textContent = 'This hosted page replays recorded results for its samples. To analyse your own ' +
        'images, paste the live-demo link here.';
      el.url.hidden = false;
      el.btn.textContent = 'Connect';
      el.btn.classList.add('btn--primary');
    }
  }

  function say(text, bad) {
    el.msg.textContent = text;
    el.msg.classList.toggle('is-bad', Boolean(bad));
  }

  async function connect(addr) {
    const base = origin(addr);
    if (!base) { say('That is not a valid link. It should start with https://', true); return; }
    busy = true; el.btn.disabled = true; say('Checking that the backend answers…');
    try {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), 8000);
      const res = await fetch(`${base}/health`, { signal: ctl.signal });
      clearTimeout(timer);
      const data = res.ok ? await res.json() : null;
      if (!data || !data.models_loaded) {
        say(data ? 'The backend is reachable but its models are still loading. Try again in a minute.'
                 : `The backend answered with ${res.status}.`, true);
        return;
      }
      location.search = `?api=${encodeURIComponent(base)}`;
    } catch {
      say('Could not reach that address. Check that the live-demo script is still running and the link is ' +
          'copied exactly.', true);
    } finally {
      busy = false; el.btn.disabled = false;
    }
  }

  el.form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (busy) return;
    if (window.DEEPSEA_MODE === 'live') { location.search = '?api=off'; return; }
    if (!el.url.value.trim()) { say('Paste the live-demo link first.', true); return; }
    connect(el.url.value);
  });

  window.addEventListener('deepsea:mode', render);
  render();
})();
