/* Backend selection for the static deployment.

   This build has no server. app.js still calls fetch('/health') and
   fetch('/predict'), so this file, loaded first, answers those two calls itself:

   recorded  (default)  Replays the real pipeline output that was captured for
                        each built-in sample. An image is matched by the SHA-256
                        of its bytes, so a recording is only ever shown for the
                        exact image it came from. Any other image is refused with
                        an explicit message; nothing is guessed or fabricated.

   live                 Forwards both calls to a running backend, for the review
                        itself. Enabled by opening the page with ?api=<url>, e.g.
                        https://xxxx.trycloudflare.com. If that backend stops
                        answering, the page drops back to recorded mode instead
                        of showing a dead page.

   The chosen backend is kept in sessionStorage so in-page navigation keeps it.
   ?api=off forgets it. window.DEEPSEA_MODE is 'recorded' or 'live' for app.js. */
(function () {
  'use strict';

  const KEY = 'deepsea_api';
  const nativeFetch = window.fetch.bind(window);

  function store(op, value) {
    try {
      if (op === 'get') return sessionStorage.getItem(KEY);
      if (op === 'set') sessionStorage.setItem(KEY, value);
      if (op === 'del') sessionStorage.removeItem(KEY);
    } catch { /* storage can be blocked; the URL parameter still works */ }
    return null;
  }

  function normalise(raw) {
    try {
      const u = new URL(raw);
      return (u.protocol === 'https:' || u.hostname === 'localhost' || u.hostname === '127.0.0.1')
        ? u.origin : null;
    } catch { return null; }
  }

  const param = new URLSearchParams(location.search).get('api');
  if (param === 'off') store('del');
  else if (param) { const o = normalise(param); if (o) store('set', o); }

  let liveBase = normalise(store('get') || '') || null;
  window.DEEPSEA_MODE = liveBase ? 'live' : 'recorded';

  const json = (obj, status = 200) =>
    new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json' } });

  let manifestPromise = null;
  const manifest = () => (manifestPromise ||= nativeFetch('recorded/manifest.json')
    .then((r) => { if (!r.ok) throw new Error('manifest'); return r.json(); }));

  async function sha256(file) {
    if (!(window.crypto && crypto.subtle)) return null;
    const buf = await file.arrayBuffer();
    const digest = await crypto.subtle.digest('SHA-256', buf);
    return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
  }

  function fallToRecorded() {
    liveBase = null;
    window.DEEPSEA_MODE = 'recorded';
  }

  async function recordedHealth() {
    try { await manifest(); } catch { return json({ status: 'down', models_loaded: false }, 503); }
    return json({ status: 'healthy', models_loaded: true, device: 'recorded',
                  model_details: { yolo_n: true, yolo_s: true, classifier: true } });
  }

  async function recordedPredict(body) {
    const file = body && body.get && body.get('file');
    const m = await manifest();
    const sha = file ? await sha256(file) : null;
    if (!sha || !m.images[sha]) {
      return json({
        detail: 'Live inference is not available on this link, so only the built-in samples ' +
                'can be analysed. Pick one of the sample images below, or open the live-demo ' +
                'link to analyse your own photo.',
      }, 501);
    }
    const r = await nativeFetch(`recorded/${sha}.json`);
    if (!r.ok) return json({ detail: 'The recorded result for this sample is missing.' }, 500);
    return json(await r.json());
  }

  window.fetch = async function (input, init) {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    const path = url.replace(location.origin, '');
    const isHealth = path === '/health';
    const isPredict = path === '/predict';
    if (!isHealth && !isPredict) return nativeFetch(input, init);

    if (liveBase) {
      try {
        const res = await nativeFetch(liveBase + path, init);
        // A tunnel that has closed answers with an HTML error page, not JSON.
        if (res.status >= 520 || res.status === 404 || res.status === 502 || res.status === 530) {
          throw new Error(`backend returned ${res.status}`);
        }
        return res;
      } catch {
        // Keep a copy of the file only if we still can; FormData is reusable.
        fallToRecorded();
      }
    }
    return isHealth ? recordedHealth() : recordedPredict(init && init.body);
  };
})();
