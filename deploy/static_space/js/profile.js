/* Model choice: deep sea (9 classes, incl. machine, with Known/Unknown) or aquarium (7 classes),
   and an on/off switch for conditional enhancement (decision point D1).

   The backend serves both stacks and picks one per request with /predict?profile=<name>.
   This module adds a two-way switch above the Analyse button, appends the chosen profile
   to every /predict call, and sets the switch automatically when a sample is picked
   (deep-sea frames use the deep-sea models, aquarium photos the aquarium ones).

   It asks /health which profiles the backend has. A backend without profiles (an older
   build) gets no switch and no parameter, so the page behaves exactly as before. */
(function () {
  'use strict';

  const actions = document.querySelector('.actions');
  const sampleRow = document.getElementById('sampleRow');
  if (!actions) return;

  const HINT = {
    deepsea: 'Trained on JEDI deep-sea footage: 9 classes including ROV hardware, and it ' +
             'marks animals unlike anything it was trained on as Unknown.',
    aquarium: 'Trained on aquarium photos: fish, jellyfish, penguin, puffin, shark, starfish, stingray.',
  };
  let profiles = null;
  let chosen = null;

  const box = document.createElement('div');
  box.className = 'profile';
  box.hidden = true;
  box.innerHTML =
    '<span class="profile__label">Model</span>' +
    '<div class="profile__seg" role="radiogroup" aria-label="Model"></div>' +
    '<label class="profile__enh" title="Decision point D1: CLAHE when the image is dark, bright or ' +
    'low-contrast, sharpening when it is blurry. Switch it off to compare.">' +
    '<input type="checkbox" id="enhanceToggle" checked> Enhance image when needed</label>' +
    '<p class="profile__hint"></p>';
  actions.parentNode.insertBefore(box, actions);
  const seg = box.querySelector('.profile__seg');
  const hint = box.querySelector('.profile__hint');
  const enh = box.querySelector('#enhanceToggle');

  function choose(name) {
    if (!profiles || !profiles[name]) return;
    chosen = name;
    seg.querySelectorAll('button').forEach((b) => {
      const on = b.dataset.profile === name;
      b.classList.toggle('is-on', on);
      b.setAttribute('aria-checked', String(on));
    });
    hint.textContent = HINT[name] || profiles[name].label;
  }

  function build(health) {
    profiles = health && health.profiles && Object.keys(health.profiles).length ? health.profiles : null;
    // On the hosted page in replay mode, /health comes from the recordings and lists
    // the profiles each sample was recorded under, so the switch works there too.
    if (!profiles) { box.hidden = true; chosen = null; return; }
    seg.innerHTML = '';
    ['deepsea', 'aquarium'].filter((n) => profiles[n]).forEach((n) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.dataset.profile = n;
      b.setAttribute('role', 'radio');
      b.textContent = profiles[n].label;
      b.addEventListener('click', () => choose(n));
      seg.appendChild(b);
    });
    box.hidden = false;
    choose(chosen && profiles[chosen] ? chosen : (health.default_profile || 'deepsea'));
  }

  // Samples carry their natural model: deep-sea frames are named ds_*.
  if (sampleRow) {
    sampleRow.addEventListener('click', (e) => {
      const b = e.target.closest('.sample');
      if (b && profiles) choose((b.dataset.name || '').startsWith('ds_') ? 'deepsea' : 'aquarium');
    }, true);
  }

  // Append ?profile= to /predict. Wraps whatever fetch is installed (api.js on the hosted page).
  const inner = window.fetch.bind(window);
  window.fetch = function (input, init) {
    // Only the page's own bare '/predict' call is rewritten; a URL that already
    // carries parameters is left exactly as the caller wrote it.
    if (chosen && input === '/predict') {
      input = `/predict?profile=${encodeURIComponent(chosen)}` + (enh.checked ? '' : '&enhance=false');
    }
    return inner(input, init);
  };

  function refresh() {
    inner('/health').then((r) => (r.ok ? r.json() : null)).then(build).catch(() => build(null));
  }
  window.addEventListener('deepsea:mode', refresh);
  refresh();
})();
