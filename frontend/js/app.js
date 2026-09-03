/* ═══════════════════════════════════════════════════════════
   Deep-Sea Creature Detection — front-end controller

   Talks to the existing FastAPI backend:
     GET  /health   → model readiness
     POST /predict  → multipart image, returns annotated PNG + trace

   Nothing here changes model behaviour; it only presents what
   /predict already returns.
   ═══════════════════════════════════════════════════════════ */

(() => {
  'use strict';

  // Mirror of config.ALLOWED_IMAGE_TYPES / MAX_IMAGE_SIZE_MB so the user
  // gets instant feedback instead of a round-trip 400.
  const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
  const MAX_SIZE_MB = 20;

  // Matches config.get_confidence_color(). cv2 draws in BGR, so these are
  // the tuples read back as RGB — i.e. what actually appears on the image.
  const CONF_COLORS = {
    high:   'rgb(83, 200, 0)',
    medium: 'rgb(255, 191, 0)',
    low:    'rgb(255, 100, 0)',
  };

  const confColor = (c) => (c >= 0.75 ? CONF_COLORS.high : c >= 0.5 ? CONF_COLORS.medium : CONF_COLORS.low);
  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  const $ = (id) => document.getElementById(id);

  const el = {
    healthPill: $('healthPill'),
    healthText: $('healthText'),
    dropZone: $('dropZone'),
    fileInput: $('fileInput'),
    dropIdle: $('dropIdle'),
    dropPreview: $('dropPreview'),
    previewImg: $('previewImg'),
    fileName: $('fileName'),
    clearBtn: $('clearBtn'),
    sampleRow: $('sampleRow'),
    analyzeBtn: $('analyzeBtn'),
    analyzeLabel: $('analyzeLabel'),
    actionNote: $('actionNote'),
    errorBox: $('errorBox'),
    errorTitle: $('errorTitle'),
    errorMsg: $('errorMsg'),
    loadingBox: $('loadingBox'),
    loadingSteps: $('loadingSteps'),
    results: $('results'),
    resultTitle: $('resultTitle'),
    resultLede: $('resultLede'),
    reviewBox: $('reviewBox'),
    reviewMsg: $('reviewMsg'),
    compareStage: $('compareStage'),
    origImg: $('origImg'),
    annotImg: $('annotImg'),
    downloadBtn: $('downloadBtn'),
    statsRow: $('statsRow'),
    detList: $('detList'),
    detCount: $('detCount'),
    traceList: $('traceList'),
    againBtn: $('againBtn'),
    nav: $('nav'),
    motes: $('motes'),
  };

  let selectedFile = null;
  let previewUrl = null;
  let modelsReady = false;
  let busy = false;
  let stepTimers = [];

  /* ── Utilities ──────────────────────────────────────── */

  function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function showError(title, message) {
    el.errorTitle.textContent = title;
    el.errorMsg.textContent = message;
    el.errorBox.hidden = false;
  }

  function clearError() {
    el.errorBox.hidden = true;
  }

  /* ── Health ─────────────────────────────────────────── */

  async function checkHealth() {
    try {
      const res = await fetch('/health');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      modelsReady = Boolean(data.models_loaded);

      if (data.models_loaded) {
        el.healthPill.dataset.state = 'ready';
        el.healthText.textContent = `Models ready · ${String(data.device).toUpperCase()}`;
      } else if (data.status === 'partial') {
        el.healthPill.dataset.state = 'partial';
        const missing = Object.entries(data.model_details || {})
          .filter(([, ok]) => !ok).map(([k]) => k).join(', ');
        el.healthText.textContent = `Partial — missing ${missing || 'models'}`;
      } else {
        el.healthPill.dataset.state = 'down';
        el.healthText.textContent = 'Models loading…';
      }
    } catch {
      modelsReady = false;
      el.healthPill.dataset.state = 'down';
      el.healthText.textContent = 'Backend offline';
    }
    syncButton();
  }

  function syncButton() {
    if (busy) return;

    if (!modelsReady) {
      el.analyzeBtn.disabled = true;
      el.actionNote.textContent =
        el.healthPill.dataset.state === 'down'
          ? 'Backend unreachable — start the server with: uvicorn webapp:app --port 8000'
          : 'Waiting for models to finish loading…';
      return;
    }
    el.analyzeBtn.disabled = !selectedFile;
    el.actionNote.textContent = selectedFile
      ? 'Runs the full ensemble pipeline locally. Usually a few seconds on CPU.'
      : 'Select an image to begin.';
  }

  /* ── File selection ─────────────────────────────────── */

  function validate(file) {
    if (!ALLOWED_TYPES.includes(file.type)) {
      const ext = (file.name.split('.').pop() || '').toLowerCase();
      if (ext === 'heic' || ext === 'heif') {
        return 'HEIC images (the iPhone default) are not supported. In Photos, export the image as JPEG first, or change Camera settings to "Most Compatible".';
      }
      return `"${file.type || ext || 'unknown'}" is not a supported format. Please use JPEG, PNG or WebP.`;
    }
    const mb = file.size / (1024 * 1024);
    if (mb > MAX_SIZE_MB) {
      return `That image is ${mb.toFixed(1)} MB. The maximum accepted size is ${MAX_SIZE_MB} MB.`;
    }
    return null;
  }

  function setFile(file, { fromSample = false } = {}) {
    const problem = validate(file);
    if (problem) {
      showError('That file cannot be used', problem);
      return;
    }
    clearError();

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    selectedFile = file;
    previewUrl = URL.createObjectURL(file);

    el.previewImg.src = previewUrl;
    el.fileName.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
    el.dropIdle.hidden = true;
    el.dropPreview.hidden = false;
    el.dropZone.classList.add('has-file');

    if (!fromSample) {
      el.sampleRow.querySelectorAll('.sample').forEach((b) => b.classList.remove('is-active'));
    }
    syncButton();
  }

  function clearFile() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    selectedFile = null;
    el.fileInput.value = '';
    el.previewImg.removeAttribute('src');
    el.dropIdle.hidden = false;
    el.dropPreview.hidden = true;
    el.dropZone.classList.remove('has-file');
    el.sampleRow.querySelectorAll('.sample').forEach((b) => b.classList.remove('is-active'));
    clearError();
    syncButton();
  }

  /* ── Loading choreography ───────────────────────────── */

  function startLoading() {
    el.loadingBox.hidden = false;
    const steps = [...el.loadingSteps.children];
    steps.forEach((s) => s.classList.remove('is-active', 'is-done'));

    // The API is a single round-trip with no progress stream, so these are
    // paced indicators. The real per-stage timings are shown in the result.
    const pacing = [0, 450, 1500, 1900, 2900];
    stepTimers = pacing.map((delay, i) =>
      setTimeout(() => {
        steps.forEach((s, j) => {
          s.classList.toggle('is-active', j === i);
          s.classList.toggle('is-done', j < i);
        });
      }, delay)
    );
  }

  function stopLoading() {
    stepTimers.forEach(clearTimeout);
    stepTimers = [];
    [...el.loadingSteps.children].forEach((s) => {
      s.classList.remove('is-active');
      s.classList.add('is-done');
    });
    el.loadingBox.hidden = true;
  }

  /* ── Predict ────────────────────────────────────────── */

  async function analyze() {
    if (!selectedFile || busy) return;

    busy = true;
    clearError();
    el.results.hidden = true;
    el.analyzeBtn.classList.add('is-busy');
    el.analyzeBtn.disabled = true;
    el.analyzeLabel.textContent = 'Analysing…';
    el.actionNote.textContent = 'Running detection, fusion and classification…';
    startLoading();

    const body = new FormData();
    body.append('file', selectedFile, selectedFile.name);

    try {
      const res = await fetch('/predict', { method: 'POST', body });

      if (!res.ok) {
        let detail = `The server responded with ${res.status}.`;
        try {
          const err = await res.json();
          if (err && err.detail) detail = err.detail;
        } catch { /* non-JSON error body */ }

        const titles = {
          400: 'Image rejected',
          413: 'Image too large',
          503: 'Models not ready',
          500: 'Inference failed',
        };
        showError(titles[res.status] || 'Request failed', detail);
        return;
      }

      render(await res.json());
    } catch (e) {
      showError(
        'Could not reach the server',
        `${e.message}. Check that the backend is still running on this port.`
      );
    } finally {
      stopLoading();
      busy = false;
      el.analyzeBtn.classList.remove('is-busy');
      el.analyzeLabel.textContent = 'Analyse image';
      syncButton();
    }
  }

  /* ── Rendering ──────────────────────────────────────── */

  function render(data) {
    const dets = data.detections || [];
    const summary = data.summary || {};
    const trace = data.decision_trace || [];

    // The agent flags an image for review by returning zero detections and a
    // banner image; the reason is the last line of the trace.
    const needsReview = dets.length === 0 && (summary.total_objects || 0) === 0;

    el.origImg.src = previewUrl;
    el.annotImg.src = `data:image/png;base64,${data.annotated_image}`;
    el.downloadBtn.href = el.annotImg.src;
    el.downloadBtn.download = `annotated-${(selectedFile.name || 'image').replace(/\.[^.]+$/, '')}.png`;

    if (needsReview) {
      const last = trace[trace.length - 1] || '';
      el.resultTitle.textContent = 'No reliable detections';
      el.resultLede.textContent = 'The pipeline ran, but stopped short of reporting results.';
      el.reviewMsg.textContent = last.replace(/^Final state:\s*/i, '') ||
        'Confidence stayed below the acceptance threshold.';
      el.reviewBox.hidden = false;
    } else {
      const n = summary.total_objects || dets.length;
      el.resultTitle.textContent = `${n} creature${n === 1 ? '' : 's'} detected`;
      el.resultLede.textContent =
        `Boxes were fused from two detectors and each region re-checked by the species classifier.`;
      el.reviewBox.hidden = true;
    }

    renderStats(summary, needsReview);
    renderDetections(dets);
    renderTrace(trace);

    el.results.hidden = false;

    // The results section starts display:none, so its reveal targets may never have
    // intersected. Show them outright rather than depending on the observer firing.
    el.results.querySelectorAll('.reveal').forEach((n) => {
      n.style.transitionDelay = '0ms';
      n.classList.add('is-in');
    });

    el.results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderStats(summary, needsReview) {
    const b = summary.inference_breakdown || {};
    const m = summary.models_used || {};
    const total = summary.inference_time_ms || 0;

    const rows = (obj) =>
      Object.entries(obj).map(([k, v]) => `<div><span>${k}</span><span>${v}</span></div>`).join('');

    el.statsRow.innerHTML = `
      <div class="stat">
        <div class="stat__v">${summary.total_objects ?? 0}</div>
        <div class="stat__k">Objects detected</div>
      </div>
      <div class="stat">
        <div class="stat__v" style="color:${needsReview ? 'var(--text-mute)' : confColor(summary.average_confidence || 0)}">
          ${needsReview ? '—' : pct(summary.average_confidence || 0)}
        </div>
        <div class="stat__k">Average confidence</div>
      </div>
      <div class="stat">
        <div class="stat__v">${total < 1000 ? Math.round(total) : (total / 1000).toFixed(2)}<small>${total < 1000 ? 'ms' : 's'}</small></div>
        <div class="stat__k">Total inference</div>
        <div class="stat__sub">${rows({
          Preprocess: `${b.preprocessing_ms ?? 0} ms`,
          'YOLOv8n': `${b.detector_nano_ms ?? 0} ms`,
          'YOLOv8s': `${b.detector_small_ms ?? 0} ms`,
          WBF: `${b.wbf_ms ?? 0} ms`,
          // Only shown when the agent actually took these actions, so a zero
          // row isn't rendered as though the work happened and cost nothing.
          ...(b.zoom_ms ? { Zoom: `${b.zoom_ms} ms` } : {}),
          Classify: `${b.classification_ms ?? 0} ms`,
          ...(b.reclassification_ms ? { 'Re-classify': `${b.reclassification_ms} ms` } : {}),
          Annotate: `${b.annotation_ms ?? 0} ms`,
          Agent: `${b.agent_decision_ms ?? 0} ms`,
        })}</div>
      </div>
      <div class="stat">
        <div class="stat__v">3<small>models</small></div>
        <div class="stat__k">Pipeline</div>
        <div class="stat__sub">${rows({
          'Detector A': m.detector_1 || 'YOLOv8n',
          'Detector B': m.detector_2 || 'YOLOv8s',
          Classifier: m.classifier || 'ResNet18',
          Fusion: 'WBF',
        })}</div>
      </div>`;
  }

  function renderDetections(dets) {
    el.detCount.textContent = dets.length;

    if (!dets.length) {
      el.detList.innerHTML =
        `<p class="panel__empty">No objects were reported.<br>The confidence gate rejected this image rather than guessing.</p>`;
      return;
    }

    el.detList.innerHTML = dets.map((d) => {
      const c = d.combined_confidence;
      const label = d.agreement ? d.detector_class : `${d.detector_class} / ${d.classifier_class}`;
      const bar = (v) =>
        `<div class="det__bar"><i style="width:${(v * 100).toFixed(1)}%;background:${confColor(v)}"></i></div>`;

      return `
        <article class="det" style="border-left-color:${confColor(c)}">
          <div class="det__top">
            <span class="det__id">${d.id}</span>
            <span class="det__name">${escapeHtml(label)}</span>
            <span class="det__conf" style="color:${confColor(c)}">${pct(c)}</span>
          </div>
          <div class="det__models">
            <div class="det__model">
              <span class="who">Detector</span>
              <span class="cls">${escapeHtml(d.detector_class)}</span>
              ${bar(d.detector_confidence)}
              <span class="det__pct">${pct(d.detector_confidence)}</span>
            </div>
            <div class="det__model">
              <span class="who">Classifier</span>
              <span class="cls">${escapeHtml(d.classifier_class)}</span>
              ${bar(d.classifier_confidence)}
              <span class="det__pct">${pct(d.classifier_confidence)}</span>
            </div>
          </div>
          <div class="det__bbox">
            <span class="det__agree ${d.agreement ? 'det__agree--yes' : 'det__agree--no'}">
              ${d.agreement ? '✓ models agree' : '⚠ resolved by vote'}
            </span>
            &nbsp; bbox [${d.bbox.map((v) => Math.round(v)).join(', ')}]
          </div>
        </article>`;
    }).join('');
  }

  function renderTrace(trace) {
    if (!trace.length) {
      el.traceList.innerHTML = `<p class="panel__empty">No trace returned.</p>`;
      return;
    }

    el.traceList.innerHTML = trace.map((line) => {
      let cls = '';
      if (/^Action:/i.test(line)) cls = 'is-action';
      if (/^Final/i.test(line)) cls = 'is-final';
      if (/review|too blurry|too low|cannot/i.test(line)) cls = 'is-warn';

      let html = escapeHtml(line);
      const colon = html.indexOf(':');
      if (colon > -1 && colon < 42) {
        html = `<b>${html.slice(0, colon + 1)}</b>${html.slice(colon + 1)}`;
      }
      html = html.replace(/\b\d+\.\d+\b/g, (m) => `<code>${m}</code>`);

      return `<li class="${cls}">${html}</li>`;
    }).join('');
  }

  /* ── Events ─────────────────────────────────────────── */

  el.dropZone.addEventListener('click', (e) => {
    if (e.target.closest('#clearBtn')) return;
    if (!selectedFile) el.fileInput.click();
  });

  el.dropZone.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && !selectedFile) {
      e.preventDefault();
      el.fileInput.click();
    }
  });

  el.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) setFile(e.target.files[0]);
  });

  el.clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
    el.fileInput.click();
  });

  ['dragenter', 'dragover'].forEach((evt) =>
    el.dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      el.dropZone.classList.add('is-drag');
    })
  );

  ['dragleave', 'drop'].forEach((evt) =>
    el.dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      if (evt === 'dragleave' && el.dropZone.contains(e.relatedTarget)) return;
      el.dropZone.classList.remove('is-drag');
    })
  );

  el.dropZone.addEventListener('drop', (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) setFile(f);
  });

  // Page-level drop guard so a stray miss doesn't navigate away from the app.
  ['dragover', 'drop'].forEach((evt) =>
    window.addEventListener(evt, (e) => {
      if (!el.dropZone.contains(e.target)) e.preventDefault();
    })
  );

  el.sampleRow.addEventListener('click', async (e) => {
    const btn = e.target.closest('.sample');
    if (!btn || busy) return;

    el.sampleRow.querySelectorAll('.sample').forEach((b) => b.classList.remove('is-active'));
    btn.classList.add('is-active');

    try {
      const res = await fetch(btn.dataset.src);
      const blob = await res.blob();
      setFile(new File([blob], btn.dataset.name, { type: blob.type || 'image/jpeg' }), { fromSample: true });
    } catch {
      showError('Sample unavailable', 'That sample image could not be loaded from the server.');
    }
  });

  el.analyzeBtn.addEventListener('click', analyze);

  el.againBtn.addEventListener('click', () => {
    clearFile();
    el.results.hidden = true;
    document.getElementById('try').scrollIntoView({ behavior: 'smooth' });
  });

  el.compareStage && document.querySelectorAll('.toggle__btn').forEach((btn) =>
    btn.addEventListener('click', () => {
      document.querySelectorAll('.toggle__btn').forEach((b) => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      el.compareStage.dataset.view = btn.dataset.view;
    })
  );

  /* ── Ambience ───────────────────────────────────────── */

  window.addEventListener('scroll', () => {
    el.nav.classList.toggle('is-stuck', window.scrollY > 20);
  }, { passive: true });

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (!reduceMotion && el.motes) {
    const frag = document.createDocumentFragment();
    for (let i = 0; i < 28; i++) {
      const m = document.createElement('span');
      const size = 1 + Math.random() * 3;
      m.className = 'mote';
      m.style.cssText =
        `left:${Math.random() * 100}%;bottom:-10px;width:${size}px;height:${size}px;` +
        `opacity:${0.2 + Math.random() * 0.5};` +
        `animation-duration:${14 + Math.random() * 22}s;animation-delay:${-Math.random() * 30}s`;
      frag.appendChild(m);
    }
    el.motes.appendChild(frag);
  }

  // Reveal-on-scroll for the main content blocks.
  const targets = document.querySelectorAll(
    '.section__head, .pipe, .pipe__note, .vs, .uniq__card, .claim, .try, .metrics, .charts, .species'
  );
  if ('IntersectionObserver' in window && !reduceMotion) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-in');
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

    targets.forEach((t, i) => {
      t.classList.add('reveal');
      t.style.transitionDelay = `${Math.min(i % 6, 5) * 55}ms`;
      io.observe(t);
    });
  }

  /* ── Boot ───────────────────────────────────────────── */

  checkHealth();
  // Models load during FastAPI's lifespan startup; re-poll until they land.
  const healthPoll = setInterval(() => {
    if (modelsReady) { clearInterval(healthPoll); return; }
    checkHealth();
  }, 4000);
})();
