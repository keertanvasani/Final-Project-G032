/* Interactive result view: a confidence-threshold slider and the Known/Unknown badge.

   app.js hands this module each /predict response. It redraws the boxes on a canvas
   (the server's annotated PNG has every box baked in, so a threshold can only be
   applied by drawing them again) and keeps the detection list in step with the slider.

   Optional fields. The module works with the plain /predict response and lights up
   more when these are present, and never invents either one:

     detection.novelty = { score, unknown, method }
         Set by a backend that has a validated novelty scorer. Drives the
         Known / Unknown badge and the "Hide unknown" filter. When absent, no badge
         is shown: the aquarium-trained models' novelty scores are near chance
         (see experiments/RESULTS.md section 7), so labelling from them would mislead.

     detection.gt = { outcome, gt_class }   and   response.ground_truth
         Present on the recorded deep-sea samples, where the dataset's own labels
         say whether each box is right. Enables the "% correct among shown boxes"
         readout and colour-by-correctness. outcome is one of
         TP, CLS_ERR, OOD_ERR, LOC_ERR, BG_FP (PROTOCOL.md section 2). */
(function () {
  'use strict';

  const COLORS = {
    high: 'rgb(83,200,0)', medium: 'rgb(255,191,0)', low: 'rgb(255,100,0)',
    right: 'rgb(83,200,0)', wrong: 'rgb(248,113,113)', unknown: 'rgb(167,139,250)',
  };
  const confColor = (c) => (c >= 0.75 ? COLORS.high : c >= 0.5 ? COLORS.medium : COLORS.low);

  // Plain-language outcomes; the key is the harness code.
  const OUTCOME = {
    TP: { label: 'Correct', good: true },
    CLS_ERR: { label: 'Wrong species', good: false },
    OOD_ERR: { label: 'Not in its vocabulary', good: false },
    LOC_ERR: { label: 'Misplaced box', good: false },
    BG_FP: { label: 'Nothing there', good: false },
  };

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const S = { dets: [], img: null, hasGt: false, hasNovelty: false };
  let el = null;

  function threshold() { return Number(el.slider.value) / 100; }

  function visible(d) {
    if (d.combined_confidence < threshold()) return false;
    if (el.hideUnknown.checked && d.novelty && d.novelty.unknown) return false;
    return true;
  }

  /* ── Badge ──────────────────────────────────────────── */

  // Returns '' when the response carries no novelty score, so nothing is shown
  // rather than a guess.
  function badge(d) {
    const n = d && d.novelty;
    if (!n || typeof n.unknown !== 'boolean') return '';
    const score = typeof n.score === 'number' ? ` · ${n.score.toFixed(2)}` : '';
    const tip = n.method ? ` title="Novelty score (${esc(n.method)})"` : '';
    return n.unknown
      ? `<span class="kbadge kbadge--unknown"${tip}>Unknown${score}</span>`
      : `<span class="kbadge kbadge--known"${tip}>Known${score}</span>`;
  }

  // 'right' | 'wrong' | 'review' | null. When the verifier answered, its answer is what the
  // page shows, so that answer is graded (server: ground_truth.py, taxonomy rule).
  function verdict(d) {
    const o = d && d.gt && OUTCOME[d.gt.outcome];
    if (!o) return null;
    const v = d.gt.verifier;
    if (v) return v.correct == null ? 'review' : v.correct ? 'right' : 'wrong';
    return o.good ? 'right' : 'wrong';
  }

  function truthChip(d) {
    const o = d.gt && OUTCOME[d.gt.outcome];
    if (!o) return '';
    const what = d.gt.gt_class && d.gt.outcome !== 'BG_FP' && d.gt.outcome !== 'LOC_ERR'
      ? ` · actually ${esc(d.gt.gt_class)}` : '';
    const v = d.gt.verifier, k = verdict(d);
    let label = o.label;
    if (v && k === 'review') label = 'Held for review';
    else if (v && k === 'right' && v.level !== 'species') label = `Correct at the ${esc(v.answer)} level`;
    else if (v && k === 'right') label = 'Correct';
    return `<span class="tchip tchip--${k}">${label}${what}</span>`;
  }

  /* ── Canvas ─────────────────────────────────────────── */

  function draw() {
    if (!S.img) return;
    const cv = el.canvas, W = S.img.naturalWidth, H = S.img.naturalHeight;
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    const g = cv.getContext('2d');
    g.drawImage(S.img, 0, 0, W, H);

    const lw = Math.max(2, Math.round(W / 240));
    const fs = Math.max(12, Math.round(W / 46));
    g.font = `600 ${fs}px ui-sans-serif, system-ui, "Segoe UI", sans-serif`;
    g.textBaseline = 'middle';
    const byTruth = el.byTruth.checked && S.hasGt;

    // Low confidence first, so a confident box is never hidden under a weak one.
    const order = S.dets.map((d, i) => i).sort(
      (a, b) => S.dets[a].combined_confidence - S.dets[b].combined_confidence);

    order.forEach((i) => {
      const d = S.dets[i];
      if (!visible(d)) return;
      const [x1, y1, x2, y2] = d.bbox;
      const unk = d.novelty && d.novelty.unknown;
      let col = confColor(d.combined_confidence);
      if (byTruth && d.gt) {
        const k = verdict(d);
        col = k === 'right' ? COLORS.right : k === 'review' ? COLORS.unknown : COLORS.wrong;
      }
      else if (unk) col = COLORS.unknown;
      if (d.taxon && !(byTruth && d.gt)) {   // the verifier's answer sets the colour when present
        col = d.taxon.level == null ? COLORS.unknown : confColor(d.taxon.p_correct);
      }

      g.lineWidth = lw;
      g.strokeStyle = col;
      g.setLineDash(unk ? [lw * 3, lw * 2] : []);
      g.strokeRect(x1, y1, x2 - x1, y2 - y1);
      g.setLineDash([]);

      let text = `${d.classifier_class} ${Math.round(d.combined_confidence * 100)}%`;
      if (unk) text = `? ${text}`;
      const tx = d.taxon;   // risk-controlled verifier: the deepest answer that is safe at its error target
      if (tx) {
        if (tx.level == null) text = `needs review (~${d.classifier_class})`;
        else if (tx.level === 'species') text = `${d.classifier_class} ${Math.round(tx.p_correct * 100)}%`;
        else text = `${tx.node} ${Math.round(tx.p_correct * 100)}% (~${d.classifier_class})`;
      }
      const tw = g.measureText(text).width + fs * 0.7, th = fs * 1.45;
      const ly = y1 - th >= 0 ? y1 - th : y1;
      g.fillStyle = col;
      g.fillRect(x1 - lw / 2, ly, tw, th);
      g.fillStyle = '#04070d';
      g.fillText(text, x1 + fs * 0.25, ly + th / 2 + 1);
    });
  }

  /* ── Readout + list ─────────────────────────────────── */

  function refresh() {
    const t = threshold();
    el.thrValue.textContent = t.toFixed(2);
    const shown = S.dets.filter(visible);
    el.thrCount.innerHTML = `<b>${shown.length}</b> of ${S.dets.length} boxes shown`;

    if (S.hasGt) {
      const ok = shown.filter((d) => verdict(d) === 'right').length;
      const held = shown.filter((d) => verdict(d) === 'review').length;
      const bad = shown.length - ok - held;
      el.thrQuality.hidden = false;
      el.thrQuality.innerHTML = shown.length
        ? `<span class="q q--right"><b>${ok}</b> correct</span>` +
          `<span class="q q--wrong"><b>${bad}</b> wrong</span>` +
          (held ? `<span class="q q--rate"><b>${held}</b> held for review</span>` : '') +
          `<span class="q q--rate">${Math.round((100 * ok) / shown.length)}% of the boxes you can see are right</span>`
        : `<span class="q q--rate">No boxes at this threshold.</span>`;
    } else {
      el.thrQuality.hidden = true;
    }

    document.querySelectorAll('#detList .det').forEach((node) => {
      const d = S.dets[Number(node.dataset.idx)];
      node.classList.toggle('is-hidden', !(d && visible(d)));
    });
    draw();
  }

  /* ── Public ─────────────────────────────────────────── */

  function render(data, imageUrl) {
    if (!el) return;
    S.dets = (data && data.detections) || [];
    S.hasGt = S.dets.some((d) => d.gt && OUTCOME[d.gt.outcome]);
    S.hasNovelty = S.dets.some((d) => d.novelty && typeof d.novelty.unknown === 'boolean');

    el.controls.hidden = S.dets.length === 0;
    el.byTruthWrap.hidden = !S.hasGt;
    el.hideUnknownWrap.hidden = !S.hasNovelty;
    el.byTruth.checked = false;
    el.hideUnknown.checked = false;
    el.slider.value = 0;
    el.note.hidden = !S.hasGt;

    const img = new Image();
    img.onload = () => { S.img = img; refresh(); };
    img.src = imageUrl;
  }

  function init() {
    el = {
      canvas: $('liveCanvas'), controls: $('thrControls'), slider: $('thrSlider'),
      thrValue: $('thrValue'), thrCount: $('thrCount'), thrQuality: $('thrQuality'),
      byTruth: $('colorByTruth'), byTruthWrap: $('colorByTruthWrap'),
      hideUnknown: $('hideUnknown'), hideUnknownWrap: $('hideUnknownWrap'),
      note: $('thrNote'),
    };
    if (!el.canvas || !el.slider) { el = null; return; }

    el.slider.addEventListener('input', refresh);
    el.byTruth.addEventListener('change', refresh);
    el.hideUnknown.addEventListener('change', refresh);
    document.querySelectorAll('[data-thr]').forEach((b) =>
      b.addEventListener('click', () => { el.slider.value = Number(b.dataset.thr); refresh(); }));
  }

  window.DeepSeaInteractive = { init, render, badge, truthChip };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
