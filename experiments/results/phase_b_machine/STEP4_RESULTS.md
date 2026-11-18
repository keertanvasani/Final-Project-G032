# Step 4 — free local foundation model as a second opinion: BioCLIP feasibility (2026-11-18)

Question: can BioCLIP (imageomics/bioclip, ViT-B/16, open weights, run locally on the
RTX 5050) serve, without training, as an escalation action for hard boxes — a
classifier second opinion constrained to the 9 classes or "unknown"?

Script: `experiments/step4_bioclip_zeroshot.py` (needs `open_clip_torch`; weights cached in
`models/hf_cache`, 571 MB). Split: **S_cal** only. Result file:
`step4/bioclip_zeroshot_S_cal.json`. Prompts are fixed in the script (3 templates ×
common and Latin names per class; a fixed list of 28 other marine groups for "other").

## A. Zero-shot classification of ground-truth crops (1,351 known-class crops)

| | Accuracy |
|---|---|
| **ResNet-18 (deployed)** | **0.868** |
| BioCLIP zero-shot, 9 classes | 0.332 |
| BioCLIP zero-shot, 8 animal classes only (no "machine") | 0.370 (ResNet on the same crops 0.835) |
| BioCLIP zero-shot, full taxonomic-path prompts | 0.180 |
| Average of ResNet + BioCLIP probabilities | 0.828 (worse than ResNet alone) |
| BioCLIP features + linear probe trained on S_cal (5-fold by dive) | 0.638 |
| Average of ResNet + that probe | 0.867 (no gain) |

BioCLIP is right where ResNet is wrong on only 2.4% of crops, and wrong where ResNet is
right on 56%. Per class it is weakest exactly on the deep-sea invertebrates (sea cucumber
0.03, shrimp 0.09, sponge 0.11, starfish 0.11) and strongest on fish (0.58).

## B. Known/Unknown (1,351 known vs 222 held-out-taxa crops)

| Score | AUROC |
|---|---|
| **ResNet energy (deployed)** | **0.769** |
| BioCLIP, max similarity to the 9 classes | 0.450 (below chance) |
| BioCLIP, probability on "other marine groups" (optimistic: the list names groups containing held-out taxa) | 0.617 |
| Energy + BioCLIP closed score | 0.654 |

## C. As a second opinion on the agent's own boxes (learned policy, 1,527 boxes)

Agent outcomes: 902 TP, 252 LOC_ERR, 162 OOD_ERR, 124 BG_FP, 87 CLS_ERR.

* On the 87 wrongly labelled boxes, BioCLIP has the right class on 18 (21%).
* On the 902 correct boxes it disagrees with the agent on 63%.
* Switching to BioCLIP whenever it disagrees: **−549 correct boxes**; the hindsight
  upper bound (switch only when it fixes the box) is +18.
* On the 162 unseen animals reported as known, its "other" score fires on 48% — but also
  on 28% of correct boxes.
* Cost: ~10 ms per crop on the laptop GPU.

## Verdict

**Rejected as an escalation action.** On this footage (low-resolution ROV frames, small
crops, deep-sea invertebrates) BioCLIP zero-shot is far weaker than the trained ResNet-18
and adds nothing as a second opinion or as a Known/Unknown score; even a probe trained on
its features does not complement the ResNet. No escalation rung is added to the agent.

The agent's largest error class is localisation (LOC_ERR, 252 boxes), not classification
(CLS_ERR, 87): the remaining headroom is in where the boxes are, which argues for the
step-5 work (tiling for small animals, retraining the classifier on detector crops) and,
for re-detection, an open-vocabulary detector (OWLv2) rather than another classifier.
