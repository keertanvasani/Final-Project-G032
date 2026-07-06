# Demo Script — Deep-Sea Creature Detection & Classification System

Use these talking points when presenting the system to professors, reviewers, or government stakeholders. Each point is designed to be spoken in ~30 seconds, with a live demo action alongside it.

---

## 1. The Problem: Why Automated Underwater Detection Matters

> "Monitoring marine ecosystems currently requires marine biologists to manually review thousands of hours of underwater footage. This is slow, expensive, and error-prone. Our system automates species detection and classification from underwater imagery — turning hours of manual work into seconds of automated analysis."

**Demo action:** Show a sample underwater image with multiple creatures visible.

---

## 2. Our Approach: Why One Model Isn't Enough

> "A single detection model can miss objects or misclassify species. Our system uses a **multi-model ensemble** — three independent AI models working together — to deliver more reliable results than any single model could achieve alone. This is the same ensemble strategy used in production AI systems at companies like Google and in medical imaging diagnostics."

**Demo action:** Point to the pipeline badges in the UI showing "YOLOv8-n ✓ YOLOv8-s ✓ Classifier ✓".

---

## 3. Stage 1: Dual Detection with Weighted Boxes Fusion

> "We run two independently trained YOLOv8 models — a lightweight 'nano' variant and a more powerful 'small' variant — on every image. Each model is trained with different augmentation strategies, so they learn to detect creatures from different perspectives. Their predictions are merged using **Weighted Boxes Fusion**, which intelligently averages overlapping bounding boxes rather than simply discarding duplicates. The result: more precise bounding boxes and fewer missed detections compared to using either model alone."

**Demo action:** Upload an image and highlight detections. Note the tight bounding boxes.

---

## 4. Stage 2: Independent Species Verification

> "For every detected creature, we crop that region and pass it through a separate **ResNet18 image classifier** that independently identifies the species. This acts as a second opinion — if both the detector and the classifier agree on the species, we mark it as **high confidence**. If they disagree, we flag it for human review. This two-stage verification dramatically reduces false identifications."

**Demo action:** Point to a detection where `agreement: true` and note the high combined confidence.

---

## 5. Combined Confidence Scoring: Trust Through Transparency

> "Every detection comes with a transparent confidence breakdown: the detector's confidence, the classifier's confidence, and a combined score. The combined score weights both models — 60% from the detection ensemble, 40% from the classifier — with an additional 10% boost when both models agree. This gives reviewers a clear, quantifiable measure of how much to trust each identification. There are no black boxes here — every decision is explainable."

**Demo action:** Show the detection results panel with the confidence breakdown for several detections.

---

## 6. Real-Time Feasibility and Deployment Readiness

> "The entire pipeline runs on a standard laptop in under 500 milliseconds per image — no specialized hardware required, though it's even faster with GPU acceleration. The system is served via a REST API, making it easy to integrate with existing monitoring platforms, mobile apps, or web dashboards. It's designed as a production-ready prototype that can be deployed on boats, research vessels, or monitoring stations with minimal infrastructure."

**Demo action:** Point to the inference time in the stats footer. Upload 2–3 more images to show consistent speed.

---

## 7. Agentic Self-Correction Loop: Intelligent Autonomy

> "The system doesn't just run the same steps every time — it inspects each image and its own confidence levels, and decides whether to enhance the image, re-check a detection, or escalate to a human reviewer. That decision-making loop is what makes this an agentic system rather than a fixed pipeline. And when a photo is genuinely too degraded to trust — badly out of focus, for example — the system does not guess. It prints a clear warning directly on the returned image: 'Image too blurry to analyze reliably. Please try another photo.' No silent failures, no fabricated low-confidence labels handed to a reviewer as if they were real."

**Demo action:** Point to the `decision_trace` array in the JSON response showing the logs of image quality assessment, pre-enhancements applied (like CLAHE/sharpening), zoom-and-recheck actions, and classifier majority vote resolutions. Then upload a deliberately blurry photo to show the red warning banner rendered directly onto the returned image, and note that every prediction — success or warning — is automatically archived as a PNG under `outputs/` for later review.

---

## Key Technical Facts (for Q&A)

| Metric | Value |
|--------|-------|
| **Models used** | 3 (YOLOv8n + YOLOv8s + ResNet18) |
| **Fusion method** | Weighted Boxes Fusion (WBF) |
| **Species supported** | 7 (fish, jellyfish, penguin, puffin, shark, starfish, stingray) |
| **Training dataset** | Aquarium Combined (638 annotated images) |
| **Inference time** | ~300–500ms on CPU, ~50–100ms on GPU |
| **API framework** | FastAPI (Python) |
| **Hardware requirement** | Standard laptop/PC (GPU optional) |

## Common Questions & Answers

**Q: Why not just use one, larger model?**
> A: A single model has systematic blind spots. By using two different architectures trained with different strategies, we cover more failure modes. WBF's box averaging also produces more geometrically accurate detections than any single model's NMS output.

**Q: Could this work with other species/environments?**
> A: Absolutely. The pipeline is modular — swap in a different training dataset (e.g., coral reef species, deep-sea hydrothermal vent organisms) and retrain the same architecture. The ensemble framework stays identical.

**Q: How does this compare to human accuracy?**
> A: On our test dataset, the ensemble achieves detection recall comparable to trained annotators, with the advantage of processing images in milliseconds rather than minutes. The confidence scoring also quantifies uncertainty, which human reviewers typically can't do consistently.

**Q: What happens when the models disagree?**
> A: When the detector and classifier predict different species, the `agreement` flag is set to `false` and the combined confidence is naturally lower (no agreement boost). This flags uncertain detections for human review — the system is designed to augment human expertise, not replace it.

**Q: What happens if a photo is too blurry to analyze?**
> A: The agent tries to recover it first — CLAHE contrast correction and a sharpening filter both run automatically when the quality check detects a problem. If detection confidence is still too low afterward, the system does not guess: it flags the image for human review, prints "Image too blurry to analyze reliably. Please try another photo." directly onto the returned image, and logs the same message in the decision trace. Every output, including these warnings, is auto-saved to disk in `outputs/` for later review.
