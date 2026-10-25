---
title: Deep-Sea Creature Detection
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Deep-Sea Creature Detection and Classification

Agentic underwater detection framework — Team 32, Amrita CSE.

A YOLOv8-n / YOLOv8-s ensemble fused with Weighted Boxes Fusion, a ResNet-18
species classifier, and a rule-based controller that decides when to enhance,
zoom, re-classify, or abstain.

This Space serves the aquarium-trained models on CPU. Expect a few seconds per
image, and a slower first request after the Space has been idle.

The models are trained on aquarium imagery. Deep-sea footage is out of domain,
and the interface reports where that matters; see the Findings section of the
site.
