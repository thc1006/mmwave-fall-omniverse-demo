---
name: mmwave-fall-ml
description: >
  Help design, train, and serve the mmWave fall detection model, including
  data processing, PyTorch training scripts, and the FastAPI prediction API.
---

# Skill: mmWave fall detection ML / API

## When to use this skill

Claude should use this skill when the user asks you to:

- Work on code under `ml/` (model definition, training, evaluation)
- Modify `services/api/main.py` or related backend code
- Design data schemas for radar episodes and model outputs
- Improve training, validation, or metrics for fall vs normal classification
- Integrate the model predictions into notifications or dashboards

## Objectives

- Keep the ML code simple, readable, and easy to experiment with.
- Make it straightforward to plug in new features (e.g., different radar representations).
- Ensure the API is well-typed and safe to call from Omniverse or other clients.
- Preserve a clear contract: input is radar frames, output is a fall probability and label.

## ML responsibilities

- `ml/fallnet_model.py`
  - Maintain a small, well-documented PyTorch model (`FallNet`) suitable for rapid iteration.
  - Use type hints and dataclasses (`FallNetConfig`) to keep hyperparameters explicit.

- `ml/train_fallnet.py`
  - Expect `.npz` episodes from `ml/data/normal/` and `ml/data/fall/`.
  - Perform basic train/validation splitting as needed.
  - Log simple metrics (loss, accuracy) per epoch.
  - Save `fallnet.pt` with both model weights and config.

## API responsibilities

- `services/api/main.py`
  - Load the model once on startup (lazy-loading is fine) and keep it on CPU or GPU.
  - Define clear Pydantic models for the request and response payloads.
  - Keep endpoints minimal: `/health` and `/predict` are sufficient for the core demo.
  - Ensure robust error messages when the model or data is missing.

## Guardrails

- Do not overcomplicate the model: the purpose is a **demo**, not SOTA radar research.
- Avoid heavy third-party dependencies beyond PyTorch, FastAPI, and basic scientific Python.
- Keep training scripts runnable on a single GPU or CPU-only machine where possible (for debugging).

## Example tasks for this skill

- “Add a simple validation split and report per-class precision/recall for fall vs normal.”
- “Add feature normalization and experiment with a 1D-CNN architecture instead of an MLP.”
- “Update the API to return a short natural-language explanation along with the numeric scores.”
