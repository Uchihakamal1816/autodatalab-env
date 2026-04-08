---
title: AutoDataLab Data Cleaning
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - data-cleaning
  - reinforcement-learning
  - ecommerce
---

# AutoDataLab — OpenEnv data cleaning

OpenEnv environment for **real-world tabular data cleaning** (pandas): duplicates, imputation, robust outliers, optional plot declaration, deterministic grading.

See the **repository root** [`README.md`](../README.md) for full action/observation specs, tasks, baseline scores, and Docker/HF instructions.

## Quick start (local)

```bash
pip install -e ".[dev]"
openenv validate --verbose
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Baseline

```bash
python baseline_inference.py --oracle
```

## Docker image (from repository root)

The `Dockerfile` at the **parent** of this folder copies `data_cleaning_env/` into the image:

```bash
cd ..
docker build -t autodatalab-openenv .
```

## Deploy to Hugging Face

```bash
openenv push
```

Ensure the Space uses the **Docker** SDK and is tagged **`openenv`**.
