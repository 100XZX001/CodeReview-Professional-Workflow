---
title: CodeReview Training
emoji: "🤖"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# CodeReview Professional Workflow

`CodeReview Professional Workflow` is an OpenEnv environment for training code-fixing agents on realistic review loops instead of one-shot coding tasks. The agent has to inspect buggy code, run tests, lint the patch, query docs, and persuade a simulated author before the episode is considered solved.

## Quick links

| Artifact | Link |
| --- | --- |
| Hugging Face Space | [100XZX001/CodeReview-Professional-Workflow](https://huggingface.co/spaces/100XZX001/CodeReview-Professional-Workflow) |
| Colab-ready training notebook | [notebooks/code_review_unsloth_training.ipynb](notebooks/code_review_unsloth_training.ipynb) |
| Local training script | [training.py](training.py) |
| OpenEnv manifest | [openenv.yaml](openenv.yaml) |
| Submission slide deck | [submission_assets/code_review_openenv_submission.pptx](submission_assets/code_review_openenv_submission.pptx) |
| Training artifacts folder | [outputs/README.md](outputs/README.md) |

## Why this environment

Most code agents are evaluated on static patch generation. Real review work is messier:

- you have to diagnose the failure mode before patching
- you often need tool feedback before you know whether the fix is safe
- you may need to explain the fix to another developer before it is accepted

This environment turns that workflow into a multi-step RL setting with dense rewards and stateful interaction.

## How the environment works

Each episode samples one injected bug from five difficulty bands:

1. `easy`: null checks, missing defaults, simple indexing mistakes
2. `medium`: off-by-one and wrong-operator bugs
3. `hard`: numerical safety failures like divide-by-zero
4. `harder`: concurrency issues like missing locks
5. `hardest`: deadlock and coordination mistakes

The agent can take actions such as:

- `inspect`
- `run_tests`
- `run_linter`
- `query_docs`
- `fix`
- `comment`
- `question`
- `done`

Rewards combine test delta, lint delta, tool usage, exploration behavior, step penalties, and terminal success. The observation includes the current code, latest tool output, previous scores, author confidence, progress counters, and recent action history.

## OpenEnv-first setup

This repo is structured as an OpenEnv environment rather than a custom one-off app:

- the environment metadata lives in [openenv.yaml](openenv.yaml)
- the Space is configured as a Docker-based OpenEnv deployment
- runtime dependencies are kept lightweight for the Space build
- training-only packages live separately so judges can run the environment without pulling the full training stack

The project now targets `openenv-core>=0.2.3`.

## Training

The main training entrypoint is [training.py](training.py), which uses Unsloth plus a PPO-style loop over real environment interaction. For judges who want a rerunnable workflow, the repo also includes a Colab-ready notebook:

- [notebooks/code_review_unsloth_training.ipynb](notebooks/code_review_unsloth_training.ipynb)

### Install locally

```bash
pip install -e .
pip install -r requirements-training.txt
```

### Run training

```bash
python training.py
```

The training run writes the evidence plots in the working directory:

- `warmup_loss.png`
- `reward_curve.png`
- `loss_curve.png`
- `training_summary.png`

For submission hygiene, copy a real run into `outputs/<run-name>/` and link that folder from this README before final judging.

## Results and evidence

The expected evidence bundle for a real training run is:

- warm-up loss curve
- PPO reward curve
- PPO loss curve
- combined summary panel

Use [outputs/README.md](outputs/README.md) as the landing page for committed run artifacts.

## Submission materials

This repo is set up so every judge-facing artifact can be reached from this README:

- environment Space: [100XZX001/CodeReview-Professional-Workflow](https://huggingface.co/spaces/100XZX001/CodeReview-Professional-Workflow)
- training notebook: [notebooks/code_review_unsloth_training.ipynb](notebooks/code_review_unsloth_training.ipynb)
- slide deck: [submission_assets/code_review_openenv_submission.pptx](submission_assets/code_review_openenv_submission.pptx)
- evidence folder: [outputs/README.md](outputs/README.md)

No large video files are stored in the repo; any future video or blog submission should be linked by URL from this README.
