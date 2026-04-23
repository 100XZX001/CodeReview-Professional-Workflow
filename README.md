---
title: Code Review Professional Workflow
emoji: 🔥
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Code Review Professional Workflow

Multi‑turn code review environment for professional tasks. Agent must inspect, test, lint, query docs, and negotiate with a simulated author to fix injected bugs. Supports DPO training on full trajectories.

## Quick Start

```python
from environment import CodeReviewEnv
env = CodeReviewEnv()
obs = env.reset()
print(obs.code_snippet)
```

## Environment Endpoints

- `POST /reset` – reset environment (optional `task` parameter)
- `POST /step` – take an action (JSON)
- `GET /state` – get full environment state
- `GET /health` – health check
- `GET /metadata` – environment metadata
- `GET /schema` – action/observation schemas
- `POST /mcp` – minimal MCP endpoint

## Tasks

| Difficulty | Bug Type |
|------------|----------|
| easy | Missing null check |
| medium | Inefficient loop |
| hard | Division by zero |
| harder | Race condition (missing lock) |
| hardest | Potential deadlock |

## Deployment

```bash
openenv push
```

## License

MIT
```