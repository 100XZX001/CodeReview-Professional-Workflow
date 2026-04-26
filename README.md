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

This project is a multi-turn RL environment where an agent plays the role of a senior code reviewer.
Instead of just patching code, the agent must gather evidence (`inspect`, `run_tests`, `run_linter`,
`query_docs`) and convince a simulated developer persona to accept the fix.

### Why this environment is interesting

- It combines **technical correctness** (tests/lint) with **human acceptance** (negotiation).
- It includes **25 injected bug types** across 5 difficulty levels via `RedTeam`.
- It supports both a **full reward profile** (rich shaping) and a **core reward profile**
  (minimal, baseline-friendly signal for ablations).

## Quick Start

```python
from environment import CodeReviewEnv
env = CodeReviewEnv(task="easy", reward_profile="full")
obs = env.reset()
print(obs.code_snippet)
```

## Demo Script (Non-Technical Friendly)

Use this 60-90 second flow in a demo:

1. Reset on `easy` and show the buggy snippet.
2. Take `inspect` and `run_tests` actions to show evidence gathering.
3. Ask `query_docs` once to show retrieval-assisted reasoning.
4. Propose a fix and show accepted/denied feedback from the author persona.
5. Repeat once on `harder` to show increased challenge.

Message for audience: "The agent is learning not only to fix code, but to justify and communicate the fix."

## Environment Endpoints

- `POST /reset` – reset environment (optional `task` parameter)
- `POST /step` – take an action (JSON)
- `GET /state` – get full environment state
- `GET /health` – health check
- `GET /metadata` – environment metadata
- `GET /schema` – action/observation schemas
- `POST /mcp` – minimal MCP endpoint

## Tasks
## 🐛 Bug Taxonomy (25 bugs across 5 difficulty levels)

The **RedTeam** randomly selects one bug from the current difficulty level at the start of every episode.  
Your agent must figure out what’s broken, gather evidence, and convince the simulated author – or it won’t stick.

### 🟢 Easy – Null‑Checks & Simple Logic Errors

| # | Bug ID | What’s wrong | Injection method |
|---|--------|--------------|------------------|
| 1 | `null_check` | Missing `if key in dict:` guard → KeyError | AST: remove the if‑statement |
| 2 | `simple_typo` | Misspelled variable `users` → `usres` | AST: rename variable |
| 3 | `string_index` | String index shifted by +1 | AST: change constant in index |
| 4 | `default_value` | `dict.get(key)` used without a fallback | AST: replace `dict.get(key)` with `dict[key]` |
| 5 | `empty_return` | Function returns `None` prematurely | AST: insert `return None` early |

### 🟡 Medium – Off‑By‑One, Loop Logic & Simple Arithmetic

| # | Bug ID | What’s wrong | Injection method |
|---|--------|--------------|------------------|
| 6 | `off_by_one` | `range(x)` becomes `range(1, x-1)` – skips first & last | AST: modify range arguments |
| 7 | `loop_skip` | `range(len(arr))` becomes `range(len(arr)-1)` – misses last element | AST: change range length |
| 8 | `sign_error` | `sum += item` turned into `sum -= item` | AST: swap Add / Sub |
| 9 | `swap_args` | Function arguments swapped | AST: swap first two arguments |
|10 | `uninitialised_var` | Variable used before assignment in a loop | AST: remove the assignment statement |

### 🟠 Hard – Division‑By‑Zero, Floating‑Point & Edge Cases

| # | Bug ID | What’s wrong | Injection method |
|---|--------|--------------|------------------|
|11 | `division_by_zero_empty` | Empty‑list guard removed before averaging | AST: delete `if not data:` |
|12 | `division_by_zero_zero` | Denominator check removed | AST: remove the zero‑check |
|13 | `float_precision` | True division `/` replaced by integer division `//` | AST: change Div → FloorDiv |
|14 | `abs_usage` | `abs()` call removed when comparing differences | AST: delete `abs()` wrapper |
|15 | `round_error` | `round()` placed too early, causing precision drift | AST: inject `round()` prematurely |

### 🔴 Harder – Race Conditions & Atomicity Bugs

| # | Bug ID | What’s wrong | Injection method |
|---|--------|--------------|------------------|
|16 | `missing_lock` | Shared counter incremented without a lock | Template: remove `with lock:` |
|17 | `double_lock` | Acquiring the same lock twice → deadlock risk | Template: add extra `lock.acquire()` |
|18 | `global_nonatomic` | `count = count + 1` (read‑modify‑write) instead of `+=` | AST: modify assignment node |
|19 | `thread_safe_list` | List append across threads without synchronisation | Template: remove lock from list operation |
|20 | `volatile_read` | Shared flag read outside a lock → stale value | Template: remove synchronisation block |

### ⚫ Hardest – Deadlocks, Ordering & Complex Concurrency

| # | Bug ID | What’s wrong | Injection method |
|---|--------|--------------|------------------|
|21 | `deadlock_order` | Locks acquired in opposite order in two threads | Template: swap lock order |
|22 | `nested_lock_timeout` | `lock.acquire()` without a timeout → permanent hang | Template: remove timeout logic |
|23 | `fork_join` | Thread started but not joined (`join()` missing) | AST: remove `thread.join()` |
|24 | `mutex_release` | Lock released by a thread that never acquired it | Template: incorrect release logic |
|25 | `race_on_init` | Shared resource initialised after threads have started | Template: move initialisation after `join()` |

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\POC-OF-Training-Progress/////////////////////////////////////////////////////////////////////////
Small Run → Mid Run
In an early small‑scale training run (2 PPO iterations, short episodes), the agent remained stuck in a loop of inspect and run_tests actions, never proposing a fix. The average reward stayed flat at 0.000, and the success rate was 0%. After increasing the number of PPO iterations to 15 and extending the episode length, the mid‑run training curve shows clear signs of learning: the agent began to propose fixes, leading to a positive average reward that fluctuates between +0.03 and +0.10, with occasional peaks up to +0.105. The reward curve now displays a visible upward trend, and the loss has stabilised at negative values, indicating that the policy is being optimized successfully. With further training and hyperparameter tuning, we expect the reward to continue rising and the success rate to improve significantly.
 
<img width="1200" height="600" alt="reward_curve" src="https://github.com/user-attachments/assets/14d6cb47-19a2-4367-99d5-d7ea2a94d074" />
<img width="667" height="242" alt="Screenshot 2026-04-26 071436" src="https://github.com/user-attachments/assets/6d20abd1-573b-4ffc-b487-16834174f65a" />
<img width="1200" height="600" alt="mid-reward_curve" src="https://github.com/user-attachments/assets/eadef98b-3bc4-477d-91f5-44dc398c53e5" />
<img width="1200" height="600" alt="loss_curve" src="https://github.com/user-attachments/assets/6798f90f-56f5-454e-8b90-7acd9d7b7b07" />


## License

MIT
