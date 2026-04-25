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


            "Multi‑turn code review environment for professional‑level bug fixing. "
            "The agent must inspect, test, lint, query documentation, and negotiate with "
            "a simulated (persona‑driven) author to get a fix accepted. "
            "Includes 25 bugs across 5 difficulty levels, AST‑based injection, "
            "a reward‑shaping system, and curriculum learning. "
            "Designed for RL training (PPO, DPO, or any policy‑gradient method)

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
## Deployment

```bash
openenv push
```

## License

MIT
