# training.py – Memory‑safe: Phi‑3‑mini + Expert Demos + Fast PPO (2 iterations)
import os
os.environ["TRITON_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"        # Issue #12: prevent OOM from parallel tokenization

import torch._dynamo
torch._dynamo.config.disable = True
import json
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import re
import random
import matplotlib.pyplot as plt

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

from environment import CodeReviewEnv
from redteam import BUG_DB
from models import (
    RunTests, RunLinter, Inspect,
    ProposeFix, WriteComment, AskQuestion,
    Done, Skip, QueryDocs, map_to_env as model_map_to_env
)

# ======================================================================
@dataclass
class AgentAction:
    action_type: str
    content: Optional[str] = None

def parse_action(output: str) -> AgentAction:
    try:
        data = json.loads(output)
        return AgentAction(
            action_type=data.get("action_type", "").lower(),
            content=data.get("content")
        )
    except:
        pass
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return AgentAction(
                action_type=data.get("action_type", "").lower(),
                content=data.get("content")
            )
        except:
            pass
    action_pattern = r'"action_type"\s*:\s*"(\w+)"'
    match = re.search(action_pattern, output)
    if match:
        return AgentAction(action_type=match.group(1).lower())
    output_lower = output.lower()
    if "test" in output_lower:
        return AgentAction("run_tests")
    if "lint" in output_lower:
        return AgentAction("run_linter")
    if "inspect" in output_lower:
        return AgentAction("inspect")
    if "doc" in output_lower or "documentation" in output_lower:
        return AgentAction("query_docs", "bug fix guidance")
    return AgentAction("invalid", output)

def map_to_env(action: AgentAction):
    return model_map_to_env(action.action_type, action.content)

# ======================================================================
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        max_seq_length=480,               # smaller window for memory
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=32,
        lora_dropout=0.0,
    )
    return model, tokenizer

def test_model_sanity(model, tokenizer) -> bool:
    print("\n" + "="*60)
    print("SANITY CHECK: Testing base model generation")
    print("="*60)
    test_prompt = "Hello, how are you?"
    messages = [{"role": "user", "content": test_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", max_length=256, truncation=True).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            min_new_tokens=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print(f"Prompt: {test_prompt}")
    print(f"Response: {repr(response)}")
    if len(response) == 0:
        print("❌ Model produces empty output – cannot train.")
        return False
    print("✓ Model sanity check PASSED\n")
    return True

# ======================================================================
def _expert_fix_from_context(obs) -> str:
    """
    Build a conservative fix template named `fix` (required by tests).
    Uses bug hints + code snippet patterns to create realistic fixes.
    """
    bug = (getattr(obs, "bug_description", "") or "").lower()
    code = getattr(obs, "code_snippet", "") or ""

    if "division" in bug or "average" in code.lower():
        return (
            "def fix(data):\n"
            "    if not data:\n"
            "        return 0\n"
            "    return sum(data) / len(data)"
        )

    if "operator" in bug or "sign" in bug:
        return (
            "def fix(a, b):\n"
            "    return a + b"
        )

    if "off_by_one" in bug or "loop" in bug:
        return (
            "def fix(items):\n"
            "    return len(items)"
        )

    if "null" in bug or "key" in bug or "dict" in code.lower():
        return (
            "def fix(payload):\n"
            "    users = payload.get('users', {})\n"
            "    user_id = payload.get('id')\n"
            "    return users.get(user_id)"
        )

    # Concurrency-heavy tasks (harder/hardest).
    if "race" in bug or "missing_lock" in bug or "thread_safe" in bug or "global_nonatomic" in bug:
        return (
            "import threading\n"
            "_lock = threading.Lock()\n"
            "\n"
            "def fix(counter):\n"
            "    with _lock:\n"
            "        if counter is None:\n"
            "            return 0\n"
            "        return counter + 1"
        )

    if "deadlock" in bug or "double_lock" in bug or "lock order" in bug or "nested_lock" in bug:
        return (
            "import threading\n"
            "_lock_a = threading.Lock()\n"
            "_lock_b = threading.Lock()\n"
            "\n"
            "def fix(work):\n"
            "    first, second = (_lock_a, _lock_b)\n"
            "    if id(first) > id(second):\n"
            "        first, second = second, first\n"
            "    with first:\n"
            "        with second:\n"
            "            return work() if callable(work) else work"
        )

    if "fork_join" in bug or "join" in bug:
        return (
            "import threading\n"
            "\n"
            "def fix(worker):\n"
            "    t = threading.Thread(target=worker)\n"
            "    t.start()\n"
            "    t.join()\n"
            "    return True"
        )

    # Generic safe fallback keeps the RL pipeline alive for unknown bugs.
    return (
        "def fix(data):\n"
        "    if data is None:\n"
        "        return None\n"
        "    return data"
    )


def _expert_supervised_policy(obs) -> str:
    """
    Real workflow policy:
    inspect -> tests/linter -> docs -> fix -> negotiate -> done.
    """
    author_msg = (getattr(obs, "author_response", "") or "").lower()
    tool_output = (getattr(obs, "last_tool_output", "") or "").lower()

    if not getattr(obs, "tests_run", False):
        if "inspect" not in tool_output:
            return '{"action_type": "inspect"}'
        return '{"action_type": "run_tests"}'

    if not getattr(obs, "linter_run", False):
        return '{"action_type": "run_linter"}'

    if not getattr(obs, "docs_queried", False):
        return '{"action_type": "query_docs", "content": "python bug fixing best practices for edge cases and null safety"}'

    # Use docs again on hard tasks when evidence is still weak.
    if getattr(obs, "current_test_score", 0.0) < 0.6 and getattr(obs, "step", 0) >= 3:
        bug_hint = (getattr(obs, "bug_description", "") or "concurrency bug").replace('"', "'")
        return json.dumps(
            {
                "action_type": "query_docs",
                "content": f"python {bug_hint} lock ordering race condition mitigation patterns",
            }
        )

    # If test quality is poor, propose a concrete fix.
    if getattr(obs, "current_test_score", 0.0) < 0.95:
        fix_code = _expert_fix_from_context(obs)
        return json.dumps({"action_type": "fix", "content": fix_code})

    # If author is still unconvinced, provide causal explanation.
    if author_msg and ("not convinced" in author_msg or "explain" in author_msg or "brief" in author_msg):
        return (
            '{"action_type": "comment", "content": "This fix works because it handles the failing edge case directly, '
            'keeps behavior deterministic, and aligns with the observed test and lint feedback. '
            'The change is intentionally small to reduce regression risk."}'
        )

    # If negotiation is strong enough and quality is good, terminate.
    conf = float(getattr(obs, "author_confidence", 0.0))
    threshold = float(getattr(obs, "author_threshold", 0.5))
    score = float(getattr(obs, "current_test_score", 0.0))
    if conf >= threshold and score >= 0.8:
        return '{"action_type": "done"}'

    # Nudge conversation forward when tests are okay but acceptance is pending.
    return (
        '{"action_type": "question", "content": "Would you like a quick walkthrough of a failing scenario, the root cause, and how the fix prevents regressions?"}'
    )

# ======================================================================
def supervised_warmup(model, tokenizer, env, n_episodes=16, epochs=1, max_steps=8):
    print("\n" + "="*60)
    print("SUPERVISED WARM-UP: Real environment demonstrations")
    print("="*60)

    examples = []
    tasks = ["easy", "medium", "hard", "harder", "hardest"]
    for ep in range(n_episodes):
        task = random.choice(tasks)
        env.set_task(task)
        obs = env.reset()
        history = []
        done = False

        steps = 0
        while not done and steps < max_steps:
            prompt = build_prompt(obs, history)
            action_text = _expert_supervised_policy(obs)
            action = parse_action(action_text)
            env_action = map_to_env(action)
            next_obs, _, done, _ = env.step(env_action)

            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": action_text},
            ]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            examples.append({"text": full_text})

            history.append(f"Agent: {action_text}")
            history.append(f"Env: {next_obs.last_tool_output}")
            history = history[-8:]
            obs = next_obs
            steps += 1

        print(f"Supervised episode {ep+1}: task={task}, steps={steps}, done={done}")

    if not examples:
        print("No supervised examples generated; skipping warm-up.")
        return

    dataset = Dataset.from_list(examples)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=480,
        args=TrainingArguments(
            output_dir="warmup_output",
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            logging_steps=50,
            save_strategy="no",
            bf16=True,
        ),
    )
    print(f"Training on {len(examples)} real env examples for {epochs} epochs...")
    trainer.train()
    print("✓ Supervised warm-up (real env) complete\n")
    torch.cuda.empty_cache()

# ======================================================================
def generate_action_with_logprob(prompt, model, tokenizer, temperature=0.0, max_retries=2):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", max_length=480, truncation=True).to("cuda")
    
    for attempt in range(max_retries):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=(temperature > 0),
                temperature=max(temperature, 0.01) if temperature > 0 else 1.0,
                min_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        action_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        logprobs = []
        for idx, token_id in enumerate(generated_ids):
            if idx < len(outputs.scores):
                token_logits = outputs.scores[idx][0]
                token_logprob = F.log_softmax(token_logits, dim=-1)[token_id].item()
                logprobs.append(token_logprob)
        total_logprob = sum(logprobs) if logprobs else -100.0
        
        if not action_text:
            fallback_actions = [
                '{"action_type": "run_tests"}',
                '{"action_type": "run_linter"}',
                '{"action_type": "inspect"}',
                '{"action_type": "skip"}',
            ]
            action_text = random.choice(fallback_actions)
            total_logprob = -50.0
            print(f"[WARN] Empty generation → using fallback: {action_text}")
            return action_text, total_logprob
        
        try:
            json.loads(action_text)
            return action_text, total_logprob
        except:
            if attempt == max_retries - 1:
                return '{"action_type":"skip"}', -100.0
            continue
    return '{"action_type":"skip"}', -100.0

# ======================================================================
def build_prompt(obs, history_lines: List[str]) -> str:
    author_msg = getattr(obs, "author_response", "") or ""
    tool_output = getattr(obs, "last_tool_output", "") or ""
    author_personality = getattr(obs, "author_personality", "defensive")
    
    prompt = f"""You are an AI code review agent. Your goal is to convince a simulated human developer to accept your proposed fix and name your proposed fix function fix.

The developer has a **{author_personality}** personality and will only accept if you provide solid evidence:
- Tests pass (high pass ratio)
- Lint is clean (zero errors)
- Documentation or references are provided
- Your reasoning is clear, uses words like "because" or "therefore", and is detailed (over 30 words if needed)

Workflow:
1. Use `inspect` to understand the code.
2. Use `run_tests` and `run_linter` to gather evidence.
3. Use `query_docs` when you need references or language-specific guidance.
4. Propose a fix (`fix`) and explain why it works (`comment` or `question`).
5. If the developer pushes back, read their response carefully and address their specific concern.
6. Once convinced, use `done` to finish.

Code:
{obs.code_snippet}

Author says:
{author_msg if author_msg else "(no response yet – start with inspection)"}

Last tool output:
{tool_output if tool_output else "(none)"}

Available actions:
run_tests, run_linter, inspect, query_docs, fix, comment, question, done

Respond ONLY in JSON:
{{"action_type": "...", "content": "..."}}"""
    
    if history_lines:
        history = "\n".join(history_lines[-6:])
        prompt += f"\n\nPrevious steps:\n{history}"
    return prompt

# ======================================================================
@dataclass
class Trajectory:
    states: List[str]
    actions: List[str]
    rewards: List[float]
    logprobs: List[float]
    dones: List[bool]
    def __len__(self): return len(self.states)

def collect_trajectory(env, model, tokenizer, max_steps=6, temperature=0.0):
    obs = env.reset()
    history_lines = []
    states, actions, rewards, logprobs, dones = [], [], [], [], []
    for step in range(max_steps):
        prompt = build_prompt(obs, history_lines)
        states.append(prompt)
        action_text, logprob = generate_action_with_logprob(prompt, model, tokenizer, temperature)
        actions.append(action_text)
        logprobs.append(logprob)
        action = parse_action(action_text)
        env_action = map_to_env(action)
        next_obs, reward, done, _ = env.step(env_action)
        rewards.append(reward.value)
        dones.append(done)
        history_lines.append(f"Agent: {action_text}")
        history_lines.append(f"Env: {next_obs.last_tool_output}")
        obs = next_obs
        if done: break
    return Trajectory(states, actions, rewards, logprobs, dones)

def collect_trajectories(env, model, tokenizer, n_trajectories, max_steps=6,
                         task_levels=None, task_weights=None):
    if task_levels is None:
        task_levels = list(BUG_DB.keys())
    if task_weights is not None and len(task_weights) != len(task_levels):
        raise ValueError("task_weights must match task_levels length")
    if task_weights is not None and sum(task_weights) <= 0:
        raise ValueError("task_weights must have a positive total")
    trajectories = []
    for i in range(n_trajectories):
        sampled_task = random.choices(task_levels, weights=task_weights, k=1)[0]
        env.set_task(sampled_task)
        traj = collect_trajectory(env, model, tokenizer, max_steps)
        total_reward = sum(traj.rewards)
        print(f"Trajectory {i+1}/{n_trajectories}: task={sampled_task}, steps={len(traj)}, reward={total_reward:.3f}")
        trajectories.append(traj)
    return trajectories

def compute_returns_and_advantages(rewards, dones, gamma=0.99, standardize=True):
    """
    Compute discounted returns and REINFORCE-style baseline advantages.
    Advantages are centered and optionally standardised.
    """
    n = len(rewards)
    returns = [0.0]*n
    running = 0.0
    for t in reversed(range(n)):
        if dones[t]: running = 0.0
        running = rewards[t] + gamma * running
        returns[t] = running
    if standardize:
        advantages = np.array(returns) - np.mean(returns)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages / adv_std).tolist()
    else:
        advantages = returns.copy()
    return advantages, returns

def ppo_update(trajectories, model, tokenizer, optimizer, n_epochs=1, clip_epsilon=0.2,
               entropy_coef=0.01, gamma=0.99):
    model.train()
    all_states, all_actions, all_old_logprobs, all_advantages = [], [], [], []
    for traj in trajectories:
        advantages, _ = compute_returns_and_advantages(traj.rewards, traj.dones, gamma=gamma, standardize=True)
        all_states.extend(traj.states)
        all_actions.extend(traj.actions)
        all_old_logprobs.extend(traj.logprobs)
        all_advantages.extend(advantages)
    n_samples = len(all_states)
    total_loss, total_policy_loss, total_entropy, n_updates = 0.0, 0.0, 0.0, 0
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        for i in indices:
            state = all_states[i]
            action = all_actions[i]
            old_logprob = all_old_logprobs[i]
            advantage = all_advantages[i]
            messages = [{"role": "user", "content": state}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_text = formatted + action
            inputs = tokenizer(full_text, return_tensors="pt", max_length=480, truncation=True).to("cuda")
            outputs = model(**inputs)
            logits = outputs.logits
            action_ids = tokenizer.encode(action, add_special_tokens=False)
            prefix_ids = tokenizer.encode(formatted, add_special_tokens=False)
            action_start = len(prefix_ids)
            logprobs = []
            entropy = 0.0
            for idx, token_id in enumerate(action_ids):
                position = action_start + idx - 1
                if 0 <= position < logits.shape[1]:
                    token_logits = logits[0, position]
                    log_probs = F.log_softmax(token_logits, dim=-1)
                    token_logprob = log_probs[token_id]
                    logprobs.append(token_logprob)
                    probs = F.softmax(token_logits, dim=-1)
                    entropy += -(probs * log_probs).sum()
            if not logprobs: continue
            new_logprob = sum(logprobs)
            avg_entropy = entropy / len(logprobs) if logprobs else 0.0
            ratio = torch.exp(new_logprob - old_logprob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
            policy_loss = -torch.min(surr1, surr2)
            loss = policy_loss - entropy_coef * avg_entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_entropy += avg_entropy.item()
            n_updates += 1
    torch.cuda.empty_cache()
    return {"loss": total_loss / n_updates if n_updates else 0.0,
            "policy_loss": total_policy_loss / n_updates if n_updates else 0.0,
            "entropy": total_entropy / n_updates if n_updates else 0.0}

def evaluate_policy(env, model, tokenizer, n_episodes=3, max_steps=6,
                    task_levels=None, verbose=False):
    """Evaluate the current policy across task levels. Returns metrics + optional traces."""
    model.eval()
    if task_levels is None:
        task_levels = list(BUG_DB.keys())
    total_rewards = []
    traces = []  # human-readable behavior logs
    for ep in range(n_episodes):
        task = task_levels[ep % len(task_levels)]
        env.set_task(task)
        traj = collect_trajectory(env, model, tokenizer, max_steps, temperature=0.0)
        ep_reward = sum(traj.rewards)
        total_rewards.append(ep_reward)
        if verbose:
            actions_taken = []
            for a in traj.actions:
                try:
                    actions_taken.append(json.loads(a).get("action_type", "?"))
                except Exception:
                    actions_taken.append("?")
            traces.append({
                "task": task,
                "reward": round(ep_reward, 4),
                "steps": len(traj),
                "actions": actions_taken,
            })
    return {
        "avg_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "min_reward": float(np.min(total_rewards)),
        "max_reward": float(np.max(total_rewards)),
        "traces": traces,
    }

# ======================================================================
# MANUAL WARM-UP (no SFTTrainer → no multiprocessing OOM)
# ======================================================================
def json_warmup(model, tokenizer, json_path="training_data.json",
                n_episodes=20, epochs=2, lr=2e-5):
    """
    Supervised warm-up from pre-generated expert demonstrations.
    Uses raw cross-entropy on action tokens with manual gradient steps.
    NO SFTTrainer, NO multiprocessing – runs safely on any GPU.
    """
    print("\n" + "="*60)
    print("SUPERVISED WARM-UP: training_data.json (manual cross-entropy)")
    print("="*60)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Each episode = 7 steps. Select n_episodes worth.
    steps_per_episode = 7
    max_examples = n_episodes * steps_per_episode
    if max_examples < len(data):
        data = data[:max_examples]

    print(f"  {len(data)} examples ({len(data)//steps_per_episode} episodes), "
          f"{epochs} epoch(s), lr={lr}")

    model.train()
    warmup_opt = AdamW(model.parameters(), lr=lr)
    warmup_losses = []   # per-epoch avg loss

    for epoch in range(epochs):
        random.shuffle(data)
        epoch_loss = 0.0
        n_valid = 0

        for i, example in enumerate(data):
            prompt = example["prompt"]
            action = example["action"]

            # ---- tokenize full sequence (prompt + action) ----
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": action},
            ]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer(full_text, return_tensors="pt",
                               max_length=480, truncation=True).to("cuda")

            # ---- find where the action tokens start ----
            prompt_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer.encode(prompt_only, add_special_tokens=False)
            prompt_len = len(prompt_ids)

            total_len = inputs.input_ids.shape[1]
            if prompt_len >= total_len:
                continue  # prompt was truncated away, skip

            # ---- cross-entropy on action tokens only ----
            outputs = model(**inputs)
            logits = outputs.logits

            # next-token prediction: logits[t] predicts token[t+1]
            shift_logits = logits[0, prompt_len - 1 : total_len - 1]
            shift_labels = inputs.input_ids[0, prompt_len : total_len]

            min_len = min(shift_logits.shape[0], shift_labels.shape[0])
            if min_len == 0:
                continue

            loss = F.cross_entropy(shift_logits[:min_len], shift_labels[:min_len])

            warmup_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            warmup_opt.step()

            epoch_loss += loss.item()
            n_valid += 1

            if (i + 1) % 25 == 0:
                avg = epoch_loss / n_valid
                print(f"    epoch {epoch+1}  step {i+1:3d}/{len(data)}  "
                      f"running_loss={avg:.4f}")

        avg_loss = epoch_loss / max(n_valid, 1)
        warmup_losses.append(avg_loss)
        print(f"  Epoch {epoch+1} done: avg_loss={avg_loss:.4f}  "
              f"({n_valid} valid examples)")

    torch.cuda.empty_cache()
    print(f"✓ Warm-up complete.  Loss: "
          f"{' → '.join(f'{l:.4f}' for l in warmup_losses)}\n")
    return warmup_losses


# ======================================================================
# MAIN TRAINING PIPELINE
# ======================================================================
def train_ppo():
    # --- Hyperparameters ---
    n_iterations = 8            # enough for a clear upward trend
    trajectories_per_iter = 4   # on-policy data per iteration
    n_epochs = 1
    max_steps = 6
    learning_rate = 3e-5
    clip_epsilon = 0.2
    entropy_coef = 0.01
    gamma = 0.99

    # --- Pre-load embedder before LLM (Issue #13) ---
    from rltool import ToolBox
    print("Pre-loading sentence-transformer embedder...")
    ToolBox._get_embedder()
    print("✓ Embedder ready")

    # --- Load model ---
    print("Loading model...")
    model, tokenizer = load_model()
    if not test_model_sanity(model, tokenizer):
        return
    env = CodeReviewEnv()
    task_levels = list(BUG_DB.keys())

    # ==================================================================
    # PHASE 0: BASELINE (untrained policy)
    # ==================================================================
    print("\n" + "="*60)
    print("PHASE 0 – BASELINE EVALUATION (untrained)")
    print("="*60)
    baseline = evaluate_policy(env, model, tokenizer, n_episodes=5,
                               max_steps=max_steps, task_levels=task_levels,
                               verbose=True)
    baseline_reward = baseline["avg_reward"]
    print(f"Baseline avg reward: {baseline_reward:.4f}  "
          f"(min={baseline['min_reward']:.4f}, max={baseline['max_reward']:.4f})")
    print("Baseline behavior:")
    for t in baseline["traces"]:
        print(f"  task={t['task']:8s}  reward={t['reward']:+.4f}  "
              f"steps={t['steps']}  actions={t['actions']}")

    # ==================================================================
    # PHASE 1: SUPERVISED WARM-UP (expert demos, manual CE)
    # ==================================================================
    warmup_losses = json_warmup(
        model, tokenizer,
        json_path="training_data.json",
        n_episodes=20,  # 140 examples (20 × 7 steps)
        epochs=2,
        lr=2e-5,
    )

    # Post-warmup evaluation
    print("="*60)
    print("POST WARM-UP EVALUATION")
    print("="*60)
    post_warmup = evaluate_policy(env, model, tokenizer, n_episodes=5,
                                  max_steps=max_steps, task_levels=task_levels,
                                  verbose=True)
    warmup_reward = post_warmup["avg_reward"]
    print(f"Post-warmup avg reward: {warmup_reward:.4f}  "
          f"(Δ vs baseline: {warmup_reward - baseline_reward:+.4f})")
    print("Post-warmup behavior:")
    for t in post_warmup["traces"]:
        print(f"  task={t['task']:8s}  reward={t['reward']:+.4f}  "
              f"steps={t['steps']}  actions={t['actions']}")

    # ==================================================================
    # PHASE 2: TRUE RL – PPO (on-policy, real environment interaction)
    # ==================================================================
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    print(f"\n{'='*60}")
    print(f"PHASE 2 – PPO TRAINING: {n_iterations} iterations × "
          f"{trajectories_per_iter} trajectories (true RL)")
    print(f"{'='*60}\n")

    reward_history = []
    eval_history = []
    loss_history = []
    policy_loss_history = []
    entropy_history = []

    for iteration in range(n_iterations):
        print(f"\n--- PPO Iteration {iteration + 1}/{n_iterations} ---")

        # Collect on-policy trajectories from REAL environment
        trajectories = collect_trajectories(
            env, model, tokenizer, trajectories_per_iter, max_steps,
            task_levels=task_levels, task_weights=None
        )
        avg_reward = float(np.mean([sum(t.rewards) for t in trajectories]))
        reward_history.append(avg_reward)
        print(f"  Collect  avg reward: {avg_reward:+.4f}")

        # PPO policy gradient update
        metrics = ppo_update(
            trajectories, model, tokenizer, optimizer,
            n_epochs=n_epochs, clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef, gamma=gamma
        )
        loss_history.append(float(metrics["loss"]))
        policy_loss_history.append(float(metrics["policy_loss"]))
        entropy_history.append(float(metrics["entropy"]))
        print(f"  Update   loss={metrics['loss']:.4f}  "
              f"policy={metrics['policy_loss']:.4f}  "
              f"entropy={metrics['entropy']:.4f}")

        # Evaluate greedy policy after update
        eval_m = evaluate_policy(env, model, tokenizer, n_episodes=3,
                                 max_steps=max_steps, task_levels=task_levels,
                                 verbose=False)
        eval_history.append(eval_m["avg_reward"])
        delta = eval_m["avg_reward"] - baseline_reward
        print(f"  Eval     avg reward: {eval_m['avg_reward']:+.4f}  "
              f"(Δ baseline: {delta:+.4f})")

    # ==================================================================
    # PHASE 3: FINAL EVALUATION (proof of learning)
    # ==================================================================
    print("\n" + "="*60)
    print("PHASE 3 – FINAL EVALUATION (after all training)")
    print("="*60)
    final = evaluate_policy(env, model, tokenizer, n_episodes=5,
                            max_steps=max_steps, task_levels=task_levels,
                            verbose=True)
    print(f"Final avg reward: {final['avg_reward']:.4f}  "
          f"(min={final['min_reward']:.4f}, max={final['max_reward']:.4f})")
    print("Final behavior:")
    for t in final["traces"]:
        print(f"  task={t['task']:8s}  reward={t['reward']:+.4f}  "
              f"steps={t['steps']}  actions={t['actions']}")

    total_improvement = final["avg_reward"] - baseline_reward
    ppo_improvement = final["avg_reward"] - warmup_reward
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"  Baseline reward:    {baseline_reward:+.4f}")
    print(f"  Post-warmup reward: {warmup_reward:+.4f}  "
          f"(warmup Δ: {warmup_reward - baseline_reward:+.4f})")
    print(f"  Final reward:       {final['avg_reward']:+.4f}  "
          f"(PPO Δ: {ppo_improvement:+.4f})")
    print(f"  Total improvement:  {total_improvement:+.4f}")
    print(f"  Reward trend (PPO): {' → '.join(f'{r:+.3f}' for r in reward_history)}")
    print(f"  Loss trend (PPO):   {' → '.join(f'{l:.4f}' for l in loss_history)}")
    if total_improvement > 0:
        print(f"  ✓ Agent IMPROVED by {total_improvement:+.4f}")
    else:
        print(f"  ✗ No overall improvement detected")
    print(f"{'='*60}")

    # ==================================================================
    # PLOTS
    # ==================================================================
    iters = list(range(1, n_iterations + 1))

    # --- 1. Warm-up loss curve ---
    if warmup_losses:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(warmup_losses) + 1), warmup_losses,
                marker="o", linewidth=2, color="tab:purple")
        ax.set_title("Warm-up Loss (supervised, per epoch)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("warmup_loss.png", dpi=150)
        plt.close(fig)

    # --- 2. PPO reward curve ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, reward_history, marker="o", linewidth=2,
            label="Collect reward", color="tab:blue")
    ax.plot(iters, eval_history, marker="s", linewidth=2, linestyle="--",
            label="Eval reward", color="tab:green")
    ax.axhline(y=baseline_reward, color="tab:gray", linestyle=":",
               linewidth=1.5, label=f"Baseline ({baseline_reward:+.3f})")
    ax.axhline(y=warmup_reward, color="tab:purple", linestyle=":",
               linewidth=1.5, label=f"Post-warmup ({warmup_reward:+.3f})")
    ax.set_title("PPO Reward per Iteration", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Reward")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("reward_curve.png", dpi=150)
    plt.close(fig)

    # --- 3. PPO loss curve ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, loss_history, marker="o", linewidth=2,
            label="Total loss", color="tab:red")
    ax.plot(iters, policy_loss_history, marker="^", linewidth=2, linestyle="--",
            label="Policy loss", color="tab:orange")
    ax.set_title("PPO Loss per Iteration", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("loss_curve.png", dpi=150)
    plt.close(fig)

    # --- 4. Combined 3-panel summary ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: warm-up loss
    if warmup_losses:
        axes[0].plot(range(1, len(warmup_losses) + 1), warmup_losses,
                     marker="o", linewidth=2, color="tab:purple")
        axes[0].set_title("A. Warm-up Loss ↓")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("CE Loss")
        axes[0].grid(alpha=0.3)

    # Panel B: PPO reward
    axes[1].plot(iters, reward_history, marker="o", linewidth=2,
                 color="tab:blue", label="Collect")
    axes[1].plot(iters, eval_history, marker="s", linewidth=2,
                 linestyle="--", color="tab:green", label="Eval")
    axes[1].axhline(y=baseline_reward, color="tab:gray", linestyle=":",
                    linewidth=1.5, label="Baseline")
    axes[1].axhline(y=warmup_reward, color="tab:purple", linestyle=":",
                    linewidth=1.5, label="Post-warmup")
    axes[1].set_title("B. PPO Reward ↑")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Avg Reward")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    # Panel C: PPO loss
    axes[2].plot(iters, loss_history, marker="o", linewidth=2,
                 color="tab:red", label="Total")
    axes[2].plot(iters, policy_loss_history, marker="^", linewidth=2,
                 linestyle="--", color="tab:orange", label="Policy")
    axes[2].set_title("C. PPO Loss ↓")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Loss")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)

    fig.suptitle("Code Review Agent – Full Training Evidence",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("training_summary.png", dpi=150)
    plt.close(fig)

    print("Plots saved: warmup_loss.png, reward_curve.png, "
          "loss_curve.png, training_summary.png")
    print("="*60)

if __name__ == "__main__":
    train_ppo()