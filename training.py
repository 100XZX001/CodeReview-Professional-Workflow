# training.py 
import torch._dynamo
torch._dynamo.config.disable = True
import json
import os
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

# Import your environment and actions (unchanged)
from environment import CodeReviewEnv
from redteam import BUG_DB
from models import (
    RunTests, RunLinter, Inspect,
    ProposeFix, WriteComment, AskQuestion,
    Done, Skip , QueryDocs 
)

# ======================================================================
# 1. ACTION PARSING (improved with fallback)
# ======================================================================
@dataclass
class AgentAction:
    action_type: str
    content: Optional[str] = None

def parse_action(output: str) -> AgentAction:
    """Robust JSON parsing with regex fallback and keyword detection."""
    # Try strict JSON first
    try:
        data = json.loads(output)
        return AgentAction(
            action_type=data.get("action_type", "").lower(),
            content=data.get("content")
        )
    except:
        pass
    
    # Try to extract JSON from markdown blocks
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
    
    # Try to find "action_type" field with regex
    action_pattern = r'"action_type"\s*:\s*"(\w+)"'
    match = re.search(action_pattern, output)
    if match:
        return AgentAction(action_type=match.group(1).lower())
    
    # Keyword detection as last resort
    output_lower = output.lower()
    if "test" in output_lower:
        return AgentAction("run_tests")
    if "lint" in output_lower:
        return AgentAction("run_linter")
    if "inspect" in output_lower:
        return AgentAction("inspect")
    if "doc" in output_lower or "documentation" in output_lower:
        # Bridge natural language mentions to rltool-backed retrieval action.
        return AgentAction("query_docs", "bug fix guidance")
    
    return AgentAction("invalid", output)

def map_to_env(action: AgentAction):
    if action.action_type == "run_tests":
        return RunTests()
    elif action.action_type == "run_linter":
        return RunLinter()
    elif action.action_type == "inspect":
        return Inspect()
    elif action.action_type == "fix":
        return ProposeFix(fix_code=action.content or "")
    elif action.action_type == "comment":
        return WriteComment(comment_text=action.content or "")
    elif action.action_type == "question":
        return AskQuestion(question=action.content or "")
    elif action.action_type == "query_docs":               # <-- new
        return QueryDocs(query_topic=action.content or "")
    elif action.action_type == "done":
        return Done()
    else:
        return Skip()

# ======================================================================
# 2. MODEL SETUP (stabilised LoRA)
# ======================================================================
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-2-2b-it-bnb-4bit",
        max_seq_length=768,
        load_in_4bit=True,
    )
    # FIXED: Lower rank (16), dropout=0 for stability
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                     # was 64 → causes collapse
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=32,            # adjusted for r=16
        lora_dropout=0.0,         # dropout can cause empty outputs
    )
    # Ensure tokenizer has correct chat template for Gemma-2
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n<start_of_turn>model\n{% elif message['role'] == 'assistant' %}{{ message['content'] }}<end_of_turn>\n{% endif %}{% endfor %}"
    return model, tokenizer

# ======================================================================
# 3. MODEL SANITY CHECK (new – ensures model can generate text)
# ======================================================================
def test_model_sanity(model, tokenizer) -> bool:
    print("\n" + "="*60)
    print("SANITY CHECK: Testing base model generation")
    print("="*60)
    test_prompt = "Hello, how are you?"
    messages = [{"role": "user", "content": test_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", max_length=768, truncation=True).to("cuda")
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
# 4. SUPERVISED WARM-UP (teaches JSON output)
# ======================================================================
def supervised_warmup(model, tokenizer, n_examples=500, epochs=8):
    print("\n" + "="*60)
    print("SUPERVISED WARM-UP: Teaching JSON format")
    print("="*60)
    
    examples = []
    action_templates = [
        '{"action_type": "run_tests"}',
        '{"action_type": "run_linter"}',
        '{"action_type": "inspect"}',
        '{"action_type": "fix", "content": "def corrected():\n    pass"}',
        '{"action_type": "comment", "content": "This looks good."}',
        '{"action_type": "question", "content": "Why is this variable used?"}',
        '{"action_type": "query_docs", "content": "KeyError"}',
        '{"action_type": "done"}',
    ]
    
    for i in range(n_examples):
        code = f"def example_{i}():\n    return {i % 10}"
        last_outputs = [
            "Tests passed: 2/3",
            "Linter found 1 error",
            "Inspection complete",
            "No previous action",
        ]
        last_output = random.choice(last_outputs)
        # Use same prompt structure as build_prompt
        prompt = f"""You are an AI code review agent. Your goal is to convince a simulated human developer to accept your proposed fix.

The developer has a **defensive** personality and will only accept if you provide solid evidence:
- Tests pass (high pass ratio)
- Lint is clean (zero errors)
- Documentation or references are provided
- Your reasoning is clear, uses words like "because" or "therefore", and is detailed (over 30 words if needed)

Workflow:
1. Use `inspect` to understand the code.
2. Use `run_tests` and `run_linter` to gather evidence.
3. Propose a fix (`fix`) and explain why it works (`comment` or `question`).
4. If the developer pushes back, read their response carefully and address their specific concern.
5. Once convinced, use `done` to finish.

Code:
{code}

Author says:
(no response yet – start with inspection)

Last tool output:
{last_output}

Available actions:
run_tests, run_linter, inspect, fix, comment, question, done, query_docs

Respond ONLY in JSON:
{{"action_type": "...", "content": "..."}}"""
        
        action_json = random.choice(action_templates)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": action_json}
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        examples.append({"text": full_text})
    
    dataset = Dataset.from_list(examples)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=TrainingArguments(
            output_dir="warmup_output",
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            logging_steps=50,
            save_strategy="no",
            bf16=True,
        ),
    )
    print(f"Training on {n_examples} examples for {epochs} epochs...")
    trainer.train()
    print("✓ Warm-up complete\n")
    torch.cuda.empty_cache() 
# ======================================================================
# 5. ACTION GENERATION WITH LOGPROB TRACKING (fixed)
# ======================================================================
def generate_action_with_logprob(
    prompt: str, 
    model, 
    tokenizer, 
    temperature: float = 0.0,   # changed: greedy by default for stability
    max_retries: int = 2
) -> Tuple[str, float]:
    """Generate action using correct chat template, with fallback."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    
    for attempt in range(max_retries):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=(temperature > 0),
                temperature=max(temperature, 0.01) if temperature > 0 else 1.0,
                min_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        action_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Compute logprob
        logprobs = []
        for idx, token_id in enumerate(generated_ids):
            if idx < len(outputs.scores):
                token_logits = outputs.scores[idx][0]
                token_logprob = F.log_softmax(token_logits, dim=-1)[token_id].item()
                logprobs.append(token_logprob)
        total_logprob = sum(logprobs) if logprobs else -100.0
        
        # If empty, use fallback
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
        
        # Validate JSON
        try:
            json.loads(action_text)
            return action_text, total_logprob
        except:
            if attempt == max_retries - 1:
                return '{"action_type":"skip"}', -100.0
            continue
    
    return '{"action_type":"skip"}', -100.0

# ======================================================================
# 6. PROMPT BUILDER (unchanged – exactly as you wrote)
# ======================================================================
def build_prompt(obs, history_lines: List[str]) -> str:
    author_msg = getattr(obs, "author_response", "") or ""
    tool_output = getattr(obs, "last_tool_output", "") or ""
    
    # Personality hint (optional but helpful)
    author_personality = getattr(obs, "author_personality", "defensive")  # e.g., from env
    
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
# 7. TRAJECTORY STORAGE (unchanged)
# ======================================================================
@dataclass
class Trajectory:
    states: List[str]
    actions: List[str]
    rewards: List[float]
    logprobs: List[float]
    dones: List[bool]
    
    def __len__(self):
        return len(self.states)
    
    def to_dict(self):
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "logprobs": self.logprobs,
            "dones": self.dones,
        }

# ======================================================================
# 8. ROLLOUT COLLECTION (uses fixed generate)
# ======================================================================
def collect_trajectory(
    env: CodeReviewEnv,
    model,
    tokenizer,
    max_steps: int = 10,
    temperature: float = 0.0   # changed to greedy
) -> Trajectory:
    obs = env.reset()
    history_lines = []
    
    states = []
    actions = []
    rewards = []
    logprobs = []
    dones = []
    
    for step in range(max_steps):
        prompt = build_prompt(obs, history_lines)
        states.append(prompt)
        
        action_text, logprob = generate_action_with_logprob(
            prompt, model, tokenizer, temperature
        )
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
        if done:
            break
    
    return Trajectory(states, actions, rewards, logprobs, dones)

def collect_trajectories(
    env: CodeReviewEnv,
    model,
    tokenizer,
    n_trajectories: int,
    max_steps: int = 10,
    task_levels: Optional[List[str]] = None,
    task_weights: Optional[List[float]] = None,
) -> List[Trajectory]:
    # Link training to RedTeam's full bug distribution by sampling tasks
    # per trajectory instead of training only on env default ("easy").
    if task_levels is None:
        task_levels = list(BUG_DB.keys())
    if task_weights is not None and len(task_weights) != len(task_levels):
        raise ValueError("task_weights must match task_levels length")
    if task_weights is not None and sum(task_weights) <= 0:
        raise ValueError("task_weights must have a positive total")

    trajectories = []
    for i in range(n_trajectories):
        # Weighted sampling supports curriculum-style training schedules.
        sampled_task = random.choices(task_levels, weights=task_weights, k=1)[0]
        env.set_task(sampled_task)
        traj = collect_trajectory(env, model, tokenizer, max_steps)
        total_reward = sum(traj.rewards)
        print(f"Trajectory {i+1}/{n_trajectories}: "
              f"task={sampled_task}, steps={len(traj)}, reward={total_reward:.3f}")
        trajectories.append(traj)
    return trajectories

# ======================================================================
# 9. ADVANTAGE ESTIMATION (unchanged)
# ======================================================================
def compute_returns_and_advantages(
    rewards: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    standardize: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Computes discounted returns and normalised advantages (no critic).
    Advantages = returns - mean(returns)  (or zero baseline).
    """
    n = len(rewards)
    returns = [0.0] * n
    running_return = 0.0
    for t in reversed(range(n)):
        if dones[t]:
            running_return = 0.0
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    if standardize:
        advantages = np.array(returns) - np.mean(returns)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages / adv_std).tolist()
    else:
        advantages = returns.copy()
    
    return advantages, returns
# ======================================================================
# 10. COMPUTE NEW LOGPROBS (unchanged)
# ======================================================================
def compute_logprob(prompt: str, action: str, model, tokenizer) -> float:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = formatted + action
    inputs = tokenizer(full_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    action_ids = tokenizer.encode(action, add_special_tokens=False)
    prefix_ids = tokenizer.encode(formatted, add_special_tokens=False)
    action_start = len(prefix_ids)
    
    logprobs = []
    for idx, token_id in enumerate(action_ids):
        position = action_start + idx - 1
        if 0 <= position < logits.shape[1]:
            token_logits = logits[0, position]
            token_logprob = F.log_softmax(token_logits, dim=-1)[token_id].item()
            logprobs.append(token_logprob)
    return sum(logprobs) if logprobs else -100.0

# ======================================================================
# 11. PPO UPDATE (unchanged except uses compute_logprob correctly)
# ======================================================================
def ppo_update(
    trajectories: List[Trajectory],
    model,
    tokenizer,
    optimizer,
    n_epochs: int = 4,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
) -> Dict[str, float]:
    model.train()
    
    all_states = []
    all_actions = []
    all_old_logprobs = []
    all_advantages = []
    all_returns = []
    
    for traj in trajectories:
        advantages, returns = compute_returns_and_advantages(
            traj.rewards, traj.dones, gamma=gamma, standardize=True
        )
        all_states.extend(traj.states)
        all_actions.extend(traj.actions)
        all_old_logprobs.extend(traj.logprobs)
        all_advantages.extend(advantages)
        all_returns.extend(returns)
    
    n_samples = len(all_states)
    total_loss = 0.0
    total_policy_loss = 0.0
    total_entropy = 0.0
    n_updates = 0
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        for i in indices:
            state = all_states[i]
            action = all_actions[i]
            old_logprob = all_old_logprobs[i]
            advantage = all_advantages[i]
            
            # Use the same chat template for PPO update
            messages = [{"role": "user", "content": state}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_text = formatted + action
            inputs = tokenizer(full_text, return_tensors="pt", max_length=768, truncation=True).to("cuda")
            
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
            
            if not logprobs:
                continue
            
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
    
    return {
        "loss": total_loss / n_updates if n_updates > 0 else 0.0,
        "policy_loss": total_policy_loss / n_updates if n_updates > 0 else 0.0,
        "entropy": total_entropy / n_updates if n_updates > 0 else 0.0,
    }
# ======================================================================
# 12. EVALUATION (unchanged)
# ======================================================================
def evaluate_policy(
    env: CodeReviewEnv,
    model,
    tokenizer,
    n_episodes: int = 10,
    max_steps: int = 10
) -> Dict[str, float]:
    model.eval()
    total_rewards = []
    episode_lengths = []
    success_count = 0
    
    for _ in range(n_episodes):
        traj = collect_trajectory(env, model, tokenizer, max_steps, temperature=0.0)
        total_reward = sum(traj.rewards)
        total_rewards.append(total_reward)
        episode_lengths.append(len(traj))
        if total_reward > 0.5:
            success_count += 1
    
    return {
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_length": np.mean(episode_lengths),
        "success_rate": success_count / n_episodes,
    }

# ======================================================================
# 13. MAIN TRAINING LOOP (added sanity check and warm-up)
# ======================================================================
def train_ppo(
    n_iterations: int = 50,
    trajectories_per_iter: int = 10,
    n_epochs: int = 2,
    max_steps: int = 10,
    learning_rate: float = 3e-5,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
    eval_every: int = 5,
    task_levels: Optional[List[str]] = None,
    curriculum_weighted_sampling: bool = True,
    reward_profile: str = "full",
):
    print("Loading model...")
    model, tokenizer = load_model()
    
    # NEW: Sanity check before any training
    if not test_model_sanity(model, tokenizer):
        print("\n❌ Model sanity check failed – cannot proceed.")
        return
    
    # NEW: Supervised warm-up to teach JSON format (500 steps with epochs=8)
    supervised_warmup(model, tokenizer, n_examples=500, epochs=8)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    env = CodeReviewEnv()
    if task_levels is None:
        task_levels = list(BUG_DB.keys())
    
    print(f"\n{'='*60}")
    print(f"Starting PPO Training")
    print(f"Iterations: {n_iterations}")
    print(f"Trajectories per iteration: {trajectories_per_iter}")
    print(f"PPO epochs: {n_epochs}")
    print(f"Reward profile: {reward_profile}")
    print(f"{'='*60}\n")
    reward_history: List[float] = []
    loss_history: List[float] = []
    
    for iteration in range(n_iterations):
        print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
        # Optional weighted curriculum:
        # start with easier tasks and smoothly ramp difficulty over training.
        if curriculum_weighted_sampling:
            progress = (iteration + 1) / max(n_iterations, 1)
            easy_w = max(0.15, 0.55 - 0.40 * progress)
            medium_w = max(0.15, 0.25 - 0.10 * progress)
            hard_w = 0.10 + 0.05 * progress
            harder_w = 0.05 + 0.20 * progress
            hardest_w = 0.05 + 0.25 * progress
            task_weight_map = {
                "easy": easy_w,
                "medium": medium_w,
                "hard": hard_w,
                "harder": harder_w,
                "hardest": hardest_w,
            }
            task_weights = [task_weight_map.get(level, 1.0) for level in task_levels]
        else:
            task_weights = None
        
        print("Collecting trajectories...")
        trajectories = collect_trajectories(
            env,
            model,
            tokenizer,
            trajectories_per_iter,
            max_steps,
            task_levels=task_levels,
            task_weights=task_weights,
        )
        
        avg_reward = np.mean([sum(t.rewards) for t in trajectories])
        avg_length = np.mean([len(t) for t in trajectories])
        reward_history.append(float(avg_reward))
        
        print(f"Avg reward: {avg_reward:.3f}")
        print(f"Avg length: {avg_length:.1f}")
        
        print("Updating policy...")
        metrics = ppo_update(
            trajectories,
            model,
            tokenizer,
            optimizer,
            n_epochs=n_epochs,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            gamma=gamma,
        )
        
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Policy loss: {metrics['policy_loss']:.4f}")
        print(f"Entropy: {metrics['entropy']:.4f}")
        loss_history.append(float(metrics["loss"]))
        
        if (iteration + 1) % eval_every == 0:
            print("\nEvaluating policy...")
            eval_metrics = evaluate_policy(env, model, tokenizer, n_episodes=10)
            print(f"Eval avg reward: {eval_metrics['avg_reward']:.3f} ± {eval_metrics['std_reward']:.3f}")
            print(f"Eval success rate: {eval_metrics['success_rate']:.2%}")
            print(f"Eval avg length: {eval_metrics['avg_length']:.1f}")
    
    print("\n" + "="*60)
    print("Training complete. Saving model...")
    model.save_pretrained("ppo_final_model")
    tokenizer.save_pretrained("ppo_final_model")
    print("Model saved to ppo_final_model/")

    # Save training curves for quick before/after comparisons.
    if reward_history:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(reward_history) + 1), reward_history, marker="o")
        plt.title("Average Reward per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("reward_curve.png", dpi=150)
        plt.close()

    if loss_history:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", color="tab:red")
        plt.title("Training Loss per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("loss_curve.png", dpi=150)
        plt.close()

    if os.path.exists("reward_curve.png") and os.path.exists("loss_curve.png"):
        print("Saved reward_curve.png and loss_curve.png")
    print("="*60)

# ======================================================================
# 14. ENTRY POINT (unchanged)
# ======================================================================
if __name__ == "__main__":
    train_ppo(
        n_iterations=30,
        trajectories_per_iter=10,
        n_epochs=4,
        max_steps=10,
        learning_rate=3e-5,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        gamma=0.99,
        eval_every=5,
    )
