# environment.py – FULLY CORRECTED RL Environment (TRUE Markov + Fixed Bugs)

import sys
import subprocess
import tempfile
import os
import re
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List
from collections import Counter

from models import (
    AnyAction, WriteComment, ProposeFix, Execute, Inspect,
    RunLinter, RunTests, QueryDocs, Skip, Done, AskQuestion,
    Observation, Reward, State
)
from grader import RigorousGrader
from redteam import RedTeam
from test_runner import TestRunner
from author import PersonaAuthor
from rltool import ToolBox

# ======================================================================
# FULLY MARKOV OBSERVATION (NOTHING HIDDEN)
# ======================================================================
@dataclass
class EnhancedObservation:
    """
    Complete Markov state - agent has ALL information needed for optimal decisions.
    Reward function depends ONLY on (state, action), not hidden variables.
    """
    # Code state
    code_snippet: str
    last_tool_output: str
    
    # Current metrics
    current_test_score: float
    current_lint_score: float
    negotiation_score: float
    
    # CRITICAL: Previous metrics (for understanding deltas)
    previous_test_score: float
    previous_lint_score: float
    
    # CRITICAL: Author internal state (affects reward gating)
    author_confidence: float
    author_threshold: float  # When author accepts
    
    # Progress tracking
    step: int
    max_steps: int
    progress_ratio: float
    
    # Tool usage flags
    tests_run: bool
    linter_run: bool
    docs_queried: bool
    
    # Action history (with outcomes)
    last_action_type: str
    action_history: List[str]  # Last 5 actions
    
    # Terminal flag
    done: bool
    
    # Additional context
    bug_description: str
    comments_count: int


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================
def execute_code(code: str, timeout_sec: int = 5) -> Tuple[bool, str, str]:
    if not code.strip():
        return False, "", "Error: Empty code"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )
        success = (result.returncode == 0)
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Timeout after {timeout_sec}s"
    except Exception as e:
        return False, "", f"Execution error: {str(e)}"
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


# ======================================================================
# ENHANCED CODE REVIEW ENVIRONMENT
# ======================================================================
@dataclass
class CodeReviewEnv:
    task: str = "easy"
    max_steps: int = 10
    step_penalty: float = 0.01
    
    # Curriculum learning
    auto_difficulty: bool = False
    success_threshold: float = 0.7
    
    # Reward shaping parameters
    delta_weight: float = 0.3
    tool_usage_bonus: float = 0.05
    diversity_bonus: float = 0.03
    
    _red_team: Optional[RedTeam] = field(init=False, default=None)
    _author: Optional[PersonaAuthor] = field(init=False, default=None)

    _current_code: str = field(init=False, default="")
    _current_bug_id: str = field(init=False, default="")
    _bug_description: str = field(init=False, default="")
    _oracle_fix: str = field(init=False, default="")

    _comments: list = field(init=False, default_factory=list)
    _test_results: Optional[str] = field(init=False, default=None)
    _lint_results: Optional[str] = field(init=False, default=None)
    _doc_results: Optional[str] = field(init=False, default=None)

    _step_count: int = field(init=False, default=0)
    _done: bool = field(init=False, default=False)
    
    # State tracking for dense rewards
    _previous_test_score: float = field(init=False, default=0.0)
    _previous_lint_score: float = field(init=False, default=0.0)
    _current_test_score: float = field(init=False, default=0.0)
    _current_lint_score: float = field(init=False, default=0.0)
    
    # Tool usage tracking
    _tests_run: bool = field(init=False, default=False)
    _linter_run: bool = field(init=False, default=False)
    _docs_queried: bool = field(init=False, default=False)
    
    # Action history
    _action_history: List[str] = field(init=False, default_factory=list)
    _last_action_type: str = field(init=False, default="none")
    
    # FIXED: Track CUMULATIVE episode reward
    _episode_total_reward: float = field(init=False, default=0.0)
    _episode_rewards: List[float] = field(init=False, default_factory=list)
    _difficulty_level: int = field(init=False, default=0)

    # ===================================================================
    def __post_init__(self):
        self.set_task(self.task)

    # ===================================================================
    def set_task(self, task: str):
        if task not in ["easy", "medium", "hard", "harder", "hardest"]:
            raise ValueError(f"Unknown task: {task}")

        self.task = task
        self._red_team = RedTeam(task)
        self._author = PersonaAuthor()

        task_to_level = {
            "easy": 0, "medium": 1, "hard": 2, 
            "harder": 3, "hardest": 4
        }
        self._difficulty_level = task_to_level[task]
        
        self._reset_internal()

    # ===================================================================
    def _reset_internal(self):
        self._step_count = 0
        self._comments = []
        self._test_results = None
        self._lint_results = None
        self._doc_results = None
        self._done = False
        
        # Reset state tracking
        self._previous_test_score = 0.0
        self._previous_lint_score = 0.0
        self._current_test_score = 0.0
        self._current_lint_score = 0.0
        
        self._tests_run = False
        self._linter_run = False
        self._docs_queried = False
        
        self._action_history = []
        self._last_action_type = "none"
        
        # FIXED: Reset episode cumulative reward
        self._episode_total_reward = 0.0

        self._author.reset()

        # Base tasks
        if self.task == "easy":
            original = "def get_user(id):\n    if id in users:\n        return users[id]"
        elif self.task == "medium":
            original = "def process_items(items):\n    for item in items:\n        print(item)"
        elif self.task == "hard":
            original = "def average(data):\n    if not data:\n        return 0\n    return sum(data) / len(data)"
        elif self.task == "harder":
            original = "counter = 0\ndef increment():\n    global counter\n    with lock:\n        counter += 1"
        else:
            original = "def safe_work():\n    with lock1:\n        with lock2:\n            do_work()"

        buggy_code, bug_id, desc, oracle = self._red_team.inject_bug(original)
        self._current_code = buggy_code
        self._current_bug_id = bug_id
        self._bug_description = desc
        self._oracle_fix = oracle
        self._comments.append(f"[RedTeam] {desc}")

    # ===================================================================
    def reset(self) -> EnhancedObservation:
        """Reset with optional curriculum adjustment."""
        if self.auto_difficulty and len(self._episode_rewards) > 0:
            recent_performance = sum(self._episode_rewards[-5:]) / min(5, len(self._episode_rewards))
            
            if recent_performance > self.success_threshold and self._difficulty_level < 4:
                self._difficulty_level += 1
                print(f"[Curriculum] Increasing difficulty to level {self._difficulty_level}")
            elif recent_performance < 0.3 and self._difficulty_level > 0:
                self._difficulty_level -= 1
                print(f"[Curriculum] Decreasing difficulty to level {self._difficulty_level}")
            
            level_to_task = {0: "easy", 1: "medium", 2: "hard", 3: "harder", 4: "hardest"}
            self.task = level_to_task[self._difficulty_level]
            self._red_team = RedTeam(self.task)
        
        self._reset_internal()
        return self._get_observation()

    # ===================================================================
    def _get_observation(self) -> EnhancedObservation:
        """
        Return COMPLETE Markov state.
        NOTHING is hidden - reward depends ONLY on (state, action).
        """
        return EnhancedObservation(
            code_snippet=self._current_code,
            last_tool_output=self._test_results or "",
            
            # Current metrics
            current_test_score=self._current_test_score,
            current_lint_score=self._current_lint_score,
            negotiation_score=self._author.get_negotiation_score(),
            
            # EXPOSED: Previous metrics (for delta understanding)
            previous_test_score=self._previous_test_score,
            previous_lint_score=self._previous_lint_score,
            
            # EXPOSED: Author internal state (affects gating)
            author_confidence=self._author._confidence,
            author_threshold=self._author.thresholds.get(self._author.personality, 0.5),
            
            # Progress
            step=self._step_count,
            max_steps=self.max_steps,
            progress_ratio=self._step_count / self.max_steps,
            
            # Tool usage
            tests_run=self._tests_run,
            linter_run=self._linter_run,
            docs_queried=self._docs_queried,
            
            # Action history
            last_action_type=self._last_action_type,
            action_history=self._action_history[-5:],
            
            # Terminal
            done=self._done,
            
            # Context
            bug_description=self._bug_description,
            comments_count=len(self._comments),
        )

    # ===================================================================
    def _compute_dense_reward(
        self, 
        action: AnyAction, 
        base_reward: float,
        action_type: str
    ) -> float:
        """
        Stabilized dense reward:
        - Decoupled terminal bonus
        - Controlled base scaling
        - Symmetric delta handling
        - Reduced reward hacking surface
        """

    # ============================================================
    # 0. BASE REWARD (controlled contribution)
    # ============================================================
        reward = 0.4 * base_reward   # ↓ reduce dominance

    # ============================================================
    # 1. DELTA REWARDS (primary learning signal)
    # ============================================================
        effective_delta_weight = self.delta_weight
        if action_type == "propose_fix":
            effective_delta_weight *= 0.4  # stronger cut to avoid overlap

        test_delta = self._current_test_score - self._previous_test_score
        lint_delta = self._current_lint_score - self._previous_lint_score

    # symmetric (no artificial dampening for negatives)
        reward += effective_delta_weight * test_delta
        reward += 0.5 * effective_delta_weight * lint_delta

    # ============================================================
    # 2. TERMINAL SUCCESS BONUS (clean & isolated)
    # ============================================================
        if action_type == "propose_fix":
            if self._current_test_score > 0.95:
                reward += 0.4   # slightly reduced to prevent saturation
            elif self._current_test_score > 0.85:
                reward += 0.2   # smoother gradient instead of jump

    # ============================================================
    # 3. TOOL USAGE (early guidance only)
    # ============================================================
        if action_type == "run_tests":
            if not self._tests_run:
                reward += self.tool_usage_bonus
            reward += 0.015

        elif action_type == "run_linter":
            if not self._linter_run:
                reward += self.tool_usage_bonus
            reward += 0.015

        elif action_type == "query_docs":
            if not self._docs_queried:
                reward += self.tool_usage_bonus * 0.5
    
        elif action_type == "ask_question":
            if self._step_count <= 3:
                reward += 0.02   # tighter window

    # ============================================================
    # 4. EXPLORATION (less noisy)
    # ============================================================
        if len(self._action_history) >= 3:
            recent = self._action_history[-3:]
            unique = len(set(recent))

            if unique == 1:
                reward -= 0.05
            elif unique == 3:
                reward += self.diversity_bonus * 0.7  # reduce randomness bias

    # ============================================================
    # 5. ANTI-HACKING
    # ============================================================
        if action_type == "propose_fix":
            if not self._tests_run:
                reward -= 0.25   # stronger enforcement
            if self._step_count < 2:
                reward -= 0.1
            if self._tests_run and self._linter_run:
                reward += 0.02

    # ============================================================
    # 6. STEP PENALTY (progress pressure)
    # ============================================================
        reward -= self.step_penalty

    # ============================================================
    # 7. CLIP (final safety)
    # ============================================================
        return max(-1.0, min(1.0, reward))
        

    # ===================================================================
    def _get_action_type(self, action: AnyAction) -> str:
        """Extract action type as string."""
        if isinstance(action, RunTests):
            return "run_tests"
        elif isinstance(action, RunLinter):
            return "run_linter"
        elif isinstance(action, QueryDocs):
            return "query_docs"
        elif isinstance(action, Execute):
            return "execute"
        elif isinstance(action, Inspect):
            return "inspect"
        elif isinstance(action, WriteComment):
            return "write_comment"
        elif isinstance(action, AskQuestion):
            return "ask_question"
        elif isinstance(action, ProposeFix):
            return "propose_fix"
        elif isinstance(action, Done):
            return "done"
        elif isinstance(action, Skip):
            return "skip"
        else:
            return "unknown"

    # ===================================================================
    def step(self, action: AnyAction) -> Tuple[EnhancedObservation, Reward, bool, Dict[str, Any]]:
        """
        TRUE RL STEP with:
        - Complete Markov observations (no hidden state)
        - Dense intermediate rewards
        - Delta-based credit assignment (no double-counting)
        - Proper episode reward tracking
        """
        if self._done:
            raise RuntimeError("Episode already finished")

        # Store previous metrics for delta computation
        self._previous_test_score = self._current_test_score
        self._previous_lint_score = self._current_lint_score
        
        base_reward = 0.0
        info = {}
        action_type = self._get_action_type(action)
        
        # Update action history
        self._action_history.append(action_type)
        self._last_action_type = action_type

        # ==============================================================
        # TOOL ACTIONS
        # ==============================================================
        if isinstance(action, Execute):
            success, stdout, stderr = execute_code(self._current_code)
            output = (stdout + stderr).strip() or "No output"
            self._test_results = f"[Execute] {'Success' if success else 'Failed'}\n{output[:300]}"
            base_reward = 0.001 if success else -0.05

        elif isinstance(action, Inspect):
            self._test_results = f"[Inspect]\n{self._current_code[:500]}"
            base_reward = 0.001

        elif isinstance(action, RunLinter):
            lint_output = ToolBox.run_linter(self._current_code)
            self._lint_results = lint_output[:500]
            self._test_results = f"[Linter]\n{self._lint_results}"
            
            self._current_lint_score = self._run_linter_score(self._current_code)
            self._linter_run = True
            base_reward = 0.002

        elif isinstance(action, RunTests):
            runner = TestRunner(self._current_bug_id)
            score, output = runner.run_tests(self._current_code)
            
            self._current_test_score = score
            self._tests_run = True
            
            self._test_results = f"[Tests] Score: {score:.2f}\n{output[:300]}"
            base_reward = 0.002
            
            if score > 0.8:
                base_reward += 0.005

        elif isinstance(action, QueryDocs):
            doc = ToolBox.query_docs(action.query_topic)
            self._doc_results = doc
            self._test_results = f"[Docs]\n{doc[:400]}"
            self._docs_queried = True
            base_reward = 0.001

        # ==============================================================
        # COMMUNICATION ACTIONS
        # ==============================================================
        elif isinstance(action, WriteComment):
            self._comments.append(f"Agent: {action.comment_text}")
            
            response = self._author.respond(
                agent_comment=action.comment_text,
                test_results=self._test_results,
                lint_results=self._lint_results,
                doc_results=self._doc_results,
                proposed_fix=None,
                original_code=self._current_code
            )
            
            self._comments.append(f"Author: {response}")
            self._test_results = f"[Comment] Author: {response[:200]}"
            base_reward = 0.001

        elif isinstance(action, AskQuestion):
            self._comments.append(f"Agent: {action.question}")
            
            response = self._author.respond(
                agent_question=action.question,
                test_results=self._test_results,
                lint_results=self._lint_results,
                doc_results=self._doc_results,
                proposed_fix=None,
                original_code=self._current_code
            )
            
            self._comments.append(f"Author: {response}")
            self._test_results = f"[Question] Author: {response[:200]}"
            base_reward = 0.002

        # ==============================================================
        # FINAL FIX ACTION
        # ==============================================================
        elif isinstance(action, ProposeFix):
            if not action.fix_code:
                base_reward = -0.05
                self._done = True
            else:
                self._current_code = action.fix_code
                
                runner = TestRunner(self._current_bug_id)
                test_score, test_output = runner.run_tests(self._current_code)
                lint_score = self._run_linter_score(self._current_code)
                negotiation_score = self._author.get_negotiation_score()
                
                # Update current scores
                self._current_test_score = test_score
                self._current_lint_score = lint_score
                
                # Component reward (scaled down to allow delta distribution)
                component_reward = (
                    0.4 * test_score +
                    0.15 * lint_score +
                    0.15 * negotiation_score
                )
                
                efficiency = 1.0 - (self._step_count / self.max_steps)
                component_reward += 0.1 * efficiency
                
                # Cross-signal consistency
                if test_score > 0.8 and lint_score < 0.3:
                    component_reward *= 0.85
                if test_score < 0.3 and lint_score > 0.8:
                    component_reward *= 0.75
                if test_score > 0.8 and negotiation_score < 0.3:
                    component_reward *= 0.8
                
                # Author gating
                threshold = self._author.thresholds.get(self._author.personality, 0.5)
                if self._author._confidence < threshold:
                    component_reward = max(0.0, component_reward - 0.2)
                    if self._step_count < self.max_steps:
                        self._done = False
                    else:
                        self._done = True
                else:
                    self._done = True
                
                base_reward = component_reward
                self._test_results = f"[Fix] Test: {test_score:.2f}, Lint: {lint_score:.2f}\n{test_output[:200]}"

        # ==============================================================
        # TERMINATION ACTIONS
        # ==============================================================
        elif isinstance(action, Skip):
            base_reward = -0.03
            self._done = True

        elif isinstance(action, Done):
            if self._tests_run:
                base_reward = self._current_test_score * 0.5 - 0.2
            else:
                base_reward = -0.04
            self._done = True

        else:
            base_reward = -0.02
            self._done = True

        # ==============================================================
        # COMPUTE FINAL DENSE REWARD (with action_type for fix detection)
        # ==============================================================
        final_reward = self._compute_dense_reward(action, base_reward, action_type)
        
        # FIXED: Track CUMULATIVE episode reward
        self._episode_total_reward += final_reward
        
        # ==============================================================
        # STEP UPDATE
        # ==============================================================
        self._step_count += 1
        
        if self._step_count >= self.max_steps:
            self._done = True
        
        # FIXED: Store TOTAL episode reward, not just last step
        if self._done:
            self._episode_rewards.append(self._episode_total_reward)
        
        obs = self._get_observation()
        
        info = {
            "test_score": self._current_test_score,
            "lint_score": self._current_lint_score,
            "test_delta": self._current_test_score - self._previous_test_score,
            "lint_delta": self._current_lint_score - self._previous_lint_score,
            "base_reward": base_reward,
            "final_reward": final_reward,
            "episode_total": self._episode_total_reward,
        }
        
        return obs, Reward(value=final_reward), self._done, info

    # ===================================================================
    def _run_linter_score(self, code: str) -> float:
        """Run pylint and return normalized score [0, 1]."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                ['pylint', tmp_path, '--score=y', '--exit-zero'],
                capture_output=True,
                text=True,
                timeout=5
            )

            match = re.search(r"rated at (\d+\.\d+)/10", result.stdout)
            if match:
                return float(match.group(1)) / 10.0
            return 0.0
        except:
            return 0.0
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    # ===================================================================
    def state(self) -> State:
        """Legacy compatibility."""
        return State(
            pr_title="Code Review",
            pr_description=self._bug_description,
            code_snippet=self._current_code,
            comments=self._comments.copy(),
            test_results=self._test_results,
            step=self._step_count,
            done=self._done
        )