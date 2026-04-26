# environment.py – FULLY CORRECTED RL Environment (TRUE Markov + Fixed Bugs)

import sys
import subprocess
import tempfile
import os
import re
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List

from models import (
    AnyAction, WriteComment, ProposeFix, Execute, Inspect,
    RunLinter, RunTests, QueryDocs, Skip, Done, AskQuestion,
    Observation, Reward, State
)
from redteam import RedTeam
from test_runner import TestRunner
from author import PersonaAuthor
from rltool import ToolBox
from rubrics import (
    ToolUsageRubric,
    TestDeltaRubric,
    LintDeltaRubric,
    TerminalSuccessRubric,
    ExplorationRubric,
    AntiHackingRubric,
    StepPenaltyRubric,
)

# ======================================================================
# FULLY MARKOV OBSERVATION (NOTHING HIDDEN)
# ======================================================================
@dataclass
class EnhancedObservation:
    code_snippet: str
    last_tool_output: str

    current_test_score: float
    current_lint_score: float
    negotiation_score: float

    previous_test_score: float
    previous_lint_score: float

    author_confidence: float
    author_threshold: float

    step: int
    max_steps: int
    progress_ratio: float

    tests_run: bool
    linter_run: bool
    docs_queried: bool

    last_action_type: str
    action_history: List[str]

    done: bool

    bug_description: str
    comments_count: int

    # default fields must be at the very end
    author_response: str = ""

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
    reward_profile: str = "full"  # "full" or "core"

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
    _last_author_response: str = field(init=False, default="")

    # FIXED: Track CUMULATIVE episode reward
    _episode_total_reward: float = field(init=False, default=0.0)
    _episode_rewards: List[float] = field(init=False, default_factory=list)
    _difficulty_level: int = field(init=False, default=0)

    # Bug-id bridge:
    # RedTeam has fine-grained IDs, while TestRunner currently expects a
    # smaller canonical set. Keep this mapping here so both modules can evolve
    # independently without breaking evaluation.
    _BUG_ID_CANONICAL_MAP = {
        # Easy-family
        "simple_typo": "null_check",
        "default_value": "null_check",
        "empty_return": "null_check",
        "string_index": "off_by_one",

        # Medium-family
        "loop_skip": "off_by_one",
        "sign_error": "wrong_operator",
        "swap_args": "wrong_operator",
        "uninitialised_var": "null_check",

        # Hard-family
        "division_by_zero_empty": "division_by_zero",
        "division_by_zero_zero": "division_by_zero",
        "float_precision": "division_by_zero",
        "abs_usage": "division_by_zero",
        "round_error": "division_by_zero",
    }

    # ===================================================================
    def __post_init__(self):
        self.set_task(self.task)

    # ===================================================================
    def _build_rubrics(self):
        """
        Build rubric stack from a named reward profile.
        - full: richer shaping for exploration/tool-use behavior
        - core: minimal stable signal for quick ablations/baselines
        """
        core_rubrics = [
            TestDeltaRubric(weight=self.delta_weight),
            LintDeltaRubric(weight=self.delta_weight),
            TerminalSuccessRubric(),
            StepPenaltyRubric(penalty=self.step_penalty),
        ]
        if self.reward_profile == "core":
            return core_rubrics
        if self.reward_profile == "full":
            return [
                *core_rubrics[:-1],  # step penalty appended at end for consistent ordering
                ToolUsageRubric(bonus=self.tool_usage_bonus),
                ExplorationRubric(penalty=-0.05, bonus=self.diversity_bonus * 0.7),
                AntiHackingRubric(),
                core_rubrics[-1],
            ]
        raise ValueError(f"Unknown reward_profile: {self.reward_profile}")

    # ===================================================================
    def set_task(self, task: str):
        if task not in ["easy", "medium", "hard", "harder", "hardest"]:
            raise ValueError(f"Unknown task: {task}")

        self.task = task
        # Use stochastic bug sampling across episodes; fixed seed here would
        # repeatedly select the same bug and weaken training diversity.
        self._red_team = RedTeam(task, seed=None)
        self._author = PersonaAuthor()
        self.rubrics = self._build_rubrics()

        task_to_level = {
            "easy": 0, "medium": 1, "hard": 2,
            "harder": 3, "hardest": 4
        }
        self._difficulty_level = task_to_level[task]

        self._reset_internal()

    # ===================================================================
    def _reset_internal(self):
        self._step_count = 0                         # ← FIXED
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
        self._last_author_response = ""

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
            # Keep curriculum stochastic for better coverage within each level.
            self._red_team = RedTeam(self.task, seed=None)

        self._reset_internal()
        return self._get_observation()

    # ===================================================================
    def _get_observation(self) -> EnhancedObservation:
        """Return COMPLETE Markov state."""
        # Keep the author's message separate from tool output.
        # Using `_test_results` here can leak unrelated outputs (tests/linter/docs)
        # and gives the policy a noisy signal for dialogue actions.
        if self._last_action_type in ("comment", "question", "fix"):
            author_response = self._last_author_response
        else:
            author_response = ""

        return EnhancedObservation(
            code_snippet=self._current_code,
            last_tool_output=self._test_results or "",
            author_response=author_response,          # ← now field exists

            current_test_score=self._current_test_score,
            current_lint_score=self._current_lint_score,
            negotiation_score=self._author.get_negotiation_score(),

            previous_test_score=self._previous_test_score,
            previous_lint_score=self._previous_lint_score,

            author_confidence=self._author._confidence,
            author_threshold=self._author.thresholds.get(self._author.personality, 0.5),

            step=self._step_count,
            max_steps=self.max_steps,
            # Guard against accidental `max_steps=0` configs.
            progress_ratio=(self._step_count / self.max_steps) if self.max_steps > 0 else 1.0,

            tests_run=self._tests_run,
            linter_run=self._linter_run,
            docs_queried=self._docs_queried,

            last_action_type=self._last_action_type,
            action_history=self._action_history[-5:],

            done=self._done,

            bug_description=self._bug_description,
            comments_count=len(self._comments),
        )

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
            return "comment"
        elif isinstance(action, AskQuestion):
            return "question"
        elif isinstance(action, ProposeFix):
            return "fix"
        elif isinstance(action, Done):
            return "done"
        elif isinstance(action, Skip):
            return "skip"
        else:
            return "unknown"

    # ===================================================================
    def _get_test_runner_bug_id(self) -> str:
        """
        Normalize RedTeam bug ids to the canonical ids understood by TestRunner.
        Falls back to the original id for known direct matches.
        """
        return self._BUG_ID_CANONICAL_MAP.get(self._current_bug_id, self._current_bug_id)

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
        # Snapshot tool-usage flags BEFORE action mutates them.
        # Rubrics use these to detect true "first-use" behavior.
        prev_tests_run = self._tests_run
        prev_linter_run = self._linter_run
        prev_docs_queried = self._docs_queried

        base_reward = 0.0
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
            runner = TestRunner(self._get_test_runner_bug_id())
            score, output = runner.run_tests(self._current_code)

            self._current_test_score = score
            self._tests_run = True

            self._test_results = f"[Tests] Score: {score:.2f}\n{output[:300]}"
            base_reward = 0.002

            if score > 0.8:
                base_reward += 0.005

        elif isinstance(action, QueryDocs):
            # Normalize query to avoid rewarding empty/noisy requests.
            query_topic = (action.query_topic or "").strip()
            doc = ToolBox.query_docs(query_topic if query_topic else "general bug fixing")
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
            self._last_author_response = response
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
                original_code=self._current_code                  # ← FIXED
            )

            self._comments.append(f"Author: {response}")
            self._last_author_response = response
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
                # Save original code BEFORE overwriting (for author.respond)
                original_buggy = self._current_code
                self._current_code = action.fix_code

                runner = TestRunner(self._get_test_runner_bug_id())
                test_score, test_output = runner.run_tests(self._current_code)
                lint_score = self._run_linter_score(self._current_code)
                negotiation_score = self._author.get_negotiation_score()

                self._current_test_score = test_score
                self._current_lint_score = lint_score

                # Author gating – determines if the episode ends, reward is separate
                threshold = self._author.thresholds.get(self._author.personality, 0.5)
                if self._author._confidence < threshold:
                    if self._step_count < self.max_steps:
                        self._done = False
                    else:
                        self._done = True
                else:
                    self._done = True

                # Get author's verbal feedback (pushback/acceptance)
                author_feedback = self._author.respond(
                    agent_comment=f"Proposed fix:\n{action.fix_code}",
                    test_results=f"Score: {test_score:.2f}",
                    lint_results=f"Score: {lint_score:.2f}",
                    doc_results=self._doc_results,
                    proposed_fix=action.fix_code,
                    original_code=original_buggy   # now correctly the buggy code, not the fix
                )
                self._test_results = f"[Fix] Author: {author_feedback[:200]}"
                self._comments.append(f"Author: {author_feedback}")
                self._last_author_response = author_feedback

                base_reward = 0.001   # rubrics provide the real signal

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
        # STEP UPDATE (before rubric computation so info contains final step)
        # ==============================================================
        self._step_count += 1
        if self._step_count >= self.max_steps:
            self._done = True

        # Get fresh observation (needed for rubrics that may read obs)
        obs = self._get_observation()

        # Prepare info dict (rubrics may need action_type and deltas)
        info = {
            "action_type": action_type,
            "test_score": self._current_test_score,
            "lint_score": self._current_lint_score,
            "test_delta": self._current_test_score - self._previous_test_score,
            "lint_delta": self._current_lint_score - self._previous_lint_score,
            "prev_tests_run": prev_tests_run,
            "prev_linter_run": prev_linter_run,
            "prev_docs_queried": prev_docs_queried,
            "docs_query_len": len((action.query_topic or "").strip()) if isinstance(action, QueryDocs) else 0,
            "base_reward": base_reward,
        }

        # ==============================================================
        # COMPUTE FINAL REWARD USING RUBRICS
        # ==============================================================
        rubric_score = sum(r(self, action, obs, None, self._done, info) for r in self.rubrics)
        final_reward = 0.4 * base_reward + rubric_score
        final_reward = max(-1.0, min(1.0, final_reward))   # safety clip

        # Track cumulative episode reward
        self._episode_total_reward += final_reward

        # Store episode total if done
        if self._done:
            self._episode_rewards.append(self._episode_total_reward)

        # Complete info
        info["final_reward"] = final_reward
        info["episode_total"] = self._episode_total_reward

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
