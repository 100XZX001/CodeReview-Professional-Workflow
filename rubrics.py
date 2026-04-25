# rubrics.py – OpenEnv Rubrics for Code Review Environment
from openenv.core import Rubric

# --------------------------------------------------------------------------------
# 1. TOOL‑USAGE BONUS (encourages first‑time use of diagnostic tools)
# --------------------------------------------------------------------------------
class ToolUsageRubric(Rubric):
    """
    Small fixed reward the first time each of the major diagnostic tools is used.
    Also gives a tiny reward for every invocation to prevent the agent from ignoring them.
    """
    def __init__(self, bonus: float = 0.05):
        self.bonus = bonus

    def __call__(self, env, action, obs, reward, done, info):
        score = 0.0
        action_type = info.get("action_type", "")

        if action_type == "run_tests":
            if not env._tests_run:
                score += self.bonus
            score += 0.015
        elif action_type == "run_linter":
            if not env._linter_run:
                score += self.bonus
            score += 0.015
        elif action_type == "query_docs":
            if not env._docs_queried:
                score += self.bonus * 0.5
        elif action_type == "ask_question" and env._step_count <= 3:
            score += 0.02
        return score


# --------------------------------------------------------------------------------
# 2. DELTA‑BASED REWARDS (primary learning signal)
# --------------------------------------------------------------------------------
class TestDeltaRubric(Rubric):
    """
    Rewards improvement in the pass ratio of the test suite.
    """
    def __init__(self, weight: float = 0.3):
        self.weight = weight

    def __call__(self, env, action, obs, reward, done, info):
        delta = env._current_test_score - env._previous_test_score
        effective = self.weight
        if info.get("action_type") == "propose_fix":
            effective *= 0.4
        return effective * delta


class LintDeltaRubric(Rubric):
    """
    Rewards improvement in lint score (normalised 0‑1).
    """
    def __init__(self, weight: float = 0.3):
        self.weight = weight

    def __call__(self, env, action, obs, reward, done, info):
        delta = env._current_lint_score - env._previous_lint_score
        effective = self.weight * 0.5
        if info.get("action_type") == "propose_fix":
            effective *= 0.4
        return effective * delta


# --------------------------------------------------------------------------------
# 3. TERMINAL SUCCESS BONUS (propose_fix only)
# --------------------------------------------------------------------------------
class TerminalSuccessRubric(Rubric):
    """
    Bonus awarded when a proposed fix achieves high test and lint scores.
    Graded: >0.85 → 0.2, >0.95 → 0.4.
    """
    def __call__(self, env, action, obs, reward, done, info):
        if info.get("action_type") != "propose_fix":
            return 0.0
        score = 0.0
        if env._current_test_score > 0.95:
            score += 0.4
        elif env._current_test_score > 0.85:
            score += 0.2
        return score


# --------------------------------------------------------------------------------
# 4. EXPLORATION & DIVERSITY (discourages repetition, encourages varied actions)
# --------------------------------------------------------------------------------
class ExplorationRubric(Rubric):
    """
    Encourages diverse action sequences.
    - Penalty if last 3 actions are all the same.
    - Bonus if they are all different.
    """
    def __init__(self, penalty: float = -0.05, bonus: float = 0.021):
        self.penalty = penalty
        self.bonus = bonus

    def __call__(self, env, action, obs, reward, done, info):
        if len(env._action_history) < 3:
            return 0.0
        recent = env._action_history[-3:]
        unique = len(set(recent))
        if unique == 1:
            return self.penalty
        elif unique == 3:
            return self.bonus
        return 0.0


# --------------------------------------------------------------------------------
# 5. ANTI‑HACKING & CONSISTENCY (prevents reward without real work)
# --------------------------------------------------------------------------------
class AntiHackingRubric(Rubric):
    """
    Penalises suspicious behaviour:
    - proposing a fix without ever running tests.
    - proposing a fix too early (step < 2).
    Additional cross‑signal penalties are applied in the environment (not as a rubric)
    because they require modifying the base reward, not adding to it.
    """
    def __call__(self, env, action, obs, reward, done, info):
        if info.get("action_type") != "propose_fix":
            return 0.0
        score = 0.0
        if not env._tests_run:
            score -= 0.25
        if env._step_count < 2:
            score -= 0.1
        # tiny boost if the agent did the “right” preparation
        if env._tests_run and env._linter_run:
            score += 0.02
        return score


# --------------------------------------------------------------------------------
# 6. STEP PENALTY (time pressure)
# --------------------------------------------------------------------------------
class StepPenaltyRubric(Rubric):
    """
    Simple per‑step penalty to encourage efficient resolution.
    """
    def __init__(self, penalty: float = -0.01):
        self.penalty = penalty

    def __call__(self, env, action, obs, reward, done, info):
        return self.penalty
