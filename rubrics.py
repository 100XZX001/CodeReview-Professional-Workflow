# rubrics.py – Self-contained Rubrics (no external OpenEnv dependency)

class Rubric:
    """Minimal Rubric base – compatible with OpenEnv but self‑contained."""
    def __call__(self, env, action, obs, reward, done, info):
        return 0.0


# --------------------------------------------------------------------------------
# 1. TOOL‑USAGE BONUS
# --------------------------------------------------------------------------------
class ToolUsageRubric(Rubric):
    def __init__(self, bonus: float = 0.05):
        self.bonus = bonus

    def __call__(self, env, action, obs, reward, done, info):
        score = 0.0
        action_type = info.get("action_type", "")
        # Use pre-action flags from `info` so first-use bonuses are
        # computed correctly even though env flags are mutated in-step.
        prev_tests_run = info.get("prev_tests_run", env._tests_run)
        prev_linter_run = info.get("prev_linter_run", env._linter_run)
        prev_docs_queried = info.get("prev_docs_queried", env._docs_queried)

        if action_type == "run_tests":
            if not prev_tests_run:
                score += self.bonus
            score += 0.015
        elif action_type == "run_linter":
            if not prev_linter_run:
                score += self.bonus
            score += 0.015
        elif action_type == "query_docs":
            if not prev_docs_queried:
                score += self.bonus * 0.5
            # Encourage docs usage when it is likely useful:
            # - early exploration phase
            # - non-trivial query text
            if env._step_count <= 4 and info.get("docs_query_len", 0) >= 8:
                score += 0.01
            # Discourage repeated docs calls after the first-use signal.
            if prev_docs_queried:
                score -= 0.01
        elif action_type == "question" and env._step_count <= 3:
            score += 0.02
        return score


# --------------------------------------------------------------------------------
# 2. DELTA‑BASED REWARDS
# --------------------------------------------------------------------------------
class TestDeltaRubric(Rubric):
    def __init__(self, weight: float = 0.3):
        self.weight = weight

    def __call__(self, env, action, obs, reward, done, info):
        delta = env._current_test_score - env._previous_test_score
        effective = self.weight
        if info.get("action_type") == "fix":
            effective *= 0.4
        return effective * delta


class LintDeltaRubric(Rubric):
    def __init__(self, weight: float = 0.3):
        self.weight = weight

    def __call__(self, env, action, obs, reward, done, info):
        delta = env._current_lint_score - env._previous_lint_score
        effective = self.weight * 0.5
        if info.get("action_type") == "fix":
            effective *= 0.4
        return effective * delta


# --------------------------------------------------------------------------------
# 3. TERMINAL SUCCESS BONUS
# --------------------------------------------------------------------------------
class TerminalSuccessRubric(Rubric):
    def __call__(self, env, action, obs, reward, done, info):
        if info.get("action_type") != "fix":
            return 0.0
        score = 0.0
        if env._current_test_score > 0.95:
            score += 0.4
        elif env._current_test_score > 0.85:
            score += 0.2
        return score


# --------------------------------------------------------------------------------
# 4. EXPLORATION & DIVERSITY
# --------------------------------------------------------------------------------
class ExplorationRubric(Rubric):
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
# 5. ANTI‑HACKING & CONSISTENCY
# --------------------------------------------------------------------------------
class AntiHackingRubric(Rubric):
    def __call__(self, env, action, obs, reward, done, info):
        if info.get("action_type") != "fix":
            return 0.0
        score = 0.0
        if not env._tests_run:
            score -= 0.25
        if env._step_count < 2:
            score -= 0.1
        if env._tests_run and env._linter_run:
            score += 0.02
        return score


# --------------------------------------------------------------------------------
# 6. STEP PENALTY
# --------------------------------------------------------------------------------
class StepPenaltyRubric(Rubric):
    def __init__(self, penalty: float = -0.01):
        self.penalty = penalty

    def __call__(self, env, action, obs, reward, done, info):
        return self.penalty
