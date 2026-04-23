# cell 7 author.py – Final production version: stateful, evidence-driven, belief tracking

import re
import ast
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class PersonaAuthor:
    """
    Simulates a human developer with:
    - Continuous belief (confidence)
    - Evidence-based reasoning
    - Conversation memory
    - Code inspection awareness
    """

    personality: str = "defensive"   # defensive | junior | collaborative
    max_persuasion_rounds: int = 5

    # Evidence weights
    weight_test_pass: float = 0.5
    weight_lint_clean: float = 0.2
    weight_doc_found: float = 0.15
    weight_explanation_quality: float = 0.15

    # Personality thresholds
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "defensive": 0.7,
        "junior": 0.3,
        "collaborative": 0.5,
    })

    # Internal state
    _confidence: float = 0.0
    _conversation: List[Dict[str, Any]] = field(default_factory=list)
    _pushback_count: int = 0
    _last_evidence_score: float = 0.0
    _stagnation_counter: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __post_init__(self):
        self.reset()

    def reset(self):
        self._confidence = 0.0
        self._conversation.clear()
        self._pushback_count = 0
        self._last_evidence_score = 0.0
        self._stagnation_counter = 0

    # ------------------------------------------------------------------
    # Main interaction
    # ------------------------------------------------------------------
    def respond(self,
                agent_comment: str = "",
                agent_question: str = "",
                test_results: Optional[str] = None,
                lint_results: Optional[str] = None,
                doc_results: Optional[str] = None,
                proposed_fix: Optional[str] = None,
                original_code: Optional[str] = None) -> str:

        # Store conversation
        self._conversation.append({
            "comment": agent_comment,
            "question": agent_question,
            "test": test_results,
            "lint": lint_results,
            "docs": doc_results
        })

        # Extract structured evidence
        evidence = self._extract_evidence(test_results, lint_results, doc_results)

        # Code inspection
        if proposed_fix and original_code:
            evidence["code_change"] = self._inspect_code(proposed_fix, original_code)

        # Explanation score
        text = (agent_comment + " " + agent_question).lower()
        explanation_score = self._score_explanation(text)

        # Compute evidence score
        evidence_score = (
            self.weight_test_pass * evidence.get("test_pass_ratio", 0.0) +
            self.weight_lint_clean * (1 - min(1.0, evidence.get("lint_errors", 0)/10)) +
            self.weight_doc_found * (1.0 if evidence.get("doc_found") else 0.0) +
            self.weight_explanation_quality * explanation_score
        )

        evidence_score = max(0.0, min(1.0, evidence_score))

        # Detect improvement
        delta = evidence_score - self._last_evidence_score
        self._last_evidence_score = evidence_score

        if delta > 0.05:
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1

        # Update belief (momentum)
        lr = 0.3
        self._confidence = (1 - lr) * self._confidence + lr * evidence_score

        # Penalise stagnation
        if self._stagnation_counter >= 2:
            self._confidence *= 0.9

        # Decision
        threshold = self.thresholds.get(self.personality, 0.5)

        if self._confidence >= threshold or self._pushback_count >= self.max_persuasion_rounds:
            return "Alright, I'm convinced. Let's proceed with your fix."

        # Otherwise push back
        self._pushback_count += 1
        return self._generate_pushback(evidence, text)

    # ------------------------------------------------------------------
    # Evidence extraction
    # ------------------------------------------------------------------
    def _extract_evidence(self, test_results, lint_results, doc_results):
        evidence = {
            "test_pass_ratio": 0.0,
            "lint_errors": 0,
            "doc_found": False
        }

        # Parse test results
        if test_results:
            match = re.search(r'(\d+)\s*/\s*(\d+)', test_results)
            if match:
                p, t = int(match.group(1)), int(match.group(2))
                evidence["test_pass_ratio"] = p / t if t else 0.0
            elif "true" in test_results.lower():
                evidence["test_pass_ratio"] = 1.0
            elif "false" in test_results.lower():
                evidence["test_pass_ratio"] = 0.0

        # Lint errors
        if lint_results:
            evidence["lint_errors"] = len(re.findall(r'error', lint_results.lower()))

        # Docs
        if doc_results and "no relevant" not in doc_results.lower():
            evidence["doc_found"] = True

        return evidence

    # ------------------------------------------------------------------
    # Explanation scoring
    # ------------------------------------------------------------------
    def _score_explanation(self, text: str) -> float:
        score = 0.0

        if "because" in text or "therefore" in text:
            score += 0.3
        if "test" in text or "example" in text:
            score += 0.2
        if len(text.split()) > 30:
            score += 0.2
        if "error" in text or "fix" in text:
            score += 0.1

        return min(1.0, score)

    # ------------------------------------------------------------------
    # Code inspection
    # ------------------------------------------------------------------
    def _inspect_code(self, new_code: str, old_code: str) -> float:
        try:
            t1 = ast.parse(old_code)
            t2 = ast.parse(new_code)

            n1 = len(list(ast.walk(t1)))
            n2 = len(list(ast.walk(t2)))

            change = abs(n2 - n1) / max(n1, 1)
            return min(1.0, change)
        except:
            return 0.0

    # ------------------------------------------------------------------
    # Pushback generator
    # ------------------------------------------------------------------
    def _generate_pushback(self, evidence, text):
        if evidence["test_pass_ratio"] < 0.5:
            return "Tests are still failing. Show a passing case."

        if evidence["lint_errors"] > 0:
            return f"There are {evidence['lint_errors']} lint errors. Fix them."

        if not evidence["doc_found"]:
            return "Provide documentation or reference."

        if "because" not in text:
            return "Explain why this works."

        if len(text.split()) < 20:
            return "Too brief. Expand your reasoning."

        return "Not convinced yet. Give a concrete example."

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    def get_negotiation_score(self) -> float:
        penalty = 0.1 * min(3, self._pushback_count)
        return max(0.0, min(1.0, self._confidence - penalty))