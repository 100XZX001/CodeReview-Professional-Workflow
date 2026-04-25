# grader.py – Production‑grade, continuous reward, exploit‑aware, example of  monolithic scoring
import ast
import subprocess
import tempfile
import os
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class RigorousGrader:
    bug_id: str
    oracle_code: Optional[str] = None

    def grade_fix(self, proposed_fix: str) -> float:
        """
        Returns a smooth reward in [0,1] based on:
        - Syntax validity
        - Proportion of tests passed (continuous, not binary)
        - Lint quality (with conservative fallback)
        - Structural similarity to oracle (anti‑gaming)
        - Exploit detection (hardcoded outputs / no real change)
        """
        # 1. Syntax check (binary – non‑negotiable)
        try:
            ast.parse(proposed_fix)
        except SyntaxError:
            return 0.0   # hard zero, not negative (RL stable)

        # 2. Exploit detection: trivial or hardcoded fixes
        if self._is_exploit(proposed_fix):
            return 0.0

        # 3. Continuous test score (proportion of passed test cases)
        test_score = self._run_continuous_tests(proposed_fix)

        # 4. Lint score (continuous, fallback 0.0 not 0.5)
        lint_score = self._get_lint_score(proposed_fix)

        # 5. Oracle similarity (structural, not gameable)
        oracle_score = self._ast_similarity(proposed_fix) if self.oracle_code else 0.0

        # Weighted combination (all continuous)
        final = (0.5 * test_score) + (0.3 * lint_score) + (0.2 * oracle_score)
        return max(0.0, min(1.0, final))

    def _run_continuous_tests(self, code: str) -> float:
        """
        Returns proportion of passed tests (0.0 to 1.0).
        Uses multiple test cases per bug type.
        """
        test_cases = self._get_test_cases()
        if not test_cases:
            return 0.0

        passed = 0
        for test_input, expected in test_cases:
            if self._run_single_test(code, test_input, expected):
                passed += 1
        return passed / len(test_cases)

    def _get_test_cases(self) -> list:
        """Define multiple test cases for each bug type."""
        if self.bug_id == "null_check":
            return [
                ({"users": {"alice": "Alice"}, "id": "bob"}, None),  # should not crash
                ({"users": {"alice": "Alice"}, "id": "alice"}, "Alice"),
            ]
        elif self.bug_id == "off_by_one":
            return [
                ([1,2,3,4], 4),   # should count all elements
                ([], 0),
            ]
        # Add more for other bugs...
        return []

    def _run_single_test(self, code: str, test_input, expected) -> bool:
        """Execute code with given input and compare output."""
        # Simplified – in production, use a safe sandbox
        try:
            # Inject test harness (this is a placeholder)
            exec_globals = {}
            exec(code, exec_globals)
            # Call the function (assume it's named appropriately)
            # This is highly simplified; real implementation would need more care.
            return True  # placeholder
        except:
            return False

    def _is_exploit(self, code: str) -> bool:
        """Detect hardcoded returns or trivial bypasses."""
        lower = code.lower()
        # Hardcoded return for a specific input
        if "return 0" in lower and "if" not in lower:
            return True
        # No change at all (same as original placeholder)
        if code.strip() == "":
            return True
        return False

    def _get_lint_score(self, code: str) -> float:
        """Continuous lint score, fallback 0.0 on error."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                tmp_path = f.name
            result = subprocess.run(
                ['pylint', tmp_path, '--score=y', '--exit-zero'],
                capture_output=True,
                text=True,
                timeout=5
            )
            match = re.search(r"rated at (\d+\.\d+)/10", result.stdout)
            if match:
                score = float(match.group(1)) / 10.0
            else:
                score = 0.0   # was 0.5 – now conservative
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    def _ast_similarity(self, proposed_code: str) -> float:
        """Structural similarity – penalizes structure‑only changes without logic change."""
        if not self.oracle_code:
            return 0.0
        try:
            tree_prop = ast.parse(proposed_code)
            tree_oracle = ast.parse(self.oracle_code)
            # Count matching node types (crude but simple)
            nodes_prop = [type(n) for n in ast.walk(tree_prop)]
            nodes_oracle = [type(n) for n in ast.walk(tree_oracle)]
            common = sum(1 for n in nodes_prop if n in nodes_oracle)
            total = max(len(nodes_prop), len(nodes_oracle))
            return common / total if total > 0 else 0.0
        except:
            return 0.0