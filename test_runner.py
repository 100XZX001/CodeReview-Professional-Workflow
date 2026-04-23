# test_runner.py – Full production version with continuous scoring, dynamic function detection, and randomised tests
import subprocess
import tempfile
import os
import json
import ast
import random
import sys
from typing import Tuple, List, Any, Optional
from dataclasses import dataclass

@dataclass
class TestRunner:
    bug_id: str
    timeout_sec: int = 5
    max_memory_mb: int = 256
    fuzz_rounds: int = 3   # number of random test cases per bug

    def run_tests(self, fix_code: str) -> Tuple[float, str]:
        """
        Returns (score, output_message) where score is proportion of passed tests (0.0–1.0).
        """
        # 1. Detect the function defined in the agent's code (dynamic)
        func_name = self._get_defined_function_name(fix_code)
        if not func_name:
            return 0.0, "No function definition found in agent code"

        # 2. Generate the test script (includes fixed + fuzzed test cases)
        test_script = self._generate_test_script(fix_code, func_name)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(test_script)
            tmp_path = f.name

        try:
            # Resource limiting (Linux only; fallback otherwise)
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (self.max_memory_mb * 1024 * 1024, self.max_memory_mb * 1024 * 1024))
            except Exception:
                pass

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                encoding='utf-8'
            )
            # Parse JSON output
            try:
                data = json.loads(result.stdout.strip())
                passed = data.get("passed", 0)
                total = data.get("total", 1)
                score = passed / total if total > 0 else 0.0
                return score, result.stdout.strip()
            except json.JSONDecodeError:
                # Fallback: look for "True" (legacy)
                if "True" in result.stdout:
                    return 1.0, result.stdout
                return 0.0, result.stdout
        except subprocess.TimeoutExpired:
            return 0.0, "Test execution timed out"
        except Exception as e:
            return 0.0, f"Unexpected error: {str(e)}"
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    def _get_defined_function_name(self, code: str) -> Optional[str]:
        """Extract the first function name defined in the code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except SyntaxError:
            pass
        return None

    def _generate_test_script(self, fix_code: str, func_name: str) -> str:
        """Generate a test script that runs fixed + fuzzed test cases and outputs JSON."""
        test_cases = self._get_test_cases(func_name)
        fuzzed_cases = self._generate_fuzzed_cases(func_name)
        all_cases = test_cases + fuzzed_cases

        lines = []
        lines.append(fix_code)
        lines.append("")
        lines.append("import json")
        lines.append("")
        lines.append("def run_tests():")
        lines.append(f"    test_cases = {json.dumps(all_cases)}")
        lines.append("    passed = 0")
        lines.append("    total = len(test_cases)")
        lines.append("    for args, expected in test_cases:")
        lines.append(f"        try:")
        lines.append(f"            result = {func_name}(*args) if isinstance(args, list) else {func_name}(args)")
        lines.append(f"            if result == expected:")
        lines.append(f"                passed += 1")
        lines.append(f"        except Exception:")
        lines.append(f"            pass")
        lines.append("    return {'passed': passed, 'total': total}")
        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append("    result = run_tests()")
        lines.append("    print(json.dumps(result))")
        return "\n".join(lines)

    def _get_test_cases(self, func_name: str) -> List[Tuple[List[Any], Any]]:
        """
        Return a list of (arguments, expected_output) for the given bug_id.
        Uses the actual function name (dynamic) for consistency.
        """
        if self.bug_id == "null_check":
            return [
                ([{"users": {"alice": "Alice"}, "id": "bob"}], None),   # missing key should not crash
                ([{"users": {"alice": "Alice"}, "id": "alice"}], "Alice"),
            ]
        elif self.bug_id == "off_by_one":
            return [
                ([[1,2,3,4]], 4),
                ([[]], 0),
            ]
        elif self.bug_id == "division_by_zero":
            return [
                ([[]], 0),
                ([[1,2,3]], 2.0),
            ]
        elif self.bug_id == "wrong_operator":
            return [
                ([5,3], 8),
                ([-1,1], 0),
            ]
        else:
            # For missing_lock, deadlock_order, etc., return empty list (will be handled gracefully)
            return []

    def _generate_fuzzed_cases(self, func_name: str) -> List[Tuple[List[Any], Any]]:
        """
        Generate random test cases to prevent memorisation.
        Only for bugs where meaningful fuzzing is possible.
        """
        cases = []
        if self.bug_id == "null_check":
            # Random users dictionary and random ids
            for _ in range(self.fuzz_rounds):
                users = {f"user_{i}": f"name_{i}" for i in range(random.randint(1, 5))}
                # Pick existing or missing key
                if random.random() > 0.5:
                    key = random.choice(list(users.keys()))
                    expected = users[key]
                else:
                    key = "missing_" + str(random.randint(100, 999))
                    expected = None
                cases.append(([{"users": users, "id": key}], expected))
        elif self.bug_id == "off_by_one":
            for _ in range(self.fuzz_rounds):
                length = random.randint(0, 10)
                arr = list(range(length))
                cases.append(([arr], length))
        elif self.bug_id == "division_by_zero":
            for _ in range(self.fuzz_rounds):
                length = random.randint(0, 10)
                data = [random.randint(-100, 100) for _ in range(length)]
                expected = sum(data)/length if length else 0
                cases.append(([data], expected))
        elif self.bug_id == "wrong_operator":
            for _ in range(self.fuzz_rounds):
                a = random.randint(-100, 100)
                b = random.randint(-100, 100)
                cases.append(([a, b], a + b))
        return cases