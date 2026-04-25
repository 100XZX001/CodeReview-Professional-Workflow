# redteam.py – AST-based bug injection (no dataset, always modifies given code)
import ast
import random
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict

# ----------------------------------------------------------------------
# AST-based bug injector
# ----------------------------------------------------------------------
class ASTBugInjector(ast.NodeTransformer):
    def __init__(self, bug_type: str):
        super().__init__()
        self.bug_type = bug_type
        self.modified = False

    def visit_If(self, node: ast.If):
        if self.bug_type == "null_check" and not self.modified:
            if node.body and len(node.body) == 1:
                self.modified = True
                return node.body[0]   # remove the if, directly execute body
        return self.generic_visit(node)

    def visit_For(self, node: ast.For):
        if self.bug_type == "off_by_one" and not self.modified:
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == "range":
                    # Change range(x) to range(1, x-1) to introduce off-by-one
                    new_iter = ast.Call(
                        func=ast.Name(id='range', ctx=ast.Load()),
                        args=[
                            ast.Constant(value=1),
                            ast.BinOp(
                                left=node.iter.args[0],
                                op=ast.Sub(),
                                right=ast.Constant(value=1)
                            )
                        ],
                        keywords=[]
                    )
                    node.iter = new_iter
                    self.modified = True
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        if self.bug_type == "wrong_operator" and not self.modified:
            if isinstance(node.op, ast.Add):
                node.op = ast.Sub()
                self.modified = True
        return self.generic_visit(node)

# ----------------------------------------------------------------------
# RedTeam Controller
# ----------------------------------------------------------------------
@dataclass
class RedTeam:
    task: str
    seed: Optional[int] = 42
    noise_prob: float = 0.2
    _random: random.Random = field(init=False)

    def __post_init__(self):
        self._random = random.Random(self.seed)

    def inject_bug(self, original_code: str) -> Tuple[str, str, str, str]:
        """
        Always modifies the given original_code using an AST bug.
        Returns (buggy_code, bug_type, description, oracle_fix).
        oracle_fix is the original (correct) code.
        """
        bug_types = ["null_check", "off_by_one", "wrong_operator"]
        bug_type = self._random.choice(bug_types)

        try:
            tree = ast.parse(original_code)
        except SyntaxError:
            # If the code can't be parsed, return it unchanged
            return original_code, "parse_error", "Syntax error in original code", original_code

        injector = ASTBugInjector(bug_type)
        modified_tree = injector.visit(tree)
        ast.fix_missing_locations(modified_tree)

        if injector.modified:
            buggy_code = ast.unparse(modified_tree)
            oracle_fix = original_code
            description = f"AST bug: {bug_type}"
        else:
            # Fallback: no injection possible (e.g., code doesn't contain the target pattern)
            buggy_code = original_code
            oracle_fix = original_code
            bug_type = "no_op"
            description = "No suitable code structure found for injection"

        # Add noise
        if self._random.random() < self.noise_prob:
            buggy_code += "\n# TODO: refactor later"

        return buggy_code, bug_type, description, oracle_fix
