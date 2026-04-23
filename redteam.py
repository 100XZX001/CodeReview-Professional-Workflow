# redteam.py – AST-based bug injection + dataset examples + noise
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
                return node.body[0]
        return self.generic_visit(node)

    def visit_For(self, node: ast.For):
        if self.bug_type == "off_by_one" and not self.modified:
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == "range":
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
# Dataset-driven realistic bugs
# ----------------------------------------------------------------------
DATASET_EXAMPLES: List[Dict] = [
    {
        "bug_type": "mutation_side_effect",
        "original": """def update_config_file(filepath, new_pair, default_config=None):
    if default_config is None:
        default_config = {}
    config = default_config.copy()
    key, value = new_pair
    config[key] = value
    return config""",
        "buggy": """def update_config_file(filepath, new_pair, default_config={}):
    config = default_config
    key, value = new_pair
    config[key] = value
    return config""",
    },
    {
        "bug_type": "infinite_loop",
        "original": """def retry(attempts, max_retries):
    while attempts < max_retries:
        success = False
        if success:
            break
        attempts += 1""",
        "buggy": """def retry(attempts, max_retries):
    while attempts < max_retries:
        success = False
        if success:
            break""",
    }
]

# ----------------------------------------------------------------------
# RedTeam Controller
# ----------------------------------------------------------------------
@dataclass
class RedTeam:
    task: str
    seed: Optional[int] = 42
    noise_prob: float = 0.2
    dataset_prob: float = 0.4   # probability of using dataset instead of AST
    _random: random.Random = field(init=False)

    def __post_init__(self):
        self._random = random.Random(self.seed)

    def inject_bug(self, original_code: str) -> Tuple[str, str, str, str]:
        """
        Returns:
        (buggy_code, bug_type, description, oracle_fix)
        """
        # Decide injection mode
        use_dataset = self._random.random() < self.dataset_prob

        if use_dataset:
            example = self._random.choice(DATASET_EXAMPLES)
            buggy_code = example["buggy"]
            oracle_fix = example["original"]
            bug_type = example["bug_type"]
            description = f"Dataset bug: {bug_type}"
        else:
            bug_types = ["null_check", "off_by_one", "wrong_operator"]
            bug_type = self._random.choice(bug_types)

            try:
                tree = ast.parse(original_code)
            except SyntaxError:
                return original_code, "parse_error", "Syntax error", original_code

            injector = ASTBugInjector(bug_type)
            modified_tree = injector.visit(tree)
            ast.fix_missing_locations(modified_tree)

            if injector.modified:
                buggy_code = ast.unparse(modified_tree)
                oracle_fix = original_code
            else:
                # fallback: no injection
                buggy_code = original_code
                oracle_fix = original_code
                bug_type = "no_op"
                description = "No modification applied"

            description = f"AST bug: {bug_type}"

        # Add noise
        if self._random.random() < self.noise_prob:
            buggy_code += "\n# TODO: refactor later"

        return buggy_code, bug_type, description, oracle_fix