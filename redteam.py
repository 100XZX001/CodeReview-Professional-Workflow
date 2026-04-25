# redteam.py – Task‑aware bug injection (25 bugs, 5 difficulty levels)
import ast
import random
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict

# ----------------------------------------------------------------------
# 1. AST Bug Injector (extended for all simple bugs)
# ----------------------------------------------------------------------
class ASTBugInjector(ast.NodeTransformer):
    def __init__(self, bug_type: str):
        super().__init__()
        self.bug_type = bug_type
        self.modified = False

    # --- Easy: null_check, simple_typo, string_index, default_value, empty_return ---
    def visit_If(self, node: ast.If):
        # null_check: remove the if-guard
        if self.bug_type == "null_check" and not self.modified:
            if node.body and len(node.body) == 1:
                self.modified = True
                return node.body[0]
        # division_by_zero_empty: remove the empty check
        if self.bug_type == "division_by_zero_empty" and not self.modified:
            # pattern: if not data: return 0  – we delete the entire if
            if (isinstance(node.test, ast.UnaryOp) and
                isinstance(node.test.op, ast.Not) and
                isinstance(node.test.operand, ast.Name)):
                self.modified = True
                return None  # signal to remove this node from parent
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if self.bug_type == "simple_typo" and not self.modified:
            if node.id == "users":
                self.modified = True
                return ast.Name(id="usres", ctx=node.ctx)
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if self.bug_type == "string_index" and not self.modified:
            if isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Constant):
                old_val = node.slice.value.value
                if isinstance(old_val, int):
                    self.modified = True
                    node.slice = ast.Index(value=ast.Constant(value=old_val + 1))
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # default_value: change dict.get(key) to dict[key] (no default)
        if self.bug_type == "default_value" and not self.modified:
            if (isinstance(node.func, ast.Attribute) and
                node.func.attr == "get" and len(node.args) == 1):
                self.modified = True
                return ast.Subscript(
                    value=node.func.value,
                    slice=ast.Index(value=node.args[0]),
                    ctx=node.ctx
                )
        # abs_usage: remove abs()
        if self.bug_type == "abs_usage" and not self.modified:
            if isinstance(node.func, ast.Name) and node.func.id == "abs":
                self.modified = True
                return node.args[0]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # empty_return: insert a premature return None
        if self.bug_type == "empty_return" and not self.modified:
            self.modified = True
            node.body.insert(0, ast.Return(value=ast.Constant(value=None)))
        return self.generic_visit(node)

    # --- Medium: off_by_one, loop_skip, sign_error, swap_args, uninitialised_var ---
    def visit_For(self, node: ast.For):
        if (self.bug_type in ("off_by_one", "loop_skip")) and not self.modified:
            if (isinstance(node.iter, ast.Call) and
                isinstance(node.iter.func, ast.Name) and
                node.iter.func.id == "range"):
                if self.bug_type == "off_by_one":
                    new_iter = ast.Call(
                        func=ast.Name(id='range', ctx=ast.Load()),
                        args=[
                            ast.Constant(value=1),
                            ast.BinOp(left=node.iter.args[0], op=ast.Sub(), right=ast.Constant(value=1))
                        ],
                        keywords=[]
                    )
                    node.iter = new_iter
                    self.modified = True
                elif self.bug_type == "loop_skip" and len(node.iter.args) == 1:
                    new_iter = ast.Call(
                        func=ast.Name(id='range', ctx=ast.Load()),
                        args=[ast.BinOp(left=node.iter.args[0], op=ast.Sub(), right=ast.Constant(value=1))],
                        keywords=[]
                    )
                    node.iter = new_iter
                    self.modified = True
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        # sign_error: flip Add/Sub, wrong_operator: Add->Sub, float_precision: Div->FloorDiv
        if not self.modified:
            if self.bug_type in ("wrong_operator", "sign_error"):
                if isinstance(node.op, ast.Add):
                    node.op = ast.Sub()
                    self.modified = True
                elif isinstance(node.op, ast.Sub):
                    node.op = ast.Add()
                    self.modified = True
            elif self.bug_type == "float_precision" and isinstance(node.op, ast.Div):
                node.op = ast.FloorDiv()
                self.modified = True
        return self.generic_visit(node)

    def visit_arguments(self, node: ast.arguments):
        # swap_args: swap first two arguments of a function
        if self.bug_type == "swap_args" and not self.modified and len(node.args) >= 2:
            self.modified = True
            node.args[0], node.args[1] = node.args[1], node.args[0]
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # uninitialised_var: remove an assignment statement (replaced with Pass)
        if self.bug_type == "uninitialised_var" and not self.modified:
            self.modified = True
            return ast.Pass()
        return self.generic_visit(node)

# ----------------------------------------------------------------------
# 2. Bug database (25 bugs, categorized by difficulty)
# ----------------------------------------------------------------------
BUG_DB = {
    "easy": {
        "null_check":    {"type": "ast", "bug_type": "null_check"},
        "simple_typo":   {"type": "ast", "bug_type": "simple_typo"},
        "string_index":  {"type": "ast", "bug_type": "string_index"},
        "default_value": {"type": "ast", "bug_type": "default_value"},
        "empty_return":  {"type": "ast", "bug_type": "empty_return"},
    },
    "medium": {
        "off_by_one":     {"type": "ast", "bug_type": "off_by_one"},
        "loop_skip":      {"type": "ast", "bug_type": "loop_skip"},
        "sign_error":     {"type": "ast", "bug_type": "sign_error"},
        "swap_args":      {"type": "ast", "bug_type": "swap_args"},
        "uninitialised_var": {"type": "ast", "bug_type": "uninitialised_var"},
    },
    "hard": {
        "division_by_zero_empty": {"type": "ast", "bug_type": "division_by_zero_empty"},
        "division_by_zero_zero":  {"type": "ast", "bug_type": "division_by_zero_empty"},  # same injector
        "float_precision":        {"type": "ast", "bug_type": "float_precision"},
        "abs_usage":              {"type": "ast", "bug_type": "abs_usage"},
        "round_error":            {"type": "ast", "bug_type": "round_error"},  # can be extended
    },
    "harder": {
        "missing_lock": {
            "type": "template",
            "buggy": "counter = 0\ndef increment():\n    global counter\n    counter += 1",
            "oracle": "counter = 0\nimport threading\nlock = threading.Lock()\ndef increment():\n    global counter\n    with lock:\n        counter += 1",
        },
        "double_lock": {
            "type": "template",
            "buggy": "import threading\nlock = threading.Lock()\ndef do_work():\n    lock.acquire()\n    lock.acquire()\n    print('working')\n    lock.release()",
            "oracle": "import threading\nlock = threading.Lock()\ndef do_work():\n    with lock:\n        print('working')",
        },
        "global_nonatomic": {
            "type": "template",
            "buggy": "count = 0\ndef add():\n    global count\n    count = count + 1",
            "oracle": "count = 0\ndef add():\n    global count\n    count += 1",
        },
        "thread_safe_list": {
            "type": "template",
            "buggy": "import threading\nitems = []\ndef append_item(item):\n    items.append(item)",
            "oracle": "import threading\nitems = []\nlock = threading.Lock()\ndef append_item(item):\n    with lock:\n        items.append(item)",
        },
        "volatile_read": {
            "type": "template",
            "buggy": "import threading\nstop = False\ndef worker():\n    while not stop:\n        pass",
            "oracle": "import threading\nstop = False\nlock = threading.Lock()\ndef worker():\n    while True:\n        with lock:\n            if stop:\n                break",
        },
    },
    "hardest": {
        "deadlock_order": {
            "type": "template",
            "buggy": "import threading\nlock1 = threading.Lock()\nlock2 = threading.Lock()\ndef thread1():\n    with lock1:\n        with lock2:\n            pass\ndef thread2():\n    with lock2:\n        with lock1:\n            pass",
            "oracle": "import threading\nlock1 = threading.Lock()\nlock2 = threading.Lock()\ndef thread1():\n    with lock1:\n        with lock2:\n            pass\ndef thread2():\n    with lock1:\n        with lock2:\n            pass",
        },
        "nested_lock_timeout": {
            "type": "template",
            "buggy": "import threading\nlock = threading.Lock()\ndef work():\n    lock.acquire()\n    # critical section\n    lock.release()",
            "oracle": "import threading\nlock = threading.Lock()\ndef work():\n    if lock.acquire(timeout=1):\n        try:\n            # critical section\n        finally:\n            lock.release()",
        },
        "fork_join": {
            "type": "template",
            "buggy": "import threading\ndef worker():\n    pass\nt = threading.Thread(target=worker)\nt.start()",
            "oracle": "import threading\ndef worker():\n    pass\nt = threading.Thread(target=worker)\nt.start()\nt.join()",
        },
        "mutex_release": {
            "type": "template",
            "buggy": "import threading\nlock = threading.Lock()\ndef thread_A():\n    lock.acquire()\n    lock.release()\ndef thread_B():\n    lock.release()",
            "oracle": "import threading\nlock = threading.Lock()\ndef thread_A():\n    with lock:\n        pass\ndef thread_B():\n    with lock:\n        pass",
        },
        "race_on_init": {
            "type": "template",
            "buggy": "import threading\nitems = []\ndef init():\n    global items\n    items = [1,2,3]\nt = threading.Thread(target=init)\nt.start()\nprint(items)",
            "oracle": "import threading\nitems = []\ndef init():\n    global items\n    items = [1,2,3]\nt = threading.Thread(target=init)\nt.start()\nt.join()\nprint(items)",
        },
    },
}

# ----------------------------------------------------------------------
# 3. Derived helpers
# ----------------------------------------------------------------------
TASK_BUG_MAP = {level: list(bugs.keys()) for level, bugs in BUG_DB.items()}

TEMPLATE_BUGS = {}
for level, bugs in BUG_DB.items():
    for bug_id, bug in bugs.items():
        if bug["type"] == "template":
            TEMPLATE_BUGS[bug_id] = (bug["buggy"], bug["oracle"])

# ----------------------------------------------------------------------
# 4. RedTeam Controller (task‑aware)
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
        Returns: (buggy_code, bug_type, description, oracle_fix)
        Selects a bug appropriate for the task difficulty.
        """
        bug_list = TASK_BUG_MAP.get(self.task, ["null_check"])
        bug_type = self._random.choice(bug_list)

        # Template bug: return hardcoded buggy + oracle
        if bug_type in TEMPLATE_BUGS:
            buggy_code, oracle_code = TEMPLATE_BUGS[bug_type]
            description = f"Template bug: {bug_type}"
            if self._random.random() < self.noise_prob:
                buggy_code += "\n# TODO: refactor later"
            return buggy_code, bug_type, description, oracle_code

        # AST injection
        try:
            tree = ast.parse(original_code)
        except SyntaxError:
            return original_code, "parse_error", "Syntax error in original code", original_code

        injector = ASTBugInjector(bug_type)
        modified_tree = injector.visit(tree)
        ast.fix_missing_locations(modified_tree)

        if injector.modified:
            buggy_code = ast.unparse(modified_tree)
            oracle_fix = original_code
            description = f"AST bug: {bug_type}"
        else:
            buggy_code = original_code
            oracle_fix = original_code
            bug_type = "no_op"
            description = "No suitable code structure found for injection"

        if self._random.random() < self.noise_prob:
            buggy_code += "\n# TODO: refactor later"

        return buggy_code, bug_type, description, oracle_fix
