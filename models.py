# models.py – Typed Models (Discriminated Unions, POMDP Separation)
from typing import Literal, Union, Annotated, Optional
from pydantic import BaseModel, Field, TypeAdapter, field_validator

# ----------------------------------------------------------------------
# Action classes (discriminated union)
# ----------------------------------------------------------------------
class Action(BaseModel):
    action_type: Literal["comment", "skip", "done", "question",
                         "fix", "execute", "inspect", "run_linter",
                         "run_tests", "query_docs"]

class WriteComment(Action):
    action_type: Literal["comment"] = "comment"
    comment_text: str = Field(..., min_length=1)

class Skip(Action):
    action_type: Literal["skip"] = "skip"

class Done(Action):
    action_type: Literal["done"] = "done"

class AskQuestion(Action):
    action_type: Literal["question"] = "question"
    question: str = Field(..., min_length=1)

class ProposeFix(Action):
    action_type: Literal["fix"] = "fix"
    fix_code: str = Field(..., min_length=1)
    @field_validator('fix_code')
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('fix_code cannot be empty')
        return v

class Execute(Action):
    action_type: Literal["execute"] = "execute"

class Inspect(Action):
    action_type: Literal["inspect"] = "inspect"

class RunLinter(Action):
    action_type: Literal["run_linter"] = "run_linter"

class RunTests(Action):
    action_type: Literal["run_tests"] = "run_tests"

class QueryDocs(Action):
    action_type: Literal["query_docs"] = "query_docs"
    query_topic: str = Field(..., min_length=1)

# Discriminated union for one‑line polymorphic deserialization
AnyAction = Annotated[
    Union[WriteComment, Skip, Done, AskQuestion, ProposeFix,
          Execute, Inspect, RunLinter, RunTests, QueryDocs],
    Field(discriminator='action_type')
]
action_adapter = TypeAdapter(AnyAction)


def map_to_env(action_type: str, content: Optional[str] = None) -> AnyAction:
    """
    Convert lightweight agent outputs into typed environment actions.
    Kept at module level so training/inference code can reuse one mapping.
    """
    if action_type == "run_tests":
        return RunTests()
    if action_type == "run_linter":
        return RunLinter()
    if action_type == "inspect":
        return Inspect()
    if action_type == "fix":
        return ProposeFix(fix_code=content or "")
    if action_type == "comment":
        return WriteComment(comment_text=content or "")
    if action_type == "question":
        return AskQuestion(question=content or "")
    if action_type == "query_docs":
        return QueryDocs(query_topic=content or "")
    if action_type == "done":
        return Done()
    return Skip()

# ----------------------------------------------------------------------
# Observation (POMDP – what the agent sees)
# ----------------------------------------------------------------------
class Observation(BaseModel):
    # Base schema model used by API metadata endpoints.
    # Keep this lightweight for compatibility with legacy callers.
    code_snippet: str
    last_tool_output: str = ""
    step: int = 0
    done: bool = False

# ----------------------------------------------------------------------
# Reward (lightweight)
# ----------------------------------------------------------------------
class Reward(BaseModel):
    value: float

# ----------------------------------------------------------------------
# State (full environment state – not exposed to agent)
# ----------------------------------------------------------------------
class State(BaseModel):
    pr_title: str
    pr_description: str
    code_snippet: str
    comments: list[str]
    test_results: Optional[str]
    step: int
    done: bool