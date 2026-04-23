# models.py – Typed Models (Discriminated Unions, POMDP Separation)
from typing import Literal, Union, Annotated, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field, TypeAdapter, field_validator

# ----------------------------------------------------------------------
# Action classes (discriminated union)
# ----------------------------------------------------------------------
class Action(BaseModel):
    action_type: Literal["write_comment", "skip", "done", "ask_question",
                         "propose_fix", "execute", "inspect", "run_linter",
                         "run_tests", "query_docs"]

class WriteComment(Action):
    action_type: Literal["write_comment"] = "write_comment"
    comment_text: str = Field(..., min_length=1)

class Skip(Action):
    action_type: Literal["skip"] = "skip"

class Done(Action):
    action_type: Literal["done"] = "done"

class AskQuestion(Action):
    action_type: Literal["ask_question"] = "ask_question"
    question: str = Field(..., min_length=1)

class ProposeFix(Action):
    action_type: Literal["propose_fix"] = "propose_fix"
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

# ----------------------------------------------------------------------
# Observation (POMDP – what the agent sees)
# ----------------------------------------------------------------------
@dataclass(slots=True)
class Observation:
    code_snippet: str
    last_tool_output: str = ""
    step: int = 0
    done: bool = False

# ----------------------------------------------------------------------
# Reward (lightweight)
# ----------------------------------------------------------------------
@dataclass(slots=True)
class Reward:
    value: float

# ----------------------------------------------------------------------
# State (full environment state – not exposed to agent)
# ----------------------------------------------------------------------
@dataclass(slots=True)
class State:
    pr_title: str
    pr_description: str
    code_snippet: str
    comments: list[str]
    test_results: Optional[str]
    step: int
    done: bool