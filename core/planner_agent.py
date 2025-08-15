# iterative/core/planner_agent.py

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel


# -------------------------------
# Output Schemas (mirror Swarm proposal shape)
# -------------------------------
class PlanBody(BaseModel):
    subtasks: List[str] = Field(default_factory=list)
    implementation_plan: List[str] = Field(default_factory=list)
    # keep rationale optional for explainability
    # strip empties / None (like ProposalOutput in Swarm)
    @field_validator("subtasks", "implementation_plan", mode="before")
    @classmethod
    def _coerce_list(cls, v):
        if not v:
            return []
        out = []
        for x in v:
            if isinstance(x, str):
                s = x.strip()
                if s:
                    out.append(s)
        return out

    # enforce 1:1 pairing; allow rationale to be shorter
    @model_validator(mode="after")
    def _align_lengths(self):
        n = min(len(self.subtasks), len(self.implementation_plan))
        self.subtasks = self.subtasks[:n]
        self.implementation_plan = self.implementation_plan[:n]
        if n == 0:
            raise ValueError("No valid (subtask, implementation) pairs.")
        return self


class PlanOutput(PlanBody):
    """Alias kept for symmetry with Swarm's ProposalOutput."""
    pass


# -------------------------------
# Planner agent config (parallels SwarmAgentData)
# -------------------------------
class PlannerAgentData(BaseModel):
    name: str = "Planner"
    role: str = "Task Planner"
    description: str = (
        "Break a high-level ML task (EDA, Feature Engineering, Model Building & Evaluation) "
        "into clear, index-aligned (subtask, implementation_plan) pairs."
    )


# -------------------------------
# Prompt (matches Swarm style & constraints)
# -------------------------------
_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are {name}, a {role}. {description}\n"
     "Context:\n{context}\n\nTask: {task}"),
    ("user", """Generate a list (5-12) of subtasks required to accomplish the task,
along with a detailed, specific implementation plan for each.

Steps already completed: {summary}

For each subtask, use this format:

Subtask: <Concise subtask name>
Implementation Plan: <Detailed, step-by-step approach; conceptual but actionable; this guides Python code generation>

Important Instructions (REQUIRED):
- For every subtask you list, immediately provide its implementation plan before moving to the next subtask.
- Strictly return TWO lists of EQUAL LENGTH:
  1) 'subtasks'                 — names of each subtask.
  2) 'implementation_plan'      — the corresponding implementation details for each subtask.
- Use as many subtasks as needed, but ensure each has a matching implementation.
- Avoid vague steps and avoid subtasks without implementations.
- No code snippets in the implementation plan.
- Implementation plan must be conceptual (not a list of steps).
- Implementation should be a paragraph (not a bullet list).
""")
])


# -------------------------------
# Planner Agent (no refine here)
# -------------------------------
class PlannerAgent:
    """
    Generates the **initial** plan only.
    In the iterative loop, any improvements proposed by the Auditor
    go straight to the Developer (no round-trip to the Planner).
    """
    def __init__(self, llm: BaseChatModel, data: Optional[PlannerAgentData] = None):
        self.llm = llm
        self.data = data or PlannerAgentData()

    def generate_plan(self, task: str, context: str, summary: str = "") -> PlanOutput:
        chain = _PLANNER_PROMPT | self.llm.with_structured_output(PlanOutput, method="function_calling")
        body: PlanOutput = chain.invoke({
            "name": self.data.name,
            "role": self.data.role,
            "description": self.data.description,
            "context": context,
            "task": task,
            "summary": summary or "No prior steps."
        })
        # validators ensure aligned lists & non-empty pairs
        return PlanBody(
            subtasks=body.subtasks,
            implementation_plan=body.implementation_plan
        )
