# iterative/core/auditor_module.py

from typing import Optional
from pydantic import BaseModel, Field, model_validator
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel


# -------- Output schema (minimal) --------
class AuditDecisionOutput(BaseModel):
    accept: bool = Field(
        description="True if the current result is good enough; False if a revision is needed."
    )
    reason: str = Field(default="", description="Short reason for the decision.")
    improved_plan: Optional[str] = Field(
        default="",
        description="If accept=False, a single improved implementation plan string for the developer."
    )

    @model_validator(mode="after")
    def _require_plan_when_rejected(self):
        if not self.accept:
            if not (self.improved_plan and self.improved_plan.strip()):
                raise ValueError("accept=False requires a non-empty improved_plan string.")
        else:
            # keep it clean when accepted
            self.improved_plan = ""
        return self


# -------- Prompt (kept super simple) --------
_AUDITOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an ML Auditor. Decide if the developer's execution result is adequate for the subtask and the implementation plan. "
     "If adequate, set accept=True and leave improved_plan empty. "
     "If not adequate, set accept=False and RETURN ONE improved implementation plan as a single paragraph. "
     "The improved plan must be concrete and actionable also contain the original plan, but DO NOT write code."),
    ("user", """\
Task Phase: {task_phase}
Subtask: {subtask}

Original Implementation Plan:
{plan_text}

Execution Result (developer output/logs):
{execution_result}

Answer with:
- accept (True/False)
- reason (short)
- improved_plan (ONLY if accept=False; one paragraph; no code)
""")
])


# -------- Auditor (minimal) --------
class Auditor:
    """
    Minimal Auditor:
      - If work is acceptable → accept=True, improved_plan="".
      - If not acceptable → accept=False, improved_plan="<one paragraph to guide the developer>".
    """
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._prompt = _AUDITOR_PROMPT

    def review(
        self,
        subtask: str,
        plan_text: str,
        execution_result: str,
        task_phase: str = "",
    ) -> AuditDecisionOutput:
        chain = self._prompt | self.llm.with_structured_output(AuditDecisionOutput, method="function_calling")
        return chain.invoke({
            "task_phase": task_phase or "Unknown",
            "subtask": subtask,
            "plan_text": plan_text or "",
            "execution_result": execution_result or "",
        })
