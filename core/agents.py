# core/agents.py
"""
Iterative Agent definitions for Planner, Developer, and Auditor - 3-agent architecture.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List
from langchain.prompts import ChatPromptTemplate

# ---------------------------
# STEP 1: Define Agent Class
# ---------------------------

class agent(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the expert.")
    name: str = Field(description="Name of the expert.", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    role: str = Field(description="Role of the expert in the process.")
    description: str = Field(description="Description of their focus, concerns, and motives.")

    @field_validator("name", mode="before")
    def sanitize_name(cls, value: str) -> str:
        return value.replace(" ", "").replace(".", "")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class IterativeTeam(BaseModel):
    agents: List[agent] = Field(
        description="List of 3 agents: Planner, Developer, Auditor for iterative workflow.",
        max_items=3,
        min_items=3
    )

# ---------------------------
# STEP 2: Define Prompt Templates
# ---------------------------

gen_iterative_agents_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are assembling a virtual team of 3 English speaking data science experts for an iterative workflow on the task: **{phase}**.

            The 3-agent iterative architecture works as follows:
            1. **Planner**: Breaks down tasks into subtasks and provides implementation instructions
            2. **Developer**: Implements code based on planner's instructions  
            3. **Auditor**: Reviews planner's instructions + developer's code, provides feedback for improvements to developer.

            Dataset context: {context}

            The team members are:
            - Planner (Strategic Thinker)
            - Developer (Code Implementer) 
            - Auditor (Quality Reviewer)

            Generate detailed personas for each agent focusing on their specific role in this iterative process.
            """
        ),
        ("user", "Generate detailed personas for the 3-agent iterative team."),
    ]
)

# ---------------------------
# STEP 3: Agent Generation Function
# ---------------------------

def generate_iterative_personas(context, phase, llm):
    """Generate the 3 agents for iterative workflow"""
    
    gen_agent_chain = gen_iterative_agents_prompt | llm.with_structured_output(
        IterativeTeam, method="function_calling"
    )
    
    return gen_agent_chain.invoke({
        "context": context,
        "phase": phase
    })

# ---------------------------
# STEP 4: Agent Role Definitions
# ---------------------------

ITERATIVE_AGENT_ROLES = {
    "Planner": "Strategic Planner",
    "Developer": "Python Developer", 
    "Auditor": "Quality Auditor"
}