# core/dialogue.py
"""
Iterative dialogue system for 3-agent workflow: Planner → Developer → Auditor → Developer
"""

from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from core.agents import agent
from langchain_openai import ChatOpenAI
from util.prompt_library import ITERATIVE_PROMPT_TEMPLATES
from util.pipeline_config import PIPELINE_CONSTANTS as PIPELINE_CONFIG
from util.pipeline_util import format_pipeline_config_for_prompt

class IterativeDialogueSimulator:
    """
    Manages the 4-step iterative process:
    1. Planner creates subtasks and instructions
    2. Developer implements code
    3. Auditor reviews and provides feedback
    4. Developer refines based on feedback
    """

    def __init__(self, agents: List[agent], context: str, llm: ChatOpenAI, llm_coder: ChatOpenAI, topic: str, summary: str):
        self.agents = agents
        self.context = context
        self.llm = llm
        self.llm_coder = llm_coder
        self.topic = topic
        self.summary = summary
        
        # Map agents by role for easy access
        self.planner = self.get_agent_by_role("Planner")
        self.developer = self.get_agent_by_role("Developer") 
        self.auditor = self.get_agent_by_role("Auditor")

    def get_agent_by_role(self, role):
        """Get agent by role name"""
        return next(a for a in self.agents if role.lower() in a.role.lower())

    def run_iterative_process(self, subtask_list: List[str]) -> Dict[str, Any]:
        """
        Execute the complete 4-step iterative process for all subtasks
        """
        print(f"\n🔄 Starting iterative process for: '{self.topic}'")
        
        # Step 1: Planner creates implementation plan
        planner_output = self.step1_planner_planning(subtask_list)
        print (planner_output)
        # Step 2: Developer implements initial code
        developer_output = self.step2_developer_implementation(planner_output)
        print(developer_output)
        # Step 3: Auditor reviews and provides feedback
        auditor_output = self.step3_auditor_review(planner_output, developer_output)
        print(auditor_output)
        # Step 4: Developer refines based on feedback
        final_output = self.step4_developer_refinement(planner_output, developer_output, auditor_output)
        print(final_output)
        
        return {
            "planner_output": planner_output,
            "initial_developer_output": developer_output,
            "auditor_feedback": auditor_output,
            "final_developer_output": final_output,
            "process_complete": True
        }

    def step1_planner_planning(self, subtask_list: List[str]) -> Dict[str, Any]:
        """
        Step 1: Planner breaks down subtasks and provides implementation instructions
        """
        print(f"\n📋 Step 1: {self.planner.name} (Planner) creating implementation plan...")
        
        config_text = format_pipeline_config_for_prompt(PIPELINE_CONFIG)
        
        try:
            template = ITERATIVE_PROMPT_TEMPLATES["planner"][self.topic]
        except KeyError:
            template = ITERATIVE_PROMPT_TEMPLATES["planner"]["default"]

        prompt = template.format(
            planner_name=self.planner.name,
            planner_description=self.planner.description,
            topic=self.topic,
            context=self.context,
            summary=self.summary if self.summary else "No previous steps.",
            subtask_list="\n".join(f"- {task}" for task in subtask_list),
            pipeline_config=config_text
        )

        messages = [
            SystemMessage(content="You are a strategic planner breaking down data science tasks."),
            HumanMessage(content=prompt)
        ]

        response = self.llm(messages)
        
        return {
            "agent": self.planner.name,
            "role": "Planner",
            "planning_instructions": response.content.strip(),
            "subtasks_planned": subtask_list
        }

    def step2_developer_implementation(self, planner_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Developer implements code based on planner's instructions
        """
        print(f"\n💻 Step 2: {self.developer.name} (Developer) implementing code...")
        
        config_text = format_pipeline_config_for_prompt(PIPELINE_CONFIG)
        
        try:
            template = ITERATIVE_PROMPT_TEMPLATES["developer_initial"][self.topic]
        except KeyError:
            template = ITERATIVE_PROMPT_TEMPLATES["developer_initial"]["default"]

        prompt = template.format(
            developer_name=self.developer.name,
            developer_description=self.developer.description,
            topic=self.topic,
            context=self.context,
            planner_instructions=planner_output["planning_instructions"],
            pipeline_config=config_text
        )

        messages = [
            SystemMessage(content="You are a Python developer implementing data science solutions."),
            HumanMessage(content=prompt)
        ]

        response = self.llm_coder(messages)
        
        return {
            "agent": self.developer.name,
            "role": "Developer", 
            "implementation": response.content.strip(),
            "based_on_planner": planner_output["agent"]
        }

    def step3_auditor_review(self, planner_output: Dict[str, Any], developer_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3: Auditor reviews planner's instructions and developer's implementation
        """
        print(f"\n🔍 Step 3: {self.auditor.name} (Auditor) reviewing implementation...")
        
        config_text = format_pipeline_config_for_prompt(PIPELINE_CONFIG)
        
        try:
            template = ITERATIVE_PROMPT_TEMPLATES["auditor_review"][self.topic]
        except KeyError:
            template = ITERATIVE_PROMPT_TEMPLATES["auditor_review"]["default"]

        prompt = template.format(
            auditor_name=self.auditor.name,
            auditor_description=self.auditor.description,
            topic=self.topic,
            context=self.context,
            planner_instructions=planner_output["planning_instructions"],
            developer_implementation=developer_output["implementation"],
            pipeline_config=config_text
        )

        messages = [
            SystemMessage(content="You are a quality auditor reviewing data science implementations."),
            HumanMessage(content=prompt)
        ]

        response = self.llm(messages)
        
        return {
            "agent": self.auditor.name,
            "role": "Auditor",
            "audit_feedback": response.content.strip(),
            "reviewed_planner": planner_output["agent"],
            "reviewed_developer": developer_output["agent"]
        }

    def step4_developer_refinement(self, planner_output: Dict[str, Any], 
                                 developer_output: Dict[str, Any], 
                                 auditor_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 4: Developer refines implementation based on auditor feedback
        """
        print(f"\n🔧 Step 4: {self.developer.name} (Developer) refining based on feedback...")
        
        config_text = format_pipeline_config_for_prompt(PIPELINE_CONFIG)
        
        try:
            template = ITERATIVE_PROMPT_TEMPLATES["developer_refinement"][self.topic]
        except KeyError:
            template = ITERATIVE_PROMPT_TEMPLATES["developer_refinement"]["default"]

        prompt = template.format(
            developer_name=self.developer.name,
            developer_description=self.developer.description,
            topic=self.topic,
            context=self.context,
            planner_instructions=planner_output["planning_instructions"],
            initial_implementation=developer_output["implementation"],
            auditor_feedback=auditor_output["audit_feedback"],
            pipeline_config=config_text
        )

        messages = [
            SystemMessage(content="You are a Python developer refining code based on audit feedback."),
            HumanMessage(content=prompt)
        ]

        response = self.llm_coder(messages)
        
        return {
            "agent": self.developer.name,
            "role": "Developer (Refined)",
            "final_implementation": response.content.strip(),
            "incorporated_feedback_from": auditor_output["agent"],
            "original_planner": planner_output["agent"]
        }

    # def debug_developer_code(self, error_message: str, code: str, full_context: Dict[str, Any]) -> str:
    #     """
    #     Debug developer code when execution fails
    #     """
    #     print(f"\n🐛 Debug: {self.developer.name} fixing code errors...")
        
    #     config_text = format_pipeline_config_for_prompt(PIPELINE_CONFIG)
        
    #     try:
    #         template = ITERATIVE_PROMPT_TEMPLATES["debug_code"][self.topic]
    #     except KeyError:
    #         template = ITERATIVE_PROMPT_TEMPLATES["debug_code"]["default"]

    #     prompt = template.format(
    #         developer_name=self.developer.name,
    #         error_message=error_message,
    #         failed_code=code,
    #         planner_context=full_context.get("planner_output", {}).get("planning_instructions", ""),
    #         auditor_context=full_context.get("auditor_feedback", {}).get("audit_feedback", ""),
    #         pipeline_config=config_text
    #     )

    #     messages = [
    #         SystemMessage(content="You are a developer fixing code errors after feedback."),
    #         HumanMessage(content=prompt)
    #     ]

    #     response = self.llm_coder(messages)
    #     return response.content.strip()
    # Enhanced debug_developer_code for your iterative system
# Add this to core/dialogue.py in IterativeDialogueSimulator class

# Update core/dialogue.py debug method to match sequential system

    def debug_developer_code(self, error_message: str, code: str, full_context: Dict[str, Any], 
                            error_log: List[Dict] = None, code_history: str = "", 
                            prior_transforms: str = "", column_catalog: str = "") -> str:
        """
        Enhanced debug developer code with full context like sequential system
        """
        print(f"\n🐛 Debug: {self.developer.name} fixing code errors...")
        
        config_text = format_pipeline_config_for_prompt(PIPELINE_CONFIG)
        
        # Build error history summary like multi-agent system
        log_entries = []
        if error_log:
            relevant_errors = [
                entry for entry in error_log 
                if entry.get("task") == self.topic or entry.get("subtask") == self.topic
            ]
            
            log_entries = [
                f"""### Attempt {entry['retry']}
            **Traceback:**
            {entry['traceback']}
            **Code:**
            ```python
            {entry['code']}
            ```"""
                for entry in relevant_errors
            ]
        
        error_summary = "\n\n".join(log_entries) if log_entries else "No previous debugging attempts."
        
        try:
            template = ITERATIVE_PROMPT_TEMPLATES["debug_code"][self.topic]
        except KeyError:
            template = ITERATIVE_PROMPT_TEMPLATES["debug_code"]["default"]

        # Enhanced prompt with full context (like sequential system)
        prompt = template.format(
            developer_name=self.developer.name,
            error_message=error_message,
            failed_code=code,
            planner_context=full_context.get("planner_output", {}).get("planning_instructions", ""),
            auditor_context=full_context.get("auditor_feedback", {}).get("audit_feedback", ""),
            current_topic=self.topic,
            error_history=error_summary,
            pipeline_config=config_text,
            # 🆕 ADD THESE CONTEXT VARIABLES (like sequential system)
            code_history=code_history,
            prior_transforms=prior_transforms,
            column_catalog=column_catalog
        )

        messages = [
            SystemMessage(content="You are a developer fixing code errors with full context."),
            HumanMessage(content=prompt)
        ]

        response = self.llm_coder(messages)
        return response.content.strip()