# core/developer_module.py

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from util import pipeline_config
from util.prompt_library import PROMPT_TEMPLATES
from util.pipeline_config import PIPELINE_CONSTANTS as PIPELINE_CONFIG
from util.pipeline_util import format_pipeline_config_for_prompt


class DeveloperInteractionModule:
    """
    Used to prompt the Developer agent to generate or debug code
    in a swarm-style system, based on proposal instructions.
    """

    def __init__(self, llm_coder: ChatOpenAI, context: str, topic: str):
        self.llm_coder = llm_coder
        self.context = context
        self.topic = topic

    def generate_code(self, subtask: str, manager_instruction: str, code_history: str = "", prior_transforms: str = "", column_catalog: str = "") -> str:
        """
        Ask the Developer agent to generate Python code for a subtask.
        """
        config_text = format_pipeline_config_for_prompt(PIPELINE_CONFIG)

        if "Model Selection & Evaluation" in self.topic:
            template = PROMPT_TEMPLATES["generate_code"]["Model Selection & Evaluation"]
        else:
            template = PROMPT_TEMPLATES["generate_code"]["default"]

        prompt = template.format(
            context=self.context,
            subtask=subtask,
            manager_instruction=manager_instruction,
            code_history=code_history,
            prior_transforms=prior_transforms,
            pipeline_config=config_text,
            column_catalog=column_catalog,
        )

        messages = [
            SystemMessage(content="You are a Python-savvy developer generating analysis code."),
            HumanMessage(content=prompt),
        ]

        response = self.llm_coder(messages)
        return response.content.strip()

    def debug_code(self, subtask: str, code: str, error_message: str, error_log: list, manager_instruction: str, code_history: str = "", prior_transforms: str = "", column_catalog: str = "") -> str:
        """
        Ask the Developer agent to debug code based on an error message.
        """
        config_text = format_pipeline_config_for_prompt(PIPELINE_CONFIG)

        # Format error log into markdown-like summaries
        log_entries = [
            f"""### Attempt {entry['retry']}
**Traceback:**
{entry['traceback']}

**Code:**
```python
{entry['code']}
```"""
            for entry in error_log
            if entry["subtask"] == subtask
        ]
        error_summary = "\n\n".join(log_entries)

        if "Model Selection & Evaluation" in self.topic:
            template = PROMPT_TEMPLATES["debug_code"]["Model Selection & Evaluation"]
        else:
            template = PROMPT_TEMPLATES["debug_code"]["default"]

        prompt = template.format(
            error_message=error_message,
            manager_instruction=manager_instruction,
            code_history=code_history,
            prior_transforms=prior_transforms,
            original_code=code,
            error_code_past=error_summary,
            config_text=config_text,
            column_catalog=column_catalog,
            pipeline_config=config_text
        )

        messages = [
            SystemMessage(content="You are a developer fixing code errors after feedback."),
            HumanMessage(content=prompt),
        ]

        response = self.llm_coder(messages)
        return response.content.strip()
