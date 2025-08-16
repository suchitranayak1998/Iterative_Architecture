# orchestrators/orchestrator.py
"""
Iterative Orchestrator - coordinates the 3-agent workflow: Planner → Developer → Auditor → Developer
"""

from langchain_core.messages import SystemMessage, HumanMessage
from core.execution import DeveloperExecutor
from core.planner_agent import PlannerAgent
from core.auditor_module import Auditor, AuditDecisionOutput
from core.developer_module import DeveloperInteractionModule
from core.pipeline_state import PipelineState
import re
from util.prompt_library import PROMPT_TEMPLATES
from util.pipeline_config import PIPELINE_CONSTANTS as PIPELINE_CONFIG
from util.pipeline_util import format_pipeline_config_for_prompt

from typing import List, Tuple, Dict, Any
from reporting.validator import UnitTester, unit_test_report
from datetime import datetime

class IterativeOrchestrator:
    """
    Orchestrates the iterative 3-agent workflow for data science tasks
    """
    
    def __init__(self, df, topic: str, llm, llm_coder, summary: str = None, pipeline_state: PipelineState = None):
        """
        Initialize the iterative orchestrator
        
        :param df: The DataFrame to work with
        :param topic: Main task (EDA, Feature Engineering, Model Selection & Evaluation)
        :param agents: List of 3 agent objects [Planner, Developer, Auditor]
        :param llm: LangChain-compatible ChatOpenAI model
        :param llm_coder: LangChain-compatible ChatOpenAI model for coding
        :param summary: Summary from previous phases
        :param pipeline_state: Pipeline state tracking object
        """
        self.df = df
        self.topic = topic
        self.llm = llm
        self.llm_coder = llm_coder
        self.summary = summary
        self.context = self.build_context()
        self.planner = PlannerAgent(llm_coder)
        self.auditor = Auditor(llm_coder)
        self.executor = DeveloperExecutor(df)
        self.developer_module = DeveloperInteractionModule(
            llm_coder=self.llm_coder,
            topic=self.topic,
            context=self.context
        )
        self.pipeline_state = pipeline_state or PipelineState()
        self.subtasks = []
        self.error_log = []
        self._setup_data_integrity_validator()

    def _setup_data_integrity_validator(self):
        """Setup the data integrity validator with baseline expectations."""
        expected_shape = self.df.shape
        expected_columns = list(self.df.columns)
        target_column = "Class"  # TODO: Make this configurable
        
        self.data_validator = UnitTester(
            expected_shape=expected_shape,
            expected_columns=expected_columns,
            target_column=target_column
        )
        
        print(f"🔒 Data Integrity Validator initialized:")
        print(f"   Expected shape: {expected_shape}")
        print(f"   Essential columns: {len(expected_columns)}")
        print(f"   Target column: {target_column}")

    def extract_df_transformations(self, code: str) -> list:
        """Extract lines that modify the DataFrame 'df' using regex."""
        lines = code.split('\n')
        pattern = re.compile(r"^\s*(df(?:\[\s*.*?\s*\]|(?:\.\w+))).*")
        return [
            line.strip()
            for line in lines
            if pattern.match(line) and not line.strip().startswith("#")
        ]

    def build_context(self):
        """Build context from DataFrame for agent understanding"""
        summary_stats = self.df.describe(include='all').to_string()
        schema_info = self.df.dtypes.to_string()
        return f"Schema:\n{schema_info}\n\nSummary:\n{summary_stats}"

    # def decompose_task(self):
    #     """
    #     Use Planner agent to decompose the main task into subtasks
    #     """
    #     # Get prior context
    #     previous_subtasks = self.pipeline_state.get_all_subtasks()
    #     previous_transforms = self.pipeline_state.get_recent_transforms()

    #     # Format context
    #     subtask_text = "\n".join(f"- {s}" for s in previous_subtasks) or "None"
    #     transform_text = "\n".join(f"- {t}" for t in previous_transforms) or "None"
    #     summary_text = self.summary if self.summary else "No prior tasks."

    #     try:
    #         prompt_template = PROMPT_TEMPLATES["decompose"][self.topic]
    #     except KeyError:
    #         raise ValueError(f"No decompose prompt defined for task: {self.topic}")

    #     prompt = prompt_template.format(
    #         subtask_history=subtask_text,
    #         transform_history=transform_text,
    #         summary=summary_text,
    #         context=self.context,
    #         pipeline_config=format_pipeline_config_for_prompt(PIPELINE_CONFIG)
    #     )

    #     messages = [
    #         SystemMessage(content="You are a strategic planner decomposing data science tasks."),
    #         HumanMessage(content=prompt)
    #     ]
        
    #     response = self.llm(messages).content
    #     self.subtasks = [
    #         line.strip("0123456789. ").strip() 
    #         for line in response.split("\n") 
    #         if line.strip()
    #     ]
        
    #     return self.subtasks
    
    def ask_for_clarification(self,subtask, error_log):
        """
        Ask an expert agent (e.g., Manager) for clarification on the subtask that failed.
        Can be extended to ask a specific agent role if needed.
        """
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
        
        prompt = f"""
        You are the **Manager** in a collaborative AI team working on a multi-step machine learning pipeline. 

        The **Developer** has attempted to complete the following subtask multiple times but failed due to repeated execution errors.

        ---

        ## 🧩 Task Topic:
        {self.topic}

        ## 🔨 Subtask:
        "{subtask}"

        ## 🔁 Retry Attempts and Errors:
        {error_summary}

        ---

        Your job is to help the Developer understand the problem and get unblocked — by analyzing the traceback, diagnosing the possible issues, and suggesting the high-level logic for resolving it.

        ### 🎯 Your Objectives:
        1. **Interpret the subtask clearly.**  
        - What is this subtask asking for?
        - What kind of logic or outcome is expected?

        2. **Analyze the errors and failed attempts.**  
        - What does the traceback suggest?
        - What lines or logic are most likely causing the issue?
        - What data or columns are likely missing or malformed?

        3. **Suggest corrective logic (NO CODE).**  
        - What Could be the root cause of the failure?
        - What assumptions might be incorrect?
        - What steps could solve the issue?
        - If the problem is due to a missing column, bad assumption, or plot function misuse — state it clearly.

        ---

        ### 📌 Response Format:

        - **Subtask Interpretation:**  
        Explain in sentences what this subtask likely intends to achieve.

        - **Diagnosis:**  
        A list of bullet points highlighting:
            - What failed
            - Where it failed
            - Why it might have failed

        - **Suggested Fixes (no code):**  
        Provide steps outlining how the Developer should change their logic or assumptions to succeed.

        Do not generate any code. Focus only on understanding and reasoning.

        """
        messages = [
            SystemMessage(content="You are a helpful manager clarifying data science subtasks."),
            HumanMessage(content=prompt)
            ]
        response = self.llm_coder(messages)

        return response.content.strip()
    
    def dev_call(self, subtask, instruction, code_history, prior_transforms, column_catalog, df_copy = None):

        max_retries = 5
        retry_count = 0 
        clarification = ""
        plot_images = []
        attempt_log = []
        
        developer_reply = self.developer_module.generate_code(
            subtask=subtask,
            manager_instruction=instruction,
            code_history=code_history,
            prior_transforms=prior_transforms,
            column_catalog=column_catalog
        )

        code = self.executor.extract_code(developer_reply)
        if df_copy is None:
            success, execution_result, plot_images = self.executor.run_code(code)
        else:
            success, execution_result, plot_images = self.executor.run_code(code, df_copy)

        while not success and retry_count < max_retries:
                print(f"⚠️ Developer code failed (attempt {retry_count + 1}). Asking Developer to debug...\n")
                clarification = self.ask_for_clarification(
                        subtask=subtask,
                        error_log=attempt_log
                    )
                clarification = f"# 🔍 Clarification:\n# {clarification}\n\n" 

                attempt_log.append({
                    "task": self.topic,
                    "subtask": subtask,
                    "retry": retry_count,
                    "traceback": execution_result,
                    "code": code,
                    "clarification": clarification if clarification else ""
                })
                

                developer_reply = self.developer_module.debug_code(
                    subtask=subtask,
                    code=code,
                    error_message=execution_result,
                    error_log=attempt_log,
                    manager_instruction=clarification,
                    code_history= code_history,
                    prior_transforms=prior_transforms,
                    column_catalog=column_catalog
                )
                
                code = self.executor.extract_code(developer_reply)
                success, execution_result, plot_images = self.executor.run_code(code)

                

                if success:
                    print("✅ Fixed Developer code executed.\n")
                else:
                    retry_count += 1
        return success, execution_result, plot_images, attempt_log, code


    def run(self):
        """
        Execute the complete iterative workflow with sequential-style subtask execution
        """
        print("\n🧭 Planning phase (single planner)...")
        plan = self.planner.generate_plan(
            task=self.topic,
            context=self.context,
            summary=self.summary
        )

        print(f"📋 Planner produced {len(plan.subtasks)} subtasks.")

        pairs: List[Tuple[str, str]] = [
            (s.strip(), p.strip())
            for s, p in zip(plan.subtasks or [], plan.implementation_plan or [])
            if isinstance(p, str) and p.strip()
        ]
        if not pairs:
            print("⛔ Planner produced no valid (subtask, implementation) pairs. Aborting.")
            return {"plan": plan, "results": []}

        results: List[Dict[str, Any]] = []
        
        self.pipeline_state.update_phase(self.topic, self.subtasks)
        results = []

        print("\n💻 Developer executing subtasks...")
        for idx, (subtask, instruction) in enumerate(pairs, start=1):
            print(f"{idx}. {subtask}")
            
            auditor_accept = 0
            attempt_log = []
            # Get context that applies to all subtasks (like sequential)
            code_history = "\n\n---\n\n".join(self.pipeline_state.get_recent_code_history(n=5))
            code_history_list = self.pipeline_state.get_recent_code_history(n=5)
            
            if not code_history_list:
                print("📝 No Code History Found")
            else:
                print(f"📝 Number of code snippets in history: {len(code_history_list)}")

            prior_transforms = "\n".join(self.pipeline_state.get_recent_transforms())
            column_catalog = self.executor.get_column_catalog()

            print("plan:", instruction)

            df_copy = self.executor.df.copy()

            success, execution_result, plot_images, attempt_log, code = self.dev_call(subtask, instruction, code_history, prior_transforms, column_catalog, df_copy)

            decision: AuditDecisionOutput = self.auditor.review(
                        subtask=subtask,
                        plan_text=instruction,
                        execution_result=execution_result,
                        task_phase=self.topic
                    )
            if decision.accept:
                    print("🟢 Auditor: ACCEPT")
                    auditor_accept = 1

            if auditor_accept < 1:
                print("🟠 Auditor: REVISE → re-running with improved plan.")
                print("Improved Plan:", decision.improved_plan)
                instruction = decision.improved_plan
                success, execution_result, plot_images, attempt_log, code = self.dev_call(subtask, instruction, code_history, prior_transforms, column_catalog, df_copy)

            if success:
                print(f"✅ Developer code executed successfully")
                print("📄 Extracted Code:\n", code)
                print("📊 Execution Result:\n", execution_result)
                transform_lines = self.extract_df_transformations(code)
                for line in transform_lines:
                    self.pipeline_state.add_transform(line)
                
                print("🔒 Validating data integrity...")
                validation_results = self.data_validator.validate_dataframe_integrity(
                    self.executor.df, 
                    subtask
                )
                
                # Store validation results in pipeline state bucket
                subtask_tests = []
                for i, result in enumerate(validation_results, 1):
                    subtask_tests.append({
                        "test": f"{i}",
                        "status": "PASS" if result.passed else "FAIL", 
                        "reason": result.message
                    })
                
                # Use pipeline state as simple bucket
                self.pipeline_state.add_validation_result(
                    subtask=subtask,
                    subtask_index=idx,
                    validation_tests=subtask_tests
                )
                
                # Show console feedback
                passed = sum(1 for r in validation_results if r.passed)
                total = len(validation_results)
                
                if passed == total:
                    print("✅ Code executed successfully and passed all data integrity tests")
                    print(f"📊 Validation: {passed}/{total} tests passed")
            
            else:
                print("📄 Extracted Code:\n", code)
                print("❌ Developer failed to fix the code after 5 attempts.")
                print("📄 Final Execution Result:\n", execution_result)

            self.error_log.extend(attempt_log)

            results.append({
            "subtask": subtask,
            "Implementation_Plan": instruction,
            "code": code,
            "execution_result": execution_result,
            "images": plot_images
        })
            self.pipeline_state.add_subtask_result(
                phase=self.topic,
                subtask=subtask,
                Implementation_Plan=instruction,
                code=code,
                result=execution_result,
                images=plot_images
            )


        return results