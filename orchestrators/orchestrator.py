# orchestrators/orchestrator.py
"""
Iterative Orchestrator - coordinates the 3-agent workflow: Planner → Developer → Auditor → Developer
"""

from langchain_core.messages import SystemMessage, HumanMessage
from core.dialogue import IterativeDialogueSimulator
from core.execution import DeveloperExecutor
from core.agents import agent
from core.pipeline_state import PipelineState
import re
from util.prompt_library import ITERATIVE_PROMPT_TEMPLATES
from util.pipeline_config import PIPELINE_CONSTANTS as PIPELINE_CONFIG
from util.pipeline_util import format_pipeline_config_for_prompt
from reporting.QA import QualityAssurance
from typing import List, Tuple, Dict, Any
from reporting.validator import UnitTester, unit_test_report
from datetime import datetime

class IterativeOrchestrator:
    """
    Orchestrates the iterative 3-agent workflow for data science tasks
    """
    
    def __init__(self, df, topic: str, agents, llm, llm_coder, summary: str = None, pipeline_state: PipelineState = None):
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
        self.agents = agents
        self.llm = llm
        self.llm_coder = llm_coder
        self.summary = summary
        self.context = self.build_context()
        self.simulator = IterativeDialogueSimulator(agents, self.context, llm, llm_coder, topic, summary)
        self.executor = DeveloperExecutor(df)
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

    def decompose_task(self):
        """
        Use Planner agent to decompose the main task into subtasks
        """
        # Get prior context
        previous_subtasks = self.pipeline_state.get_all_subtasks()
        previous_transforms = self.pipeline_state.get_recent_transforms()

        # Format context
        subtask_text = "\n".join(f"- {s}" for s in previous_subtasks) or "None"
        transform_text = "\n".join(f"- {t}" for t in previous_transforms) or "None"
        summary_text = self.summary if self.summary else "No prior tasks."

        try:
            prompt_template = ITERATIVE_PROMPT_TEMPLATES["decompose"][self.topic]
        except KeyError:
            raise ValueError(f"No decompose prompt defined for task: {self.topic}")

        prompt = prompt_template.format(
            subtask_history=subtask_text,
            transform_history=transform_text,
            summary=summary_text,
            context=self.context,
            pipeline_config=format_pipeline_config_for_prompt(PIPELINE_CONFIG)
        )

        messages = [
            SystemMessage(content="You are a strategic planner decomposing data science tasks."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm(messages).content
        self.subtasks = [
            line.strip("0123456789. ").strip() 
            for line in response.split("\n") 
            if line.strip()
        ]
        
        return self.subtasks

    # def extract_code_from_response(self, response: str) -> str:
    #     """Extract Python code from LLM response"""
    #     return self.executor.extract_code(response)

    # def run(self):
    #     """
    #     Execute the complete iterative workflow
    #     """
    #     print(f"\n🎯 Iterative Task: {self.topic}")
    #     print("🧩 Decomposing into subtasks...")
        
    #     # Decompose task into subtasks
    #     self.decompose_task()
    #     self.pipeline_state.update_phase(self.topic, self.subtasks)
        
    #     print("📝 Subtasks Planned:\n")
    #     for idx, task in enumerate(self.subtasks, 1):
    #         print(f"{idx}. {task}")
        
    #     code_history = "\n\n---\n\n".join(self.pipeline_state.get_recent_code_history(n=5))
    #     code_history_list = self.pipeline_state.get_recent_code_history(n=5)
    
    #     if not code_history_list:
    #         print("📝 No Code History Found")
    #     else:
    #         print(f"📝 Number of code snippets in history: {len(code_history_list)}")

    #     # Add prior transforms context
    #     prior_transforms = "\n".join(self.pipeline_state.get_recent_transforms())
        
    #     # Add column catalog context  
    #     column_catalog = self.executor.get_column_catalog()

    #     # Run the 4-step iterative process
    #     print(f"\n🔄 Starting 4-step iterative process...")
    #     iterative_results = self.simulator.run_iterative_process(self.subtasks)
    #     print(iterative_results)

    #     # Extract and execute final code
    #     final_code = self.executor.extract_code(
    #         iterative_results["final_developer_output"]["final_implementation"]
    #     )
        
    #     print(f"\n💻 Executing final refined code...")
    #     success, execution_result, plot_images = self.executor.run_code(final_code)

        
        
    #     retry_count = 0
    #     # Debug loop if needed
    #     if not success:
            
    #         max_retries = 5
    #         while not success and retry_count < max_retries:
    #             print(f"⚠️ Code execution failed (attempt {retry_count + 1}). Debugging...")
                
    #             # ADD ERROR TO LOG
    #             self.error_log.append({
    #                 "task": self.topic,
    #                 "subtask": self.topic,
    #                 "retry": retry_count + 1,
    #                 "traceback": execution_result,
    #                 "code": final_code,
    #                 "timestamp": datetime.now().isoformat()
    #             })

    #             # Use debug method from dialogue simulator
    #             debug_response = self.simulator.debug_developer_code(
    #                 error_message=execution_result,
    #                 code=final_code,
    #                 full_context=iterative_results,
    #                 error_log=self.error_log,
    #                 code_history=code_history,        # 🆕 ADD THIS
    #                 prior_transforms=prior_transforms, # 🆕 ADD THIS
    #                 column_catalog=column_catalog  
    #             )
                
    #             final_code = self.executor.extract_code(debug_response)
    #             success, execution_result, plot_images = self.executor.run_code(final_code)
                
    #             if success:
    #                 print("✅ Debugging successful!")
    #                 # Update the final output with debugged code
    #                 iterative_results["final_developer_output"]["final_implementation"] = debug_response
    #             else:
    #                 retry_count += 1
        
    #     if success:
    #         print("✅ Code executed successfully!")  # ADD THIS LINE
    #         print("\n" + "="*70)
    #         print("🖥️  EXECUTION OUTPUT:")
    #         print("="*70)

    #         if execution_result and execution_result.strip():
    #         # Print the complete output to console
    #             print(execution_result)
    #         else:
    #             print("No console output generated")
        
    #         print("="*70)
        
    #         # Show execution summary
    #         if plot_images:
    #             print(f"📈 Generated {len(plot_images)} visualizations")
                
    #         # Track DataFrame transformations
    #         transform_lines = self.extract_df_transformations(final_code)
    #         for line in transform_lines:
    #             self.pipeline_state.add_transform(line)
            
    #         # Validate data integrity
    #         print("🔒 Validating data integrity...")
    #         validation_results = self.data_validator.validate_dataframe_integrity(
    #             self.executor.df, 
    #             self.topic
    #         )
            
    #         # Store validation results
    #         subtask_tests = []
    #         for i, result in enumerate(validation_results, 1):
    #             subtask_tests.append({
    #                 "test": f"Data Integrity Test {i}",
    #                 "status": "PASS" if result.passed else "FAIL", 
    #                 "reason": result.message
    #             })
            
    #         self.pipeline_state.add_validation_result(
    #             subtask=self.topic,
    #             subtask_index=1,
    #             validation_tests=subtask_tests
    #         )
            
    #         # Show validation feedback
    #         passed = sum(1 for r in validation_results if r.passed)
    #         total = len(validation_results)
    #         print(f"📊 Validation: {passed}/{total} tests passed")
            
    #     else:
    #         print("❌ Final code execution failed after maximum retries.")
    #         print("📄 Final Execution Result:\n", execution_result)

       
    #     results = [{
    #         "subtask": self.topic,
    #         "iterative_process": iterative_results,
    #         "planner_instructions": iterative_results["planner_output"]["planning_instructions"],
    #         "initial_developer_code": self.executor.extract_code(
    #             iterative_results["initial_developer_output"]["implementation"]
    #         ),
    #         "auditor_feedback": iterative_results["auditor_feedback"]["audit_feedback"],
    #         "final_developer_code": final_code,
    #         "execution_result": execution_result,
    #         "images": plot_images,
    #         "success": success,
    #         "subtasks_planned": self.subtasks,      
    #         "total_subtasks": len(self.subtasks),   
    #         "phase_name": self.topic        
    #     }]
       
    #     # Add to pipeline state
    #     self.pipeline_state.add_subtask_result(
    #         phase=self.topic,
    #         subtask=self.topic,
    #         summary=iterative_results["planner_output"]["planning_instructions"],
    #         code=final_code,
    #         result=execution_result,
    #         images=plot_images
    #     )

    #     return results
    # Replace your run() method with this version that maintains sequential paradigm

    def run(self):
        """
        Execute the complete iterative workflow with sequential-style subtask execution
        """
        print(f"\n🎯 Iterative Task: {self.topic}")
        print("🧩 Decomposing into subtasks...")
        
        # Decompose task into subtasks
        self.decompose_task()
        self.pipeline_state.update_phase(self.topic, self.subtasks)
        results = []

        print("📝 Subtasks Planned:\n")
        for idx, task in enumerate(self.subtasks, 1):
            print(f"{idx}. {task}")

        # Get context that applies to all subtasks (like sequential)
        code_history = "\n\n---\n\n".join(self.pipeline_state.get_recent_code_history(n=5))
        code_history_list = self.pipeline_state.get_recent_code_history(n=5)
        
        if not code_history_list:
            print("📝 No Code History Found")
        else:
            print(f"📝 Number of code snippets in history: {len(code_history_list)}")

        prior_transforms = "\n".join(self.pipeline_state.get_recent_transforms())
        column_catalog = self.executor.get_column_catalog()

        # 🔄 4-Step Iterative Process (ONCE for entire phase)
        print(f"\n🔄 Starting 4-step iterative process...")
        iterative_results = self.simulator.run_iterative_process(self.subtasks)
        print(self.subtasks)
        # Extract final refined code from 4-step process
        final_code = self.executor.extract_code(
            iterative_results["final_developer_output"]["final_implementation"]
        )
        
        # 🆕 NOW LOOP THROUGH SUBTASKS (like sequential)
        for idx, subtask in enumerate(self.subtasks, 1):
            print(f"\n\n🔍 Subtask {idx}: {subtask}")
            
            # Use the refined code from 4-step process for this subtask
            code = final_code
            plot_images = []
            success, execution_result, plot_images = self.executor.run_code(code)

            # Debugging loop (same pattern as sequential)
            max_retries = 5
            retry_count = 0
            clarified = True
            self.error_log = []  # Keep same as sequential
            clarification = ""

            if not success:
                print(f"❌ Initial code execution failed for subtask {idx}, \n code: {code}\n")

            while not success and retry_count < max_retries:
                print(f"⚠️ Developer code failed (attempt {retry_count + 1}). Asking Developer to debug...\n")
                
                # # Use same clarification pattern as sequential
                # clarification = self.ask_for_clarification(
                #     subtask=subtask,
                #     error_log=self.error_log
                # )
                # clarification = f"# 🔍 Clarification:\n# {clarification}\n\n"

                self.error_log.append({
                    "task": self.topic,
                    "subtask": subtask,
                    "retry": retry_count,
                    "traceback": execution_result,
                    "code": code,
                    # "clarification": clarification if clarified else ""
                })

                code = self.executor.extract_code(code)  # Keep same pattern
                success, execution_result, plot_images = self.executor.run_code(code)

                # Use enhanced debug method (adapted for iterative context)
                developer_reply = self.simulator.debug_developer_code(
                    error_message=execution_result,
                    code=code,
                    full_context=iterative_results,  # 🆕 Pass iterative context
                    error_log=self.error_log,
                    code_history=code_history,
                    prior_transforms=prior_transforms,
                    column_catalog=column_catalog
                )
                
                # Extract debugged code
                code = self.executor.extract_code(developer_reply)
                success, execution_result, plot_images = self.executor.run_code(code)

                if success:
                    print("✅ Fixed Developer code executed.\n")
                    # Update final code for remaining subtasks
                    final_code = code
                else:
                    retry_count += 1
            
            if success:
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

            # Build results (similar to sequential but with iterative context)
            results.append({
                "subtask": subtask,
                "conversation": [  # 🆕 Adapt iterative results to sequential format
                    {"role": "Planner", "name": iterative_results["planner_output"]["agent"], 
                    "message": iterative_results["planner_output"]["planning_instructions"]},
                    {"role": "Developer (Initial)", "name": iterative_results["initial_developer_output"]["agent"],
                    "message": iterative_results["initial_developer_output"]["implementation"]},
                    {"role": "Auditor", "name": iterative_results["auditor_feedback"]["agent"],
                    "message": iterative_results["auditor_feedback"]["audit_feedback"]},
                    {"role": "Developer (Final)", "name": iterative_results["final_developer_output"]["agent"],
                    "message": iterative_results["final_developer_output"]["final_implementation"]}
                ],
                "manager_instruction": iterative_results["planner_output"]["planning_instructions"],  # Map to sequential
                "developer_reply": iterative_results["final_developer_output"]["final_implementation"],  # Final refined code
                "code": code,
                "execution_result": execution_result,
                "images": plot_images
            })
            
            print(subtask)
            print(code)
            print(execution_result)
            self.pipeline_state.add_subtask_result(
                phase=self.topic,
                subtask=subtask,
                summary=iterative_results["planner_output"]["planning_instructions"],  # Use planner instructions as summary
                code=code,
                result=execution_result,
                images=plot_images
            )

            # Update column catalog for next subtask (DataFrame might have changed)
            column_catalog = self.executor.get_column_catalog()

        return results