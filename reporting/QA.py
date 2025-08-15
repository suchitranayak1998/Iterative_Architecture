# reporting/QA.py
"""
Quality Assurance system for iterative multi-agent workflow
"""

import pandas as pd
from core.pipeline_state import PipelineState
from typing import List, Any, Optional, Dict
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage


class QualityAssurance:
    def __init__(self, pipeline_state, llm):
        self.state = pipeline_state
        self.llm = llm
        self.results = []

    def validate_results(self, results: List[dict], task: str):
        """Validate results from iterative process"""
        for entry in results:
            # Handle iterative process results
            if "iterative_process" in entry:
                validation = self._validate_iterative_process(entry, task)
            else:
                # Fallback for standard results
                validation = self._validate_standard_result(entry, task)
            
            validation["phase"] = task
            self.results.append(validation)

    def _validate_iterative_process(self, entry: dict, task: str):
        """Validate a complete iterative process result"""
        iterative_data = entry.get("iterative_process", {})
        subtask = entry.get("subtask", "Unknown")
        
        # Extract components
        planner_output = iterative_data.get("planner_output", {})
        initial_dev_output = iterative_data.get("initial_developer_output", {})
        auditor_feedback = iterative_data.get("auditor_feedback", {})
        final_dev_output = iterative_data.get("final_developer_output", {})
        
        execution_result = entry.get("execution_result", "")
        final_code = entry.get("final_developer_code", "")
        success = entry.get("success", False)

        prompt = f"""
        You are a Quality Assurance (QA) agent reviewing a complete 4-step iterative process in a machine learning team.

        Task: {task}
        Subtask: {subtask}

        ## Iterative Process Review:

        **Step 1 - Planner Output:**
        Agent: {planner_output.get('agent', 'Unknown')}
        Instructions: {planner_output.get('planning_instructions', 'No instructions')[:500]}...

        **Step 2 - Initial Developer Implementation:**
        Agent: {initial_dev_output.get('agent', 'Unknown')}
        Implementation: {initial_dev_output.get('implementation', 'No implementation')[:300]}...

        **Step 3 - Auditor Feedback:**
        Agent: {auditor_feedback.get('agent', 'Unknown')}
        Feedback: {auditor_feedback.get('audit_feedback', 'No feedback')[:300]}...

        **Step 4 - Final Developer Implementation:**
        Agent: {final_dev_output.get('agent', 'Unknown')}
        Final Implementation: {final_dev_output.get('final_implementation', 'No final implementation')[:300]}...

        **Execution Results:**
        Success: {success}
        Code: {final_code[:200]}...
        Output: {execution_result[:300]}...

        ## QA Evaluation Criteria:

        1. **Process Completeness**: Were all 4 steps executed properly?
        2. **Agent Coordination**: Did agents build upon each other's work effectively?
        3. **Quality Improvement**: Did the auditor feedback lead to meaningful improvements?
        4. **Final Implementation**: Does the final code address the original task?
        5. **Execution Success**: Did the final code execute successfully?

        ## Your Assessment:
        
        Evaluate the overall quality of this iterative process and provide:
        
        Status: PASS or FAIL
        Reason: <detailed explanation covering process quality, agent coordination, and final outcomes>
        
        Focus on whether the iterative process improved the solution quality.
        """

        messages = [
            SystemMessage(content="You are a QA agent specializing in multi-agent iterative processes."),
            HumanMessage(content=prompt)
        ]

        response = self.llm(messages)
        
        return {
            "subtask": subtask,
            "process_type": "iterative_4_step",
            "status": "pass" if "pass" in response.content.lower() else "fail",
            "reason": response.content.strip(),
            "agents_involved": [
                planner_output.get('agent', 'Unknown'),
                initial_dev_output.get('agent', 'Unknown'), 
                auditor_feedback.get('agent', 'Unknown'),
                final_dev_output.get('agent', 'Unknown')
            ],
            "execution_success": success
        }

    def _validate_standard_result(self, entry: dict, task: str):
        """Validate standard (non-iterative) result format"""
        subtask = entry.get("subtask", "Unknown")
        code = entry.get("code", "")
        result = entry.get("execution_result", "")
        summary = entry.get("manager_instruction", "") or entry.get("planner_instructions", "")

        prompt = f"""
        You are a Quality Assurance (QA) agent reviewing a data science subtask implementation.

        Subtask: {subtask}
        Task Context: {task}
        
        Implementation Summary: {summary[:300]}...
        Code: {code[:300]}...
        Execution Output: {result[:300]}...

        Evaluate if:
        1. The subtask objective is addressed
        2. The code performs relevant logic
        3. The result aligns with expected outcome
        4. No obvious issues or shortcuts exist

        Status: PASS or FAIL  
        Reason: <short explanation>
        """

        messages = [
            SystemMessage(content="You are a QA agent verifying data science subtask implementations."),
            HumanMessage(content=prompt)
        ]

        response = self.llm(messages)
        
        return {
            "subtask": subtask,
            "process_type": "standard",
            "status": "pass" if "pass" in response.content.lower() else "fail",
            "reason": response.content.strip()
        }

    def export_report(self, phase_name: str, fmt="csv", save_dir: Path = Path("./reports")):
        """Export QA report in specified format"""
        df = pd.DataFrame(self.results)
        filename = f"qa_summary_{phase_name.lower().replace(' ', '_')}.{fmt}"
        output_path = save_dir / filename
        save_dir.mkdir(parents=True, exist_ok=True)

        if fmt == "csv":
            df.to_csv(output_path, index=False)
        elif fmt == "json":
            df.to_json(output_path, indent=2, orient="records")
        elif fmt == "md":
            with open(output_path, "w") as f:
                f.write(df.to_markdown(index=False))
        elif fmt == "html":
            # Enhanced HTML for iterative processes
            html = f"""<html>
            <head>
                <title>QA Summary - {phase_name} (Iterative Process)</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; }}
                    .pass {{ color: #27ae60; font-weight: bold; }}
                    .fail {{ color: #e74c3c; font-weight: bold; }}
                    .iterative {{ background-color: #f8f9fa; padding: 10px; margin: 5px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f6; }}
                </style>
            </head>
            <body>
            <h1>Quality Assurance Summary - {phase_name}</h1>
            <p><strong>Architecture:</strong> 3-Agent Iterative Process</p>
            <p><strong>Process Steps:</strong> Planner → Developer → Auditor → Developer</p>
            """
            
            # Add summary statistics
            total = len(df)
            passed = len(df[df['status'] == 'pass'])
            iterative = len(df[df['process_type'] == 'iterative_4_step'])
            
            html += f"""
            <h2>Summary Statistics</h2>
            <ul>
                <li>Total Evaluations: {total}</li>
                <li>Passed: <span class="pass">{passed}</span></li>
                <li>Failed: <span class="fail">{total - passed}</span></li>
                <li>Iterative Processes: {iterative}</li>
                <li>Success Rate: {(passed/total*100):.1f}%</li>
            </ul>
            """
            
            html += "<h2>Detailed Results</h2>"
            html += df.to_html(index=False, escape=False, classes="qa-table")
            html += "</body></html>"
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

        print(f"📋 QA report saved: {output_path}")
        return output_path

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of QA results"""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        return {
            "total_evaluations": len(df),
            "passed": len(df[df['status'] == 'pass']),
            "failed": len(df[df['status'] == 'fail']),
            "success_rate": len(df[df['status'] == 'pass']) / len(df) * 100,
            "iterative_processes": len(df[df.get('process_type', '') == 'iterative_4_step']),
            "standard_processes": len(df[df.get('process_type', '') == 'standard'])
        }