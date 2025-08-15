# reporting/task_boards.py
"""
Task checklist validation for iterative multi-agent system
"""

from pathlib import Path
from typing import List, Dict
import json
from langchain_core.messages import SystemMessage, HumanMessage
from util.checklists import CHECKLISTS

class TaskChecklist:
    def __init__(self, llm, task: str):
        self.llm = llm
        self.task = task
        self.expected_items = CHECKLISTS.get(task, [])

    def validate_phase(self, phase_name: str, results: List[Dict]) -> Dict:
        """Validate phase completion against expected checklist items"""
        
        # Extract information from iterative results
        summaries = []
        for r in results:
            if "iterative_process" in r:
                # Handle iterative process results
                iterative_data = r["iterative_process"]
                
                # Include all steps of iterative process
                planner_info = iterative_data.get("planner_output", {}).get("planning_instructions", "")
                initial_dev = iterative_data.get("initial_developer_output", {}).get("implementation", "")
                auditor_feedback = iterative_data.get("auditor_feedback", {}).get("audit_feedback", "")
                final_dev = iterative_data.get("final_developer_output", {}).get("final_implementation", "")
                final_code = r.get("final_developer_code", "")
                
                summary_text = f"""
                ### Iterative Process for: {r['subtask']}
                **Planner Instructions:** {planner_info[:200]}...
                **Initial Implementation:** {initial_dev[:200]}...
                **Auditor Feedback:** {auditor_feedback[:200]}...
                **Final Implementation:** {final_dev[:200]}...
                **Final Code:** {final_code[:300]}...
                """
                summaries.append(summary_text)
            else:
                # Handle standard results
                summary_text = f"### Subtask: {r['subtask']}\n{r.get('code', '')[:300]}..."
                summaries.append(summary_text)

        combined_summaries = "\n\n".join(summaries)

        prompt = f"""
        You are a checklist auditor for a machine learning pipeline using a 3-agent iterative process.

        The iterative process follows these steps:
        1. Planner: Creates strategic plan and implementation instructions
        2. Developer: Implements initial code
        3. Auditor: Reviews and provides feedback
        4. Developer: Refines implementation based on feedback

        The expected checklist items for the phase **{phase_name}** are:
        {json.dumps(self.expected_items, indent=2)}

        Here are the completed iterative processes and their outputs:
        {combined_summaries}

        For each expected checklist item, evaluate whether it was:
        - Fully Addressed: Completed thoroughly through the iterative process
        - Partially Addressed: Attempted but could be improved
        - Not Addressed: Missing or inadequate

        Consider the complete iterative cycle when evaluating - improvements made through auditor feedback should be reflected in your assessment.

        Respond in this exact JSON format:
        {{
            "Handle missing data": "Fully Addressed",
            "Analyze outliers": "Partially Addressed",
            "Create visualizations": "Fully Addressed"
        }}
        """
        
        messages = [
            SystemMessage(content="You are a QA agent evaluating iterative multi-agent process completeness."),
            HumanMessage(content=prompt)
        ]
        
        try:
            llm_response = self.llm(messages)
            # Try to parse JSON response
            response_content = llm_response.content.strip()
            
            # Clean up response if it contains markdown code blocks
            if "```json" in response_content:
                response_content = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                response_content = response_content.split("```")[1].split("```")[0].strip()
                
            checklist_result = json.loads(response_content)
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Error parsing LLM response: {e}")
            checklist_result = {"error": f"Could not parse LLM output: {str(e)}"}

        return checklist_result

    def export_report(self, report: Dict, save_path: Path, fmt="json"):
        """Export checklist validation report"""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
                
        elif fmt == "html":
            html = f"""
            <html>
            <head>
                <title>Checklist QA Report - {self.task}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; }}
                    .fully {{ color: #27ae60; font-weight: bold; }}
                    .partially {{ color: #f39c12; font-weight: bold; }}
                    .not {{ color: #e74c3c; font-weight: bold; }}
                    .iterative-info {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
                    ul {{ list-style-type: none; padding: 0; }}
                    li {{ padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>Checklist QA Report: {self.task}</h1>
                
                <div class="iterative-info">
                    <h3>🔄 Iterative Process Architecture</h3>
                    <p><strong>Process:</strong> 3-Agent Iterative (Planner → Developer → Auditor → Developer)</p>
                    <p><strong>Quality Assurance:</strong> Built-in review and refinement cycle</p>
                </div>
                
                <h2>Checklist Evaluation Results</h2>
                <ul>
            """
            
            # Count completion levels
            fully_count = sum(1 for status in report.values() if "fully" in status.lower())
            partially_count = sum(1 for status in report.values() if "partially" in status.lower())
            not_count = sum(1 for status in report.values() if "not" in status.lower())
            
            for item, status in report.items():
                if "error" in item.lower():
                    html += f'<li class="not"><strong>Error:</strong> {status}</li>'
                else:
                    css_class = "fully" if "fully" in status.lower() else "partially" if "partially" in status.lower() else "not"
                    html += f'<li class="{css_class}"><strong>{item}:</strong> {status}</li>'
            
            html += f"""
                </ul>
                
                <h2>Summary Statistics</h2>
                <ul>
                    <li class="fully">Fully Addressed: {fully_count}</li>
                    <li class="partially">Partially Addressed: {partially_count}</li>
                    <li class="not">Not Addressed: {not_count}</li>
                    <li>Total Items: {len([k for k in report.keys() if "error" not in k.lower()])}</li>
                </ul>
            </body>
            </html>
            """
            
            html_path = save_path.with_suffix(".html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"✅ HTML checklist report saved to {html_path}")

        print(f"✅ Checklist QA report saved to {save_path}")
        
    def get_completion_stats(self, report: Dict) -> Dict:
        """Get completion statistics from checklist report"""
        if "error" in report:
            return {"error": "Could not generate stats due to parsing error"}
            
        total_items = len([k for k in report.keys() if "error" not in k.lower()])
        fully_addressed = sum(1 for status in report.values() if "fully" in status.lower())
        partially_addressed = sum(1 for status in report.values() if "partially" in status.lower())
        not_addressed = sum(1 for status in report.values() if "not" in status.lower())
        
        return {
            "total_items": total_items,
            "fully_addressed": fully_addressed,
            "partially_addressed": partially_addressed,
            "not_addressed": not_addressed,
            "completion_rate": (fully_addressed / total_items * 100) if total_items > 0 else 0,
            "partial_completion_rate": ((fully_addressed + partially_addressed) / total_items * 100) if total_items > 0 else 0
        }