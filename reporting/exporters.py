# reporting/exporters.py
"""
Enhanced report exporter for iterative 3-agent system
"""

from pathlib import Path


class IterativeReportExporter:
    def __init__(self, task: str, results: list):
        self.task = task
        self.results = results or []

    def generate_markdown(self) -> str:
        report = f"# 🔄 Iterative Analysis Report: {self.task}\n\n"
        
        report += "## 🎯 Process Overview\n"
        report += "This report shows the complete 4-step iterative process:\n"
        report += "1. **Planner**: Strategic planning and task decomposition\n"
        report += "2. **Developer**: Initial implementation\n"
        report += "3. **Auditor**: Review and feedback\n"
        report += "4. **Developer**: Refined implementation\n\n"

        for idx, entry in enumerate(self.results, 1):
            iterative_process = entry.get("iterative_process", {})
            
            # report += f"## 🔧 Task {idx}: {entry.get('subtask', 'Unnamed')}\n\n"
            report += f"## 🔧 Phase: {entry.get('subtask', 'Unnamed')}\n\n"

            # Add subtasks list if available
            if "subtasks_planned" in entry:
                report += "### 📋 Planned Subtasks\n"
                for i, subtask in enumerate(entry["subtasks_planned"], 1):
                    report += f"{i}. {subtask}\n"
                report += "\n"

            # Step 1: Planner Output
            planner_output = iterative_process.get("planner_output", {})
            if planner_output:
                report += f"### 📋 Step 1: {planner_output.get('agent', 'Planner')} (Strategic Planning)\n"
                report += f"**Role:** {planner_output.get('role', 'Planner')}\n\n"
                report += f"{planner_output.get('planning_instructions', 'No instructions provided')}\n\n"

            # Step 2: Initial Developer Implementation
            initial_dev = iterative_process.get("initial_developer_output", {})
            if initial_dev:
                report += f"### 💻 Step 2: {initial_dev.get('agent', 'Developer')} (Initial Implementation)\n"
                report += f"**Role:** {initial_dev.get('role', 'Developer')}\n\n"
                
                # Extract code from initial implementation
                initial_code = entry.get("initial_developer_code", "")
                if initial_code:
                    report += "**Initial Code:**\n"
                    report += f"```python\n{initial_code}\n```\n\n"

            # Step 3: Auditor Review
            auditor_output = iterative_process.get("auditor_feedback", {})
            if auditor_output:
                report += f"### 🔍 Step 3: {auditor_output.get('agent', 'Auditor')} (Quality Review)\n"
                report += f"**Role:** {auditor_output.get('role', 'Auditor')}\n\n"
                report += f"{auditor_output.get('audit_feedback', 'No feedback provided')}\n\n"

            # Step 4: Final Developer Implementation
            final_dev = iterative_process.get("final_developer_output", {})
            if final_dev:
                report += f"### 🔧 Step 4: {final_dev.get('agent', 'Developer')} (Refined Implementation)\n"
                report += f"**Role:** {final_dev.get('role', 'Developer (Refined)')}\n\n"
                
                # Final refined code
                final_code = entry.get("final_developer_code", "")
                if final_code:
                    report += "**Final Refined Code:**\n"
                    report += f"```python\n{final_code}\n```\n\n"

            # Execution Results
            report += "### 🖥 Execution Results\n"
            result = entry.get("execution_result", "")
            success = entry.get("success", False)
            
            status_emoji = "✅" if success else "❌"
            report += f"**Status:** {status_emoji} {'Success' if success else 'Failed'}\n\n"
            
            if isinstance(result, tuple):
                result = result[0]
            report += f"```\n{result}\n```\n"

            # Images
            images = entry.get("images", [])
            if images:
                report += "### 📈 Generated Visualizations\n"
                for i, img in enumerate(images):
                    report += f"![Visualization {i+1}](data:image/png;base64,{img})\n\n"

            # Process Summary
            report += "### 📊 Process Summary\n"
            report += f"- **Planner Agent:** {planner_output.get('agent', 'N/A')}\n"
            report += f"- **Developer Agent:** {initial_dev.get('agent', 'N/A')}\n"
            report += f"- **Auditor Agent:** {auditor_output.get('agent', 'N/A')}\n"
            report += f"- **Final Status:** {'Success' if success else 'Failed'}\n"
            report += f"- **Iterations:** 4-step iterative process completed\n\n"

            report += "---\n\n"

        report += "## 📈 Overall Process Summary\n"

        # Get subtasks from results
        total_subtasks = 0
        successful_subtasks = 0

        for result in self.results:
            # Use the subtask count from orchestrator if available
            if "total_subtasks" in result:
                total_subtasks += result["total_subtasks"]
                # If the phase succeeded, count all its subtasks as successful
                if result.get("success", False):
                    successful_subtasks += result["total_subtasks"]
            else:
                # Fallback: count phases as tasks
                total_subtasks += 1
                successful_subtasks += 1 if result.get("success", False) else 0

        report += f"- **Total Subtasks:** {total_subtasks}\n"
        report += f"- **Successful Subtasks:** {successful_subtasks}\n"
        report += f"- **Success Rate:** {(successful_subtasks/total_subtasks*100):.1f}%\n"

        return report
    
    def save_markdown(self, filepath):
        md = self.generate_markdown()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"✅ Iterative markdown report saved to {filepath}")

    def save_html(self, filepath):
        try:
            import markdown
        except ImportError:
            raise ImportError("Install markdown: `pip install markdown`")

        html_content = markdown.markdown(self.generate_markdown(), extensions=["fenced_code", "tables"])
        html_content = f"<html><head><title>{self.task} Report</title></head><body>{html_content}</body></html>"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✅ Iterative HTML report saved to {filepath}")

    def export_all(self, project_name: str, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate filenames with iterative prefix
        filename_base = f"{project_name}_iterative_{self.task.lower().replace(' ', '_').replace('&', 'and')}_report"
        md_path = save_dir / f"{filename_base}.md"
        html_path = save_dir / f"{filename_base}.html"

        self.save_markdown(md_path)
        self.save_html(html_path)