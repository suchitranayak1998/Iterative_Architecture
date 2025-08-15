# reporting/summarizer.py
"""
Report summarizer for iterative multi-agent system
"""

import os
from pathlib import Path
import markdown
from langchain_core.messages import SystemMessage, HumanMessage

class ReportSummarizer:
    def __init__(self, results: list, task: str, output_path: str, llm, pipeline_state):
        self.results = results
        self.task = task
        self.output_path = output_path
        self.report_text = ""
        self.summary_html = ""
        self.llm = llm
        self.pipeline_state = pipeline_state  

    def generate_markdown(self) -> str:
        """Generate markdown report for iterative process"""
        report = f"# {self.task} Report - Iterative Process\n\n"

        for idx, entry in enumerate(self.results, 1):
            report += f"## 🔍 Task {idx}: {entry['subtask']}\n\n"

            # Check if this is iterative process result
            if "iterative_process" in entry:
                iterative_data = entry["iterative_process"]
                
                # Step 1: Planner
                if "planner_output" in iterative_data:
                    planner = iterative_data["planner_output"]
                    report += f"### 📋 Step 1: {planner.get('agent', 'Planner')} (Strategic Planning)\n"
                    report += f"{planner.get('planning_instructions', 'No instructions')}\n\n"

                # Step 2: Initial Developer
                if "initial_developer_output" in iterative_data:
                    initial_dev = iterative_data["initial_developer_output"]
                    report += f"### 💻 Step 2: {initial_dev.get('agent', 'Developer')} (Initial Implementation)\n"
                    report += f"{initial_dev.get('implementation', 'No implementation')}\n\n"

                # Step 3: Auditor Review
                if "auditor_feedback" in iterative_data:
                    auditor = iterative_data["auditor_feedback"]
                    report += f"### 🔍 Step 3: {auditor.get('agent', 'Auditor')} (Quality Review)\n"
                    report += f"{auditor.get('audit_feedback', 'No feedback')}\n\n"

                # Step 4: Final Developer
                if "final_developer_output" in iterative_data:
                    final_dev = iterative_data["final_developer_output"]
                    report += f"### 🔧 Step 4: {final_dev.get('agent', 'Developer')} (Refined Implementation)\n"
                    report += f"{final_dev.get('final_implementation', 'No final implementation')}\n\n"

            else:
                # Fallback for non-iterative format
                if "conversation" in entry:
                    report += "### 🧠 Team Discussion\n"
                    for turn in entry["conversation"]:
                        report += f"**{turn['name']} ({turn['role']})**:\n{turn['message']}\n\n"

                if "developer_reply" in entry:
                    report += "### 🎯 Developer Response\n"
                    report += f"{entry['developer_reply']}\n\n"

            # Execution Output
            report += "### 🖥 Execution Output\n"
            result = entry.get('execution_result', '')
            if result:
                report += f"```\n{result}\n```\n"
                  
            report += "---\n\n"

        return report

    def prepare_report_html(self):
        """Convert markdown to HTML"""
        md = self.generate_markdown()
        filepath = f"temp_{self.task.replace(' ', '_')}.md"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"✅ Markdown report saved to {filepath}")
        
        self.report_text = markdown.markdown(md, extensions=["fenced_code", "tables"])

    def summarize(self):
        """Generate AI summary of the iterative process"""
        print("✍️ Summarizing Iterative Report...")

        # Enhanced prompt for iterative process
        prompt = f"""
        Summarize the following iterative data science report. This report shows a 4-step process:
        1. Planner: Strategic planning and task breakdown
        2. Developer: Initial implementation  
        3. Auditor: Quality review and feedback
        4. Developer: Refined implementation

        Your summary should be structured and comprehensive. Focus on:

        1. **Strategic Insights**: Key planning decisions and rationale
        2. **Implementation Quality**: Code quality and technical approach  
        3. **Audit Findings**: Critical feedback and improvement areas
        4. **Final Outcomes**: Results after refinement and data insights
        5. **Process Effectiveness**: How well the iterative approach worked
        6. **Technical Outputs**: Important tables, metrics, or visualizations
        7. **Next Phase Recommendations**: Guidance for subsequent work

        Provide the summary in HTML format with clear headings and bullet points.

        The summary should help the team understand:
        - What was accomplished in this iterative cycle
        - Key insights for the next phase ({self.task} → next phase)
        - Quality improvements achieved through the audit process

        Report from iterative task {self.task}:
        {self.report_text}
        """
            
        messages = [
            SystemMessage(content="You are a senior data scientist summarizing iterative multi-agent workflows for team coordination."),
            HumanMessage(content=prompt)
        ]
            
        response = self.llm(messages)
        self.summary_html = response.content.strip()

        # Store in pipeline state
        if self.pipeline_state and self.task:
            self.pipeline_state.add_phase_summary(
                phase=self.task,
                summary_html=self.summary_html
            )

    def save_summary(self):
        """Save the HTML summary"""
        print(f"💾 Saving iterative summary to: {self.output_path}")
        Path(self.output_path).write_text(self.summary_html, encoding="utf-8")

    def run(self):
        """Execute the complete summarization process"""
        self.prepare_report_html()
        self.summarize()
        self.save_summary()
        print("✅ Iterative summary generation complete!")