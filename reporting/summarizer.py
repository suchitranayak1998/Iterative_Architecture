from pathlib import Path
import markdown
from langchain_core.messages import SystemMessage, HumanMessage

from core import pipeline_state

class ReportSummarizer:
    def __init__(self, results: list, title: str, output_path: str, llm, pipeline_state=None, phase_name: str = None):
        self.results = results or []
        self.title = title
        self.output_path = output_path
        self.report_text = ""
        self.summary_html = ""
        self.llm = llm
        self.pipeline_state = pipeline_state  
        self.phase_name = phase_name   

    def generate_markdown(self) -> str:
        report = f"# {self.title}\n\n"

        for idx, entry in enumerate(self.results, 1):
            subtask = entry.get("subtask", f"Subtask {idx}")
            report += f"## 🔍 {subtask}\n\n"

            instruction = entry.get("manager_instruction") or entry.get("Implementation_Plan") or ""
            if instruction:
                report += "### 🎯 Instruction / Plan\n"
                report += f"{instruction}\n\n"

            exec_out = entry.get("execution_result", "")
            report += "### 🖥 Execution Output\n"
            report += f"```\n{exec_out}\n```\n"

            report += "---\n\n"

        return report

    def prepare_report_html(self):
        md = self.generate_markdown()
        self.report_text = markdown.markdown(md, extensions=["fenced_code", "tables"])

    def summarize(self):
        print("✍️ Summarizing report...")
        prompt = f"""
Summarize the following report. Your summary should be structured and Comprehensive. Focus on:

            1. Key data patterns or anomalies  
            2. Data cleaning or transformations  
            3. Analytical insights relevant for feature selection , Model Building, and Model Evaluation
            4. Business takeaways or hypotheses for modeling
            5. show any tables or outputs that are important for the team to know.
            6. Create a table for model evaluation metrics and results if available.


            Provide the summary in HTML format with clear headings and bullet points.

            There should be summary of each subtask and the result of each subtask, and a final summary of the entire report.

            This report is intended to guide the team of AI Agents in transitioning from Exploratory Data Analysis (EDA) to Feature Engineering to Model Building, and Model Evaluation.

Report for: {self.title}
{self.report_text}
"""
        messages = [
            SystemMessage(content="You are a senior data scientist preparing a structured phase summary."),
            HumanMessage(content=prompt)
        ]
        response = self.llm(messages)
        self.summary_html = response.content.strip()

        if self.pipeline_state and self.phase_name:
            # assumes PipelineState.add_phase_summary(phase, summary_html_or_path, is_path=False)
            self.pipeline_state.add_phase_summary(
                phase=self.phase_name,
                summary_html=self.summary_html
            )

    def save_summary(self):
        Path(self.output_path).write_text(self.summary_html, encoding="utf-8")

    def run(self):
        self.prepare_report_html()
        self.summarize()
        self.save_summary()
        print(f"✅ Summary written to {self.output_path}")
