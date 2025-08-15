# core/pipeline_state.py
"""
Pipeline state management for iterative multi-agent system
"""

from typing import List, Any, Optional, Dict
import json
import os
import pandas as pd
import pickle
from datetime import datetime
from pathlib import Path
import hashlib
import re
import html as _html

class PipelineState:
    def __init__(self, project_name: str = "default"):
        self.df = None
        
        # Hash for data freshness checking
        self.original_hash = None
        
        # Pipeline tracking attributes
        self.phase_history = []
        self.subtask_history = []
        self.code_history = []
        self.summary_history = []
        self.execution_log = []
        self.code_snippets = []
        self.df_transform_history = []
        self.validation_log = []

        # Iterative process tracking
        self.iterative_process_log = []
        self.agent_interaction_log = []

        # Persistence attributes
        self.project_name = project_name
        self.save_dir = Path(f"./pipeline_cache/{project_name}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.save_dir / "dataframes").mkdir(exist_ok=True)
        (self.save_dir / "reports").mkdir(exist_ok=True)
        (self.save_dir / "phases").mkdir(exist_ok=True)

    ###################################################
    ############Logging and Tracking##################
    ###################################################

    def add_iterative_process_log(self, phase: str, process_data: Dict):
        """Store complete iterative process data"""
        self.iterative_process_log.append({
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "process_data": process_data
        })

    def add_agent_interaction(self, agent_name: str, role: str, step: str, content: str):
        """Log agent interactions during iterative process"""
        self.agent_interaction_log.append({
            "agent_name": agent_name,
            "role": role,
            "step": step,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def add_validation_result(self, subtask: str, subtask_index: int, validation_tests: List[Dict]):
        """Store validation results for a subtask"""
        self.validation_log.append({
            "subtask_name": subtask,
            "subtask_index": subtask_index,
            "tests": validation_tests
        })

    def get_validation_log(self):
        """Get current validation log for the phase"""
        return self.validation_log

    def clear_validation_log(self):
        """Clear validation log when starting a new phase"""
        self.validation_log = []
    
    def add_transform(self, line: str):
        """Append DataFrame transformation history"""
        self.df_transform_history.append(line)

    def get_recent_transforms(self, n: int = 10) -> list:
        """Fetch recent DataFrame transformation lines"""
        return self.df_transform_history[-n:] if n else self.df_transform_history

    def update_phase(self, phase_name: str, subtasks: List[str]):
        """Update the current phase with subtasks"""
        self.phase_history.append({"phase": phase_name, "subtasks": subtasks})

    def add_subtask_result(self, phase: str, subtask: str, summary: str, code: str, result: str, images: List[str]):
        """Add a subtask result to the pipeline state"""
        self.subtask_history.append({"phase": phase, "subtask": subtask, "code": code})
        self.code_history.append(code)
        self.summary_history.append({"subtask": subtask, "summary": summary})
        self.execution_log.append({"subtask": subtask, "execution_result": result})

    def get_all_subtasks(self):
        """Get all subtasks across all phases"""
        return [entry["subtask"] for entry in self.subtask_history]
    
    def get_code_history(self):
        """Get all code snippets executed in the pipeline"""
        return [code for code in self.code_history if code.strip()]
    
    def get_recent_code_history(self, n=5):
        """Get the last n code snippets executed in the pipeline"""
        return self.code_history[-n:] if self.code_history else []
    
    def _get_df_hash(self, df: pd.DataFrame) -> str:
        """Generate hash of dataframe for change detection"""
        if df is None:
            return ""
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    
    def add_phase_summary(self, phase: str, summary_html: str) -> None:
        """Store the LLM-rendered HTML summary for a phase"""
        # Overwrite if the same phase already exists
        self.summary_history = [s for s in self.summary_history if s.get("phase") != phase]
        self.summary_history.append({"phase": phase, "summary": str(summary_html or "")})
    
    @staticmethod
    def _strip_html(html_text: str) -> str:
        """Basic HTML → text: remove tags, collapse whitespace, unescape entities"""
        if not html_text:
            return ""
        text = re.sub(r"<style.*?>.*?</style>|<script.*?>.*?</script>", "", html_text, flags=re.S|re.I)
        text = re.sub(r"<[^>]+>", "", text)
        text = _html.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def get_contextual_summary(self, last_n: int = 3, max_chars: int = 12000, strip_html: bool = True) -> str:
        """Return a compact summary of the last N phases for prompt context"""
        if not self.summary_history:
            return ""

        items = self.summary_history[-last_n:] if last_n else self.summary_history
        blocks = []
        for item in items:
            phase = item.get("phase", "Phase")
            content = item.get("summary", "")
            content = self._strip_html(content) if strip_html else str(content)
            blocks.append(f"## {phase} Summary\n{content}".strip())

        text = "\n\n".join(blocks).strip()

        if max_chars and len(text) > max_chars:
            text = text[-max_chars:]
        
        return text

    ###############################################
    #########Checkpointing and Persistence#########
    ###############################################

    def save_phase(self, phase_name: str, phase_results: List[Dict], personas: List = None):
        """Save current pipeline state after completing a phase"""
        
        phase_safe_name = phase_name.lower().replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "")
        
        # Save dataframes
        if self.df is not None:
            df_path = self.save_dir / "dataframes" / f"{phase_safe_name}_df.parquet"
            self.df.to_parquet(df_path)
            print(f"📄 Dataframe saved: {df_path}")

        # Save phase results
        phase_data = {
            "phase_name": phase_name,
            "timestamp": datetime.now().isoformat(),
            "original_hash": self.original_hash,
            "current_hash": self._get_df_hash(self.df),
            "phase_results": phase_results,
            "personas": [p.dict() if hasattr(p, 'dict') else p for p in personas] if personas else [],
            "pipeline_state": {
                "phase_history": self.phase_history,
                "subtask_history": self.subtask_history,
                "code_history": self.code_history,
                "summary_history": self.summary_history,
                "execution_log": self.execution_log,
                "df_transform_history": self.df_transform_history,
                "iterative_process_log": self.iterative_process_log,
                "agent_interaction_log": self.agent_interaction_log
            }
        }
        
        phase_file = self.save_dir / "phases" / f"{phase_safe_name}.json"
        with open(phase_file, 'w', encoding='utf-8') as f:
            json.dump(phase_data, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        self._save_metadata()
        
        print(f"✅ Phase '{phase_name}' saved to: {phase_file}")

    def _save_metadata(self):
        """Save project metadata"""
        completed_phases = self.get_completed_phases()
        metadata = {
            "project_name": self.project_name,
            "original_hash": self.original_hash,
            "df_shape": self.df.shape if self.df is not None else None,
            "completed_phases": completed_phases,
            "last_updated": datetime.now().isoformat(),
            "total_subtasks": len(self.subtask_history),
            "total_transforms": len(self.df_transform_history),
            "total_iterative_processes": len(self.iterative_process_log),
            "save_directory": str(self.save_dir),
            "architecture": "3-agent iterative"
        }
        
        with open(self.save_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_completed_phases(self) -> List[str]:
        """Get list of completed phases"""
        phase_dir = self.save_dir / "phases"
        if not phase_dir.exists():
            return []
        
        phase_files = list(phase_dir.glob("*.json"))
        completed = []
        
        phase_mapping = {
            "exploratory_data_analysis_eda": "Exploratory Data Analysis (EDA)",
            "feature_engineering": "Feature Engineering", 
            "model_selection_and_evaluation": "Model Selection & Evaluation"
        }
        
        for file in phase_files:
            phase_key = file.stem
            if phase_key in phase_mapping:
                completed.append(phase_mapping[phase_key])
        
        return completed
    
    def load_from_phase(self, phase_name: str) -> bool:
        """Load pipeline state up to and including the specified phase"""
        phase_safe_name = phase_name.lower().replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "")
        phase_file = self.save_dir / "phases" / f"{phase_safe_name}.json"
        
        if not phase_file.exists():
            print(f"❌ No saved state found for phase: {phase_name}")
            print(f"   Looking for file: {phase_file}")
            return False
        
        # Load phase data
        with open(phase_file, 'r', encoding='utf-8') as f:
            phase_data = json.load(f)
        
        # Restore pipeline state
        state = phase_data["pipeline_state"]
        self.phase_history = state["phase_history"]
        self.subtask_history = state["subtask_history"]
        self.code_history = state["code_history"]
        self.summary_history = state["summary_history"]
        self.execution_log = state["execution_log"]
        self.df_transform_history = state["df_transform_history"]
        
        # Load iterative-specific data if available
        if "iterative_process_log" in state:
            self.iterative_process_log = state["iterative_process_log"]
        if "agent_interaction_log" in state:
            self.agent_interaction_log = state["agent_interaction_log"]
        
        if "original_hash" in phase_data:
            self.original_hash = phase_data["original_hash"]
        
        # Load DataFrame
        df_path = self.save_dir / "dataframes" / f"{phase_safe_name}_df.parquet"
        if df_path.exists():
            self.df = pd.read_parquet(df_path)
            print(f"📄 Dataframe loaded: {self.df.shape}")
        
        print(f"✅ Pipeline state loaded from phase: {phase_name}")
        print(f"📊 Loaded {len(self.subtask_history)} subtasks, {len(self.df_transform_history)} transforms")
        return True
    
    def get_project_summary(self) -> Dict:
        """Get a summary of the current project state"""
        completed = self.get_completed_phases()
        return {
            "project_name": self.project_name,
            "architecture": "3-agent iterative",
            "completed_phases": completed,
            "total_subtasks": len(self.subtask_history),
            "total_transforms": len(self.df_transform_history),
            "total_iterative_processes": len(self.iterative_process_log),
            "total_agent_interactions": len(self.agent_interaction_log),
            "dataframe_shape": self.df.shape if self.df is not None else None,
            "save_directory": str(self.save_dir)
        }