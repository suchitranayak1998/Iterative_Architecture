# util/pipeline_config.py
"""
Pipeline configuration for iterative 3-agent system
"""

PIPELINE_CONSTANTS = {
    "Dataset": "Pumpkin Seed Data",
    "Prediction Task": "Classification",
    "target_column": "Class",
    "random_seed": 42,
    "test_size": 0.2,
    "val_size": 0.1,  
    "model_type": "RandomForestClassifier",
    "evaluation_metrics": ["f1", "accuracy", "precision", "recall", "Confusion Matrix"],
    
    # Iterative-specific settings
    "architecture": "3-agent iterative",
    "agent_roles": ["Planner", "Developer", "Auditor"],
    "process_steps": 4,
    "max_debug_retries": 3
}

# Agent-specific configurations
AGENT_CONFIGS = {
    "Planner": {
        "temperature": 0.7,
        "focus": "strategic_planning",
        "output_format": "detailed_instructions"
    },
    "Developer": {
        "temperature": 0.2,
        "focus": "code_implementation", 
        "output_format": "executable_code"
    },
    "Auditor": {
        "temperature": 0.4,
        "focus": "quality_review",
        "output_format": "structured_feedback"
    }
}