# util/checklists.py
"""
Task checklists for iterative multi-agent system validation
"""

CHECKLISTS = {
    "Exploratory Data Analysis (EDA)": [
        "Understand data structure and types",
        "Identify missing values and patterns",
        "Explore feature distributions and statistics", 
        "Detect outliers and anomalies",
        "Assess feature correlations and relationships",
        "Generate meaningful visualizations",
        "Summarize key insights for modeling",
        "Document data quality issues"
    ],
    
    "Feature Engineering": [
        "Create meaningful derived features",
        "Handle missing values appropriately", 
        "Encode categorical variables properly",
        "Scale and normalize numerical features",
        "Remove irrelevant or redundant features",
        "Evaluate feature importance and selection",
        "Validate feature distributions", 
        "Prepare features for modeling"
    ],
    
    "Model Selection & Evaluation": [
        "Split data into train/validation/test sets",
        "Train multiple model candidates",
        "Tune hyperparameters systematically", 
        "Evaluate using appropriate metrics",
        "Perform cross-validation if needed",
        "Compare model performance objectively",
        "Interpret and validate model outputs",
        "Document final model selection rationale"
    ]
}

# Extended checklists for iterative process validation
ITERATIVE_PROCESS_CHECKLISTS = {
    "Planner Quality": [
        "Clear task decomposition provided",
        "Implementation instructions are detailed",
        "Strategic considerations addressed",
        "Technical requirements specified",
        "Success criteria defined"
    ],
    
    "Developer Quality": [
        "Code follows planner instructions",
        "Implementation is technically sound",
        "Proper error handling included",
        "Code is readable and documented",
        "Outputs are generated correctly"
    ],
    
    "Auditor Quality": [
        "Thorough review of planner instructions",
        "Comprehensive code evaluation",
        "Constructive feedback provided",
        "Improvement recommendations given",
        "Quality issues identified clearly"
    ],
    
    "Refinement Quality": [
        "Auditor feedback incorporated effectively",
        "Code improvements implemented",
        "Original requirements still met",
        "Enhanced functionality delivered",
        "Final implementation is robust"
    ]
}

# Phase-specific success criteria
PHASE_SUCCESS_CRITERIA = {
    "Exploratory Data Analysis (EDA)": {
        "data_understanding": "Complete understanding of dataset structure and characteristics",
        "quality_assessment": "Identification and documentation of data quality issues",
        "insight_generation": "Discovery of patterns and relationships relevant for modeling",
        "visualization_quality": "Clear and informative visualizations supporting analysis"
    },
    
    "Feature Engineering": {
        "feature_creation": "Meaningful features created to improve model performance", 
        "data_preprocessing": "Proper handling of missing values and data types",
        "feature_selection": "Removal of irrelevant features and selection of informative ones",
        "pipeline_readiness": "Features prepared in format suitable for model training"
    },
    
    "Model Selection & Evaluation": {
        "model_training": "Multiple models trained and compared systematically",
        "performance_evaluation": "Comprehensive evaluation using appropriate metrics",
        "hyperparameter_optimization": "Model parameters tuned for optimal performance",
        "model_validation": "Robust validation ensuring model generalizability"
    }
}

# Quality gates for iterative process
ITERATIVE_QUALITY_GATES = {
    "planning_completeness": {
        "threshold": 0.8,
        "description": "Planning phase must address 80% of expected requirements"
    },
    "implementation_accuracy": {
        "threshold": 0.9, 
        "description": "Implementation must follow 90% of planning instructions"
    },
    "audit_thoroughness": {
        "threshold": 0.85,
        "description": "Audit must cover 85% of critical review areas"
    },
    "refinement_improvement": {
        "threshold": 0.75,
        "description": "Refinement must address 75% of audit recommendations"
    }
}