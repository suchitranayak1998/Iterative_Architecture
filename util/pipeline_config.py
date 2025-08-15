PIPELINE_CONSTANTS = {

    "Dataset": "Pumpkin Seed Data",

    #Regression, Classification, Clustering, Forcasting
    "Prediction Task": "Classification",

    "target_column": "Class",

    "random_seed": 42,
    "test_size": 0.2,
    "val_size": 0.1,  

    # 🧠 Model Type
    "model_type": "RandomForestClassifier",  # used for prompt injection

    # 📏 Evaluation Metrics
    "evaluation_metrics": ["f1", "accuracy", "precision", "recall", "Confusion Matrix"],
}


