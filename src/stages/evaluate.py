import pandas as pd
import joblib
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import mlflow
import os

def evaluate():
    with open("params.yaml") as f:
        config = yaml.safe_load(f)
    
    # LOAD THE RAW TEST DATA (not the final one)
    test_data = pd.read_csv("data/processed/test.csv")
    model = joblib.load(config["model_dir"])
    
    # Pre-process TotalCharges (just like we did in train.py)
    test_data['TotalCharges'] = pd.to_numeric(test_data['TotalCharges'], errors='coerce').fillna(0)
    
    # Separate features and target
    X_test = test_data.drop(columns=["Churn", "customerID"])
    y_test = test_data["Churn"].map({'Yes': 1, 'No': 0})
    
    print("Evaluating Production Pipeline...")
    predictions = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions)
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    with mlflow.start_run():
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
            
    print(f"Stage 4: Evaluation Complete. Metrics: {metrics}")

if __name__ == "__main__":
    evaluate()