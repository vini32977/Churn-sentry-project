import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import yaml
import os

def train():
    with open("params.yaml") as f:
        config = yaml.safe_load(f)
    
    # Load the SEMI-CLEANED data (the one before get_dummies)
    # We'll modify transform.py to not do get_dummies next
    train_data = pd.read_csv("data/processed/train.csv").drop(columns=['customerID'])
    train_data['TotalCharges'] = pd.to_numeric(train_data['TotalCharges'], errors='coerce').fillna(0)
    
    X_train = train_data.drop("Churn", axis=1)
    y_train = train_data["Churn"].map({'Yes': 1, 'No': 0})

    # Identify categorical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Define the "Pre-processor"
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
        remainder='passthrough'
    )

    # Bundle Pre-processor and Model into one Pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=config["train"]["n_estimators"],
            max_depth=config["train"]["max_depth"],
            random_state=config["base"]["random_state"]
        ))
    ])

    print("Training the Production Pipeline...")
    clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, config["model_dir"])
    print("Stage 3: Production Pipeline saved!")

if __name__ == "__main__":
    train()