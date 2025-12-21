import pandas as pd
import yaml
import os

def transform():
    # Load data
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    # 1. Handle TotalCharges (Convert text to numbers, fill empty spaces with 0)
    train['TotalCharges'] = pd.to_numeric(train['TotalCharges'], errors='coerce').fillna(0)
    test['TotalCharges'] = pd.to_numeric(test['TotalCharges'], errors='coerce').fillna(0)

    # 2. Encode Target variable (Churn) to 0 and 1
    train['Churn'] = train['Churn'].map({'Yes': 1, 'No': 0})
    test['Churn'] = test['Churn'].map({'Yes': 1, 'No': 0})

    # 3. Simple Encoding for other text columns (Dropping ID as it's useless for AI)
    train.drop(columns=['customerID'], inplace=True)
    test.drop(columns=['customerID'], inplace=True)
    
    # We will use "get_dummies" to turn categories into 1s and 0s
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)

    # Save cleaned data
    os.makedirs("data/final", exist_ok=True)
    train.to_csv("data/final/train_final.csv", index=False)
    test.to_csv("data/final/test_final.csv", index=False)
    
    print("Stage 2: Transformation Success. Data cleaned and saved to data/final/")

if __name__ == "__main__":
    transform()