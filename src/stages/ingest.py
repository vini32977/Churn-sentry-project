import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

def ingest():
    # Load configuration
    with open("params.yaml") as f:
        config = yaml.safe_load(f)
    
    # Read raw data
    df = pd.read_csv("data/raw/Data.csv")
    
    # Split into Train (80%) and Test (20%)
    train, test = train_test_split(
        df, 
        test_size=config["data_ingestion"]["test_size"], 
        random_state=config["base"]["random_state"]
    )
    
    # Save the results
    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    print("Stage 1: Data split into train.csv and test.csv in data/processed/")

if __name__ == "__main__":
    ingest()