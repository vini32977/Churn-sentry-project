from fastapi import FastAPI
import joblib
import pandas as pd
import yaml

app = FastAPI()

# 1. Load the model and params when the app starts
with open("params.yaml") as f:
    config = yaml.safe_load(f)

model = joblib.load(config["model_dir"])

@app.get("/")
def home():
    return {"message": "Churn Prediction API is Running"}

@app.post("/predict")
def predict(data: dict):
    # Convert the incoming JSON data to a DataFrame
    df = pd.DataFrame([data])
    
    # Make a prediction
    prediction = model.predict(df)
    
    # Return the result
    result = "Churn" if prediction[0] == 1 else "No Churn"
    return {"prediction": result}