from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: list):
    prediction = model.predict([np.array(data)])
    return {"prefiction": int(prediction[0])}

