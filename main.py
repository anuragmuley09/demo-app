from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")

# Create a Pydantic model to define the expected input structure
class PredictionInput(BaseModel):
    data: list[float]

@app.post("/predict")
def predict(input_data: PredictionInput):
    prediction = model.predict([np.array(input_data.data)])
    return {"prediction": int(prediction[0])}