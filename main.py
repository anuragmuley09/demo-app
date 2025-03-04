from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")

class PredictionInput(BaseModel):
    data: list[float] = Field(..., min_length=4, max_length=4)

@app.post("/predict")
async def predict(request: Request):
    # Parse the request body directly
    body = await request.json()
    
    # Validate the input manually
    if 'data' not in body:
        return {"error": "Missing 'data' field"}
    
    # Ensure we have exactly 4 features
    input_data = body['data']
    if len(input_data) != 4:
        return {"error": "Input must contain exactly 4 features"}
    
    try:
        # Convert to numpy array and predict
        prediction = model.predict([np.array(input_data)])
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

# Optional: Add a health check endpoint
@app.get("/")
def health_check():
    return {"status": "healthy"}