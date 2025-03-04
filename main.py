from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = joblib.load("model.pkl")
class_names = model.classes_

class PredictionInput(BaseModel):
    data: list[float] = Field(..., min_length=4, max_length=4)

@app.post("/predict")
async def predict(request: Request):
    try:
        # Parse JSON body
        body = await request.json()
        
        # Validate the input
        if 'data' not in body:
            return JSONResponse(
                status_code=400, 
                content={"error": "Missing 'data' field"}
            )
        
        input_data = body['data']
        
        # Ensure we have exactly 4 features
        if len(input_data) != 4:
            return JSONResponse(
                status_code=400, 
                content={"error": "Input must contain exactly 4 features"}
            )
        
        # Convert to numpy array and predict
        prediction = model.predict([np.array(input_data)])
        
        # Get the class name
        predicted_class = class_names[prediction[0]]
        
        return {
            "prediction": predicted_class
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": str(e)}
        )

# Add OPTIONS method for CORS preflight
@app.options("/predict")
async def options_predict():
    return JSONResponse(status_code=200)