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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load the model with error handling
try:
    model = joblib.load("model.pkl")
    class_names = model.classes_
except Exception as e:
    print(f"Error loading model: {e}")
    class_names = []

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
        
        # Convert to numpy array with explicit dtype
        input_array = np.array(input_data, dtype=np.float64).reshape(1, -1)
        
        # Debug print
        print("Input array shape:", input_array.shape)
        print("Input array:", input_array)
        print("Model classes:", class_names)
        
        # Predict with error handling
        try:
            prediction = model.predict(input_array)
            print("Raw prediction:", prediction)
        except Exception as pred_error:
            print(f"Prediction error: {pred_error}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Prediction failed: {str(pred_error)}"}
            )
        
        # Get the class name
        try:
            predicted_class = class_names[int(prediction[0])]
        except Exception as class_error:
            print(f"Class selection error: {class_error}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Class selection failed: {str(class_error)}"}
            )
        
        return {
            "prediction": predicted_class
        }
    except Exception as e:
        print(f"Unexpected error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": str(e)}
        )

# Endpoint to check model details
@app.get("/model-info")
async def model_info():
    return {
        "classes": list(class_names),
        "model_type": str(type(model))
    }