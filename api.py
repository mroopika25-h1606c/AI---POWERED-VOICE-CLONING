"""
FastAPI Integration Example for Voice Spoofing Detection

This is a template that Member 3 (Backend) should use to integrate
the voice spoofing detector into their API.

Usage:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
    
    Then visit: http://localhost:8000/docs for interactive API docs
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import logging
import tempfile
import os
from pathlib import Path

# Import the detector
from inference.predict import VoiceSpoofingDetector, predict_audio

# ============================================================================
# SETUP
# ============================================================================

app = FastAPI(
    title="Voice Spoofing Detection API",
    description="Detect AI-generated voice cloning and deepfake impersonation attacks",
    version="1.0.0"
)

# Add CORS middleware (allow requests from frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize detector globally (loads model once on startup)
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize the detector on app startup."""
    global detector
    logger.info("🚀 Starting Voice Spoofing Detection API...")
    
    # Uncomment and provide path if you have a pretrained model
    # model_path = "models/aasist_best.pth"
    # detector = VoiceSpoofingDetector(model_path=model_path)
    
    # For now, use untrained model (for demo)
    detector = VoiceSpoofingDetector(model_path=None)
    logger.info("✓ API ready to receive requests!")


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class PredictionResponse(BaseModel):
    """Standard response format for predictions."""
    prediction: str  # "REAL" or "FAKE"
    confidence: float  # 0-100
    probabilities: Dict[str, float]  # {"real": 0.98, "fake": 0.02}
    audio_path: str
    message: str = "Prediction successful"


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str
    details: Optional[str] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Health check and API info endpoint."""
    return {
        "status": "healthy",
        "service": "Voice Spoofing Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Check if API and detector are working."""
    if detector is None:
        return {
            "status": "error",
            "message": "Detector not initialized"
        }
    return {
        "status": "healthy",
        "detector": "ready",
        "device": detector.device
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_voice(file: UploadFile = File(...)) -> Dict:
    """
    Detect if uploaded audio is real voice or AI-generated deepfake.
    
    **Request:**
    - Upload an audio file (WAV, FLAC, MP3, etc.)
    
    **Response:**
    ```json
    {
        "prediction": "REAL",
        "confidence": 98.5,
        "probabilities": {
            "real": 0.985,
            "fake": 0.015
        },
        "audio_path": "temp_audio.wav"
    }
    ```
    
    **Status Codes:**
    - 200: Prediction successful
    - 400: Invalid audio file
    - 500: Server error
    """
    
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    valid_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format. Supported: {', '.join(valid_extensions)}"
        )
    
    # Save uploaded file to temporary location
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Write uploaded file
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"📁 Processing: {file.filename}")
        
        # Run inference
        result = detector.predict(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Add success message
        result['message'] = "Prediction successful"
        
        logger.info(f"✓ Prediction: {result['prediction']}")
        return result
        
    except Exception as e:
        logger.error(f"✗ Error during prediction: {str(e)}")
        
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(files: list[UploadFile] = File(...)) -> Dict:
    """
    Predict on multiple audio files at once.
    
    **Request:**
    - Upload multiple audio files
    
    **Response:**
    ```json
    {
        "total": 3,
        "predictions": [
            {"prediction": "REAL", "confidence": 98.5, ...},
            {"prediction": "FAKE", "confidence": 87.3, ...},
            {"prediction": "REAL", "confidence": 92.1, ...}
        ]
    }
    ```
    """
    
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    predictions = []
    
    for file in files:
        try:
            temp_path = os.path.join(tempfile.gettempdir(), file.filename)
            contents = await file.read()
            
            with open(temp_path, "wb") as f:
                f.write(contents)
            
            result = detector.predict(temp_path)
            predictions.append(result)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            logger.error(f"✗ Error processing {file.filename}: {str(e)}")
            predictions.append({
                "audio_path": file.filename,
                "prediction": "ERROR",
                "error": str(e)
            })
    
    return {
        "total": len(predictions),
        "predictions": predictions
    }


# ============================================================================
# USAGE INSTRUCTIONS FOR MEMBER 3
# ============================================================================

"""
QUICK START GUIDE FOR BACKEND INTEGRATION:

1. INSTALL DEPENDENCIES:
   pip install -r requirements.txt
   pip install fastapi uvicorn python-multipart

2. RUN THE API:
   uvicorn api:app --reload

3. TEST THE API:
   - Interactive docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health
   - Upload and predict: POST http://localhost:8000/predict

4. USE IN YOUR FRONTEND/CLIENT:
   
   # Python client example
   import requests
   
   with open("audio.wav", "rb") as f:
       response = requests.post(
           "http://localhost:8000/predict",
           files={"file": f}
       )
   
   result = response.json()
   print(f"Prediction: {result['prediction']}")
   print(f"Confidence: {result['confidence']}%")

5. IMPORT DIRECTLY IN YOUR CODE:
   
   from inference.predict import VoiceSpoofingDetector
   
   detector = VoiceSpoofingDetector()
   result = detector.predict("audio.wav")
   print(result)

6. WITH PRETRAINED MODEL:
   
   detector = VoiceSpoofingDetector(model_path="models/aasist_best.pth")
   result = detector.predict("audio.wav")
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
