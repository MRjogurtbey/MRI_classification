"""
NeuroBridge AI - REST API
FastAPI backend for MRI classification model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
from pathlib import Path
import logging
from PIL import Image
import io
import torch

from utils.inference import ModelInference
from config import MODEL_PATH, CLASS_NAMES, DEVICE_AUTO, IMAGE_SIZE, CLASS_DESCRIPTIONS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NeuroBridge AI - MRI Classification API",
    description="Brain tumor classification API using ResNet18",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (allow all origins for hackathon demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[ModelInference] = None


# Response models
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_path: str


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    description: str
    device_used: str


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        model = ModelInference(
            model_path=str(MODEL_PATH),
            class_names=CLASS_NAMES,
            device=DEVICE_AUTO,
            image_size=IMAGE_SIZE
        )
        logger.info(f"Model loaded successfully on {model.device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "NeuroBridge AI - MRI Classification API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(model.device) if model else "N/A",
        model_path=str(MODEL_PATH)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    use_tta: bool = False
):
    """
    Predict brain tumor type from MRI image
    
    Args:
        file: MRI image file (jpg, png, etc.)
        use_tta: Use Test-Time Augmentation for better accuracy (slower)
    
    Returns:
        Prediction results with class, confidence, and probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Predict
        logger.info(f"Processing image: {file.filename}, TTA: {use_tta}")
        
        if use_tta:
            result = model.predict_with_tta(
                image,
                num_augmentations=5,
                return_probabilities=True
            )
        else:
            result = model.predict(
                image,
                return_probabilities=True
            )
        
        # Get class description
        predicted_class = result['predicted_class']
        description = CLASS_DESCRIPTIONS.get(predicted_class, "No description available")
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            description=description,
            device_used=str(model.device)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/classes", response_model=Dict[str, str])
async def get_classes():
    """Get available classification classes and their descriptions"""
    return CLASS_DESCRIPTIONS


if __name__ == "__main__":
    # Run server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
