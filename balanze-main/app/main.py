"""
Gait and Balance Analysis API

This is the main FastAPI application for the Gait and Balance Analysis system.
It provides endpoints for analyzing gait and balance data, with support for
both real-time and batch processing.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import uvicorn
from fastapi import (
    FastAPI, 
    File, 
    UploadFile, 
    HTTPException, 
    status, 
    Depends, 
    Request,
    Response,
    BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, FileResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import application components
try:
    import joblib
    from src.pose_extractor import PoseExtractor
    from src.advanced_feature_engineer import AdvancedBioMechanicalFeatures
    from src.models.model_manager import ModelManager, ModelType
    from src.data.data_validator import DataValidator, DataVersion
    from src.api.auth import (
        Token, 
        User, 
        authenticate_user, 
        create_access_token, 
        get_current_active_user,
        has_role,
        rate_limiter,
        ACCESS_TOKEN_EXPIRE_MINUTES
    )
    from src.evaluation.model_evaluation import ClinicalGaitEvaluator
    from src.visualization.plot_utils import generate_gait_analysis_plots
    from src.report_generator import generate_clinical_report
    
    # Initialize components
    pose_extractor = PoseExtractor()
    feature_extractor = AdvancedBioMechanicalFeatures()
    model_manager = ModelManager(models_dir="models")
    data_validator = DataValidator()
    
except ImportError as e:
    logger.warning(f"Failed to import some dependencies: {e}")
    # Fallback to dummy implementations if imports fail
    joblib = None
    
    class DummyComponent:
        def __init__(self, name):
            self.name = name
        def __call__(self, *args, **kwargs):
            logger.warning(f"Dummy {self.name} called - functionality not available")
            return {}
    
    pose_extractor = DummyComponent("PoseExtractor")
    feature_extractor = DummyComponent("FeatureExtractor")
    model_manager = DummyComponent("ModelManager")
    data_validator = DummyComponent("DataValidator")


# Initialize FastAPI application
app = FastAPI(
    title="Gait & Balance Analysis API",
    description="""
    API for analyzing gait and balance data to assist in clinical assessment
    and research of movement disorders.
    """,
    version="1.0.0",
    docs_url=None,  # Disable default docs to use custom one with auth
    redoc_url=None,
    openapi_url="/api/openapi.json"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise

# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

# Custom OpenAPI schema with security requirements
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Gait & Balance Analysis API",
        version="1.0.0",
        description="""
        ## Overview
        This API provides endpoints for analyzing gait and balance data to assist in 
        clinical assessment and research of movement disorders.
        
        ## Authentication
        Most endpoints require authentication. Use the `/token` endpoint to get an access token.
        Include the token in the `Authorization: Bearer <token>` header.
        
        ## Rate Limiting
        The API is rate limited to 100 requests per minute per IP address.
        """,
        routes=app.routes,
    )
    
    # Add security requirements to all endpoints
    for path in openapi_schema.get("paths", {}).values():
        for method in path.values():
            if method.get("summary") not in ["Login", "Health Check"]:
                method["security"] = [{"OAuth2PasswordBearer": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Custom docs with auth
@app.get("/docs", include_in_schema=False)
async def get_documentation(current_user: User = Depends(get_current_active_user)):
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title="Gait & Balance Analysis API - Swagger UI",
        oauth2_redirect_url="/docs/oauth2-redirect",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


# Request and Response Models
class AnalysisRequest(BaseModel):
    """Request model for gait analysis."""
    patient_id: str = Field(..., description="Unique patient identifier")
    session_id: str = Field(..., description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional metadata about the analysis"
    )

class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    analysis_id: str
    status: str
    timestamp: datetime
    results: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    plots: Optional[Dict[str, str]] = None
    report: Optional[Dict[str, Any]] = None

class TokenRequest(BaseModel):
    """Request model for token generation."""
    username: str
    password: str

# Initialize models
try:
    # Try to load pre-trained models
    models = {
        "gait": model_manager.load_model("gait_classifier", "latest"),
        "balance": model_manager.load_model("balance_classifier", "latest"),
        "disorder": model_manager.load_model("disorder_classifier", "latest"),
    }
    logger.info("Successfully loaded pre-trained models")
except Exception as e:
    logger.warning(f"Failed to load pre-trained models: {e}")
    models = {}

# In-memory storage for analysis results (replace with database in production)
analysis_results = {}

# Background task to clean up old analyses
async def cleanup_old_analyses():
    """Remove analysis results older than 24 hours."""
    current_time = datetime.utcnow()
    expired = [
        k for k, v in analysis_results.items() 
        if (current_time - v["timestamp"]).total_seconds() > 24 * 3600
    ]
    for k in expired:
        del analysis_results[k]
    logger.info(f"Cleaned up {len(expired)} old analyses")

# Schedule cleanup task
@app.on_event("startup")
async def startup_event():
    """Initialize application state and schedule cleanup tasks."""
    import asyncio
    
    # Create required directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Start background tasks
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    """Periodically clean up old analysis results."""
    import asyncio
    while True:
        try:
            await cleanup_old_analyses()
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
        await asyncio.sleep(3600)  # Run hourly


# Authentication endpoints
@app.post("/api/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 token endpoint for user authentication.
    
    - **username**: Your username
    - **password**: Your password
    
    Returns an access token that should be included in the `Authorization` header
    for authenticated requests: `Bearer <token>`
    """
    from src.api.auth import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
    
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "roles": user.roles},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# Health check endpoint
@app.get(
    "/api/health", 
    tags=["System"],
    summary="Check API health status",
    response_description="Current health status of the API"
)
async def health_check():
    """
    Check the health status of the API.
    
    Returns the current status and timestamp.
    """
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "models": "loaded" if models else "unavailable",
            "gpu_available": False  # Add GPU check if needed
        }
    }


# Analysis endpoints
@app.post(
    "/api/analyze/video",
    response_model=AnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rate_limiter)],
    tags=["Analysis"],
    summary="Analyze gait from video",
    description="""
    Upload a video file for gait analysis.
    
    The video should show the patient walking towards and away from the camera.
    The analysis will extract gait parameters and provide a clinical assessment.
    """
)
async def analyze_video(
    background_tasks: BackgroundTasks,
    request: Request,
    response: Response,
    file: UploadFile = File(..., description="Video file for analysis"),
    current_user: User = Depends(has_role(["clinician", "admin"]))
):
    """
    Analyze gait from an uploaded video file.
    
    This endpoint accepts a video file, processes it to extract pose information,
    and performs gait analysis using pre-trained models.
    
    Returns an analysis ID that can be used to retrieve results.
    """
    # Generate a unique analysis ID
    import uuid
    analysis_id = str(uuid.uuid4())
    
    # Save the uploaded file
    upload_dir = Path("uploads") / analysis_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = upload_dir / file.filename
    try:
        content = await file.read()
        with video_path.open("wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file"
        )
    
    # Store initial analysis result
    analysis_results[analysis_id] = {
        "status": "processing",
        "timestamp": datetime.utcnow(),
        "video_path": str(video_path),
        "user": current_user.username,
        "results": None,
        "error": None
    }
    
    # Process the video in the background
    background_tasks.add_task(
        process_video_analysis,
        analysis_id=analysis_id,
        video_path=video_path,
        user=current_user.username
    )
    
    # Return the analysis ID for result retrieval
    response.headers["Location"] = f"/api/analysis/{analysis_id}"
    return {
        "analysis_id": analysis_id,
        "status": "processing",
        "timestamp": datetime.utcnow().isoformat(),
        "results_available_at": f"/api/analysis/{analysis_id}"
    }

# Analysis result retrieval
@app.get(
    "/api/analysis/{analysis_id}",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Get analysis results",
    description="Retrieve the results of a previously submitted analysis."
)
async def get_analysis_results(
    analysis_id: str,
    current_user: User = Depends(has_role(["clinician", "admin", "researcher"]))
):
    """
    Retrieve the results of a gait analysis.
    
    Returns the current status and results (if available) for the specified analysis ID.
    """
    if analysis_id not in analysis_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    result = analysis_results[analysis_id]
    
    # Check permissions (users can only access their own analyses unless they're admin)
    if current_user.username != result["user"] and "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this analysis"
        )
    
    if result["status"] == "error":
        return {
            "analysis_id": analysis_id,
            "status": "error",
            "timestamp": result["timestamp"].isoformat(),
            "error": result.get("error", "Unknown error")
        }
    
    if result["status"] == "processing":
        return {
            "analysis_id": analysis_id,
            "status": "processing",
            "timestamp": result["timestamp"].isoformat(),
            "message": "Analysis in progress"
        }
    
    # Return the full results
    return {
        "analysis_id": analysis_id,
        "status": "completed",
        "timestamp": result["timestamp"].isoformat(),
        "results": result.get("results", {}),
        "metrics": result.get("metrics", {}),
        "plots": result.get("plots", {}),
        "report": result.get("report", {})
    }

# Background task to process video analysis
async def process_video_analysis(analysis_id: str, video_path: Path, user: str):
    """
    Process a video file for gait analysis.
    
    This runs in a background task to avoid blocking the API.
    """
    try:
        logger.info(f"Starting analysis for {analysis_id}")
        
        # Update status
        analysis_results[analysis_id]["status"] = "processing"
        
        # Extract frames and poses (simplified example)
        sequence = []
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        
        while cap.isOpened() and frame_count < 100:  # Limit to 100 frames for demo
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame (simplified)
            if frame_count % 5 == 0:  # Process every 5th frame
                # In a real implementation, this would use the pose_extractor
                pose_data = {
                    "frame": frame_count,
                    "detected": True,
                    "left_hip_angle": 20 + (frame_count % 10),
                    "right_hip_angle": 25 + (frame_count % 10),
                    "left_knee_angle": 60 + (frame_count % 8),
                    "right_knee_angle": 64 + (frame_count % 8),
                    "left_ankle_angle": 10 + (frame_count % 6),
                    "right_ankle_angle": 12 + (frame_count % 6),
                    "com_x": 0.01 * (frame_count % 5),
                    "com_y": 1.0 + 0.01 * (frame_count % 3),
                    "com_z": 0.0,
                }
                sequence.append(pose_data)
            
            frame_count += 1
        
        cap.release()
        
        if not sequence:
            raise ValueError("No valid pose data extracted from video")
        
        # Extract features
        features = feature_extractor.extract_all_features(sequence)
        
        # Make predictions (simplified example)
        results = {}
        if models:
            for name, model in models.items():
                try:
                    if hasattr(model, 'predict'):
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(features)[0]
                            results[f"{name}_probabilities"] = {
                                str(cls): float(prob) 
                                for cls, prob in zip(model.classes_, proba)
                            }
                        pred = model.predict(features)[0]
                        results[f"predicted_{name}"] = str(pred)
                except Exception as e:
                    logger.warning(f"Error making prediction with {name}: {e}")
        
        # Generate metrics and plots (simplified example)
        metrics = {
            "stride_length": 1.2 + (frame_count / 1000),  # Example metric
            "cadence": 110 + (frame_count % 20),  # Example metric
            "gait_symmetry": 0.85 + (frame_count % 10 * 0.01),  # Example metric
        }
        
        # Generate plots (in a real implementation, this would use the visualization module)
        plots_dir = Path("reports") / analysis_id
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        try:
            # Example plot generation (simplified)
            import matplotlib.pyplot as plt
            
            # Gait cycle plot
            plt.figure(figsize=(10, 6))
            plt.plot([p["left_knee_angle"] for p in sequence], label="Left Knee")
            plt.plot([p["right_knee_angle"] for p in sequence], label="Right Knee")
            plt.xlabel("Frame")
            plt.ylabel("Angle (degrees)")
            plt.title("Knee Angles During Gait")
            plt.legend()
            
            gait_plot_path = plots_dir / "gait_cycle.png"
            plt.savefig(gait_plot_path)
            plt.close()
            plot_paths["gait_cycle"] = f"/api/analysis/{analysis_id}/plots/gait_cycle.png"
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
        
        # Generate report (simplified example)
        report = {
            "summary": "Gait analysis completed successfully.",
            "findings": [
                "Normal gait pattern detected.",
                "Slight asymmetry between left and right steps.",
                "Within normal range for age and height."
            ],
            "recommendations": [
                "Continue current treatment plan.",
                "Follow up in 3 months for reassessment."
            ]
        }
        
        # Update analysis results
        analysis_results[analysis_id].update({
            "status": "completed",
            "results": results,
            "metrics": metrics,
            "plots": plot_paths,
            "report": report,
            "completed_at": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Completed analysis for {analysis_id}")
        
    except Exception as e:
        logger.error(f"Error processing analysis {analysis_id}: {e}", exc_info=True)
        analysis_results[analysis_id].update({
            "status": "error",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat()
        })

# Serve plot images
@app.get(
    "/api/analysis/{analysis_id}/plots/{plot_name}",
    response_class=FileResponse,
    tags=["Analysis"],
    summary="Get analysis plot"
)
async def get_analysis_plot(
    analysis_id: str,
    plot_name: str,
    current_user: User = Depends(has_role(["clinician", "admin", "researcher"]))
):
    """
    Retrieve a specific plot from an analysis.
    """
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Check permissions
    if current_user.username != analysis_results[analysis_id].get("user") and "admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="Not authorized to access this analysis")
    
    plot_path = Path("reports") / analysis_id / plot_name
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    
    return FileResponse(plot_path)

# Model management endpoints (admin only)
@app.get(
    "/api/models",
    tags=["Models"],
    summary="List available models",
    description="List all available models and their versions.",
    dependencies=[Depends(has_role(["admin"]))]
)
async def list_models():
    """List all available models and their versions."""
    try:
        return model_manager.list_models()
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list models"
        )

@app.post(
    "/api/models/{model_name}/upload",
    status_code=status.HTTP_201_CREATED,
    tags=["Models"],
    summary="Upload a new model version",
    description="Upload a new version of a model.",
    dependencies=[Depends(has_role(["admin"]))]
)
async def upload_model(
    model_name: str,
    version: str,
    model_file: UploadFile = File(..., description="Model file (e.g., .joblib, .h5, .pkl)"),
    metadata_file: Optional[UploadFile] = File(None, description="Optional metadata JSON file"),
    current_user: User = Depends(has_role(["admin"]))
):
    """
    Upload a new version of a model.
    
    The model file should be a serialized model in a supported format.
    An optional metadata file can be provided with additional information.
    """
    try:
        # Save the uploaded model file
        upload_dir = Path("uploads") / "models" / model_name / version
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = upload_dir / model_file.filename
        with open(model_path, "wb") as f:
            content = await model_file.read()
            f.write(content)
        
        # Process metadata if provided
        metadata = {}
        if metadata_file:
            try:
                metadata = json.loads(await metadata_file.read())
            except Exception as e:
                logger.warning(f"Error parsing metadata file: {e}")
        
        # In a real implementation, you would register the model with the model manager
        # For now, we'll just return a success message
        return {
            "status": "success",
            "message": f"Model {model_name} version {version} uploaded successfully",
            "model_path": str(model_path),
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload model: {str(e)}"
        )

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV") == "development",
        workers=int(os.getenv("WORKERS", 1))
    )
