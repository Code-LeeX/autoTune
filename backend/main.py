"""
Main entry point for the AI Pitch Correction Backend API
"""
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import settings

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI-powered pitch correction tool backend API"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:3001", "http://127.0.0.1:3001",
        "http://localhost:3002", "http://127.0.0.1:3002",
        "http://localhost:3003", "http://127.0.0.1:3003",
        "http://localhost:3004", "http://127.0.0.1:3004",
        "http://localhost:3005", "http://127.0.0.1:3005",
        "http://localhost:3006", "http://127.0.0.1:3006"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.temp_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)

# Mount static files for audio downloads
app.mount("/downloads", StaticFiles(directory=settings.output_dir), name="downloads")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.version,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.version
    }


# Import and include API routes
from api.audio_routes import router as audio_router

app.include_router(audio_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )