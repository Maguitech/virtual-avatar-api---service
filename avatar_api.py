#!/usr/bin/env python3
"""
Avatar API - REST API for audio to avatar video conversion
"""

import os
import sys
import uuid
import shutil
import asyncio
import wave
import subprocess
from datetime import datetime
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
# from sse_starlette.sse import EventSourceResponse  # Install: pip install sse-starlette
import uvicorn
import json
import base64
from io import BytesIO
import cv2

# Import the lite avatar system
from lite_avatar import liteAvatar

# Configuration
UPLOAD_DIR = "./uploads"
OUTPUT_DIR = "./api_output"
TEMP_DIR = "./temp"
DATA_DIR = "./data/preload"

# Ensure directories exist
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Audio processing functions
def convert_to_wav(input_audio_path: str, output_wav_path: str) -> bool:
    """
    Convert any audio format to WAV (16kHz, mono, 16-bit) using ffmpeg
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', input_audio_path,
            '-ar', '16000',        # Sample rate 16kHz
            '-ac', '1',            # Mono channel
            '-sample_fmt', 's16',  # 16-bit PCM
            '-y',                  # Overwrite output file
            output_wav_path
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Audio conversion failed: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] Error in audio conversion: {e}")
        return False

def validate_wav_format(wav_path: str) -> bool:
    """
    Validate that WAV file has correct format (16kHz, mono, 16-bit)
    """
    try:
        with wave.open(wav_path, 'rb') as wav_file:
            params = wav_file.getparams()
            
            print(f"[INFO] Audio format: {params.nchannels} channels, "
                  f"{params.sampwidth*8}-bit, {params.framerate}Hz, "
                  f"{params.nframes} frames")
            
            # Check if format is correct
            is_valid = (
                params.nchannels == 1 and      # Mono
                params.sampwidth == 2 and      # 16-bit (2 bytes)
                params.framerate == 16000      # 16kHz
            )
            
            if not is_valid:
                print(f"[WARNING]  Audio format needs conversion")
            else:
                print(f"[SUCCESS] Audio format is correct")
                
            return is_valid
            
    except Exception as e:
        print(f"[ERROR] Error validating WAV: {e}")
        return False

def process_uploaded_audio(uploaded_file_path: str, job_id: str) -> str:
    """
    Process uploaded audio file and ensure it's in correct format
    Returns path to processed WAV file
    """
    file_ext = Path(uploaded_file_path).suffix.lower()
    processed_wav_path = os.path.join(TEMP_DIR, f"{job_id}_processed.wav")
    
    try:
        if file_ext == '.wav':
            # Check if WAV is in correct format
            if validate_wav_format(uploaded_file_path):
                # Already correct format, just copy
                shutil.copy2(uploaded_file_path, processed_wav_path)
                print("[COPY] WAV file copied (already correct format)")
                return processed_wav_path  # Skip redundant validation
            else:
                # Convert WAV to correct format
                print("[RELOAD] Converting WAV to correct format...")
                if not convert_to_wav(uploaded_file_path, processed_wav_path):
                    raise Exception("Failed to convert WAV to correct format")
                print("[SUCCESS] WAV converted to correct format")
        else:
            # Convert other formats (MP3, etc.) to WAV
            print(f"[RELOAD] Converting {file_ext} to WAV...")
            if not convert_to_wav(uploaded_file_path, processed_wav_path):
                raise Exception(f"Failed to convert {file_ext} to WAV")
            print("[SUCCESS] Audio converted to WAV format")
        
        # Only validate if we converted (ffmpeg should produce correct format)
        # Remove redundant final validation since ffmpeg guarantees format
        return processed_wav_path
        
    except Exception as e:
        print(f"[ERROR] Audio processing failed: {e}")
        # Cleanup failed file
        if os.path.exists(processed_wav_path):
            os.remove(processed_wav_path)
        raise e

class AvatarAPI:
    def __init__(self):
        self.avatar = None
        self.processing_jobs = {}  # Track processing jobs
        self.initialize_avatar()
    
    def initialize_avatar(self):
        """Initialize the avatar system"""
        try:
            self.avatar = liteAvatar(
                data_dir=DATA_DIR,
                num_threads=4,    # 4 threads for batch processing, or 16 for original processing
                generate_offline=True,
                use_gpu=True,     # ← GPU activada
                batch_processing=False  # ← True para batch, False para loop original
            )
        except Exception as e:
            print(f"[ERROR] Failed to initialize avatar: {e}")
            raise e
    
    def process_audio_to_video(self, audio_file_path: str, job_id: str):
        """Process audio file and generate avatar video"""
        processed_wav_path = None
        try:
            self.processing_jobs[job_id] = {
                "status": "processing",
                "start_time": datetime.now(),
                "progress": 10
            }
            
            # Create job-specific output directory
            job_output_dir = os.path.join(OUTPUT_DIR, job_id)
            os.makedirs(job_output_dir, exist_ok=True)
            
            # Process and convert audio to correct format
            print(f"[PROCESSING] Processing audio for job {job_id}...")
            processed_wav_path = process_uploaded_audio(audio_file_path, job_id)
            
            # Update progress
            self.processing_jobs[job_id]["progress"] = 30
            
            # Process the audio with avatar system
            print(f"[AVATAR] Generating avatar for job {job_id}...")
            self.avatar.handle(processed_wav_path, job_output_dir)
            
            # Check for generated video (no need to rename)
            final_video = os.path.join(job_output_dir, "test_demo.mp4")
            
            if os.path.exists(final_video):
                
                # Update job status
                self.processing_jobs[job_id] = {
                    "status": "completed",
                    "start_time": self.processing_jobs[job_id]["start_time"],
                    "end_time": datetime.now(),
                    "video_path": final_video,
                    "progress": 100
                }
                
                # Temporary frames already cleaned up by lite_avatar optimization
                
                # Clean up processed WAV file
                if processed_wav_path and os.path.exists(processed_wav_path):
                    os.remove(processed_wav_path)
                    
                print(f"[SUCCESS] Job {job_id} completed successfully")
                return final_video
            else:
                raise Exception("Video generation failed - no output file")
                
        except Exception as e:
            # Clean up on failure
            if processed_wav_path and os.path.exists(processed_wav_path):
                os.remove(processed_wav_path)
                
            self.processing_jobs[job_id] = {
                "status": "failed",
                "start_time": self.processing_jobs[job_id].get("start_time", datetime.now()),
                "error": str(e),
                "progress": 0
            }
            print(f"[ERROR] Job {job_id} failed: {e}")
            raise e
    
    def process_audio_to_stream(self, audio_file_path: str, job_id: str):
        """Stream frames as they are generated - Progressive Streaming"""
        processed_wav_path = None
        try:
            self.processing_jobs[job_id] = {
                "status": "streaming",
                "start_time": datetime.now(),
                "progress": 10
            }
            
            # Process and convert audio to correct format
            print(f"[STREAMING] Processing audio for job {job_id}...")
            processed_wav_path = process_uploaded_audio(audio_file_path, job_id)
            
            # Update progress
            self.processing_jobs[job_id]["progress"] = 30
            
            # Stream frames as they are generated
            print(f"[STREAMING] Starting progressive generation for job {job_id}...")
            yield from self.avatar.handle_stream(processed_wav_path, job_id)
            
            # Mark as completed
            self.processing_jobs[job_id]["status"] = "completed"
            self.processing_jobs[job_id]["end_time"] = datetime.now()
            self.processing_jobs[job_id]["progress"] = 100
            
            # Clean up processed WAV file
            if processed_wav_path and os.path.exists(processed_wav_path):
                os.remove(processed_wav_path)
                
            print(f"[SUCCESS] Streaming job {job_id} completed successfully")
                
        except Exception as e:
            # Clean up on failure
            if processed_wav_path and os.path.exists(processed_wav_path):
                os.remove(processed_wav_path)
                
            self.processing_jobs[job_id] = {
                "status": "failed",
                "start_time": self.processing_jobs[job_id].get("start_time", datetime.now()),
                "error": str(e),
                "progress": 0
            }
            print(f"[ERROR] Streaming job {job_id} failed: {e}")
            yield {"error": str(e)}

# Global avatar API instance
avatar_api = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize avatar system
    print("[STARTUP] Initializing Avatar System...")
    global avatar_api
    avatar_api = AvatarAPI()
    print("[STARTUP] Avatar System Ready!")
    yield
    # Shutdown: Clean up resources if needed
    print("[SHUTDOWN] Cleaning up Avatar System...")
    if avatar_api:
        # Add cleanup if needed
        pass

# Create FastAPI app with lifespan
app = FastAPI(
    title="Avatar API",
    description="Convert audio to avatar video",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Avatar API is running!",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate": "Upload audio and generate avatar video",
            "POST /generate-stream": "Upload audio and stream frames in real-time",
            "GET /status/{job_id}": "Check job status",
            "GET /download/{job_id}": "Download generated video",
            "GET /jobs": "List all jobs",
            "GET /stream-client": "Progressive streaming web client"
        }
    }

@app.get("/stream-client")
async def stream_client():
    """Serve the progressive streaming client"""
    with open("progressive_client.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.post("/generate")
async def generate_avatar(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    sync: bool = False
):
    """
    Generate avatar video from audio file
    
    - **audio_file**: Audio file (WAV, MP3, M4A, etc. - will be converted to 16kHz mono WAV)
    - **sync**: If True, wait for completion before returning (default: False)
    """
    
    # Get file extension
    file_ext = Path(audio_file.filename).suffix.lower() if audio_file.filename else ""
    
    # Validate file type - accept common audio formats
    valid_audio_extensions = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.wma'}
    valid_content_types = {
        'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/mpeg', 'audio/mp3',
        'audio/mp4', 'audio/m4a', 'audio/x-m4a',
        'audio/aac', 'audio/x-aac',
        'audio/flac', 'audio/x-flac',
        'audio/ogg', 'audio/vorbis',
        'audio/x-ms-wma'
    }
    
    # Check file extension or content type
    if file_ext not in valid_audio_extensions and not any(ct in audio_file.content_type for ct in valid_content_types):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported audio format. Supported: {', '.join(valid_audio_extensions)}"
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file
        audio_path = os.path.join(UPLOAD_DIR, f"{job_id}_{audio_file.filename}")
        
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        if sync:
            # Synchronous processing
            try:
                video_path = avatar_api.process_audio_to_video(audio_path, job_id)
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "message": "Avatar video generated successfully",
                    "download_url": f"/download/{job_id}"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        else:
            # Asynchronous processing
            background_tasks.add_task(
                avatar_api.process_audio_to_video,
                audio_path,
                job_id
            )
            
            return {
                "job_id": job_id,
                "status": "queued",
                "message": "Audio uploaded successfully. Processing started.",
                "status_url": f"/status/{job_id}",
                "download_url": f"/download/{job_id}"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    
    if job_id not in avatar_api.processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = avatar_api.processing_jobs[job_id].copy()
    
    # Calculate processing time
    if "end_time" in job_info:
        processing_time = (job_info["end_time"] - job_info["start_time"]).total_seconds()
        job_info["processing_time_seconds"] = processing_time
    
    # Remove internal paths for security
    if "video_path" in job_info:
        job_info["download_url"] = f"/download/{job_id}"
        del job_info["video_path"]
    
    return {
        "job_id": job_id,
        **job_info
    }

@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download the generated avatar video"""
    
    if job_id not in avatar_api.processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = avatar_api.processing_jobs[job_id]
    
    if job_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Status: {job_info['status']}"
        )
    
    if "video_path" not in job_info:
        raise HTTPException(status_code=404, detail="Video file not found")
    
    video_path = job_info["video_path"]
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"avatar_{job_id}.mp4",
        headers={"Cache-Control": "max-age=3600"}  # Cache for 1 hour
    )

@app.get("/jobs")
async def list_jobs():
    """List all processing jobs"""
    
    jobs = []
    for job_id, job_info in avatar_api.processing_jobs.items():
        job_summary = {
            "job_id": job_id,
            "status": job_info["status"],
            "start_time": job_info["start_time"]
        }
        
        if "end_time" in job_info:
            job_summary["end_time"] = job_info["end_time"]
            job_summary["processing_time_seconds"] = (
                job_info["end_time"] - job_info["start_time"]
            ).total_seconds()
        
        if job_info["status"] == "completed":
            job_summary["download_url"] = f"/download/{job_id}"
        
        jobs.append(job_summary)
    
    return {
        "total_jobs": len(jobs),
        "jobs": sorted(jobs, key=lambda x: x["start_time"], reverse=True)
    }

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files"""
    
    if job_id not in avatar_api.processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        # Remove job output directory
        job_output_dir = os.path.join(OUTPUT_DIR, job_id)
        if os.path.exists(job_output_dir):
            shutil.rmtree(job_output_dir)
        
        # Remove uploaded audio file
        for file in os.listdir(UPLOAD_DIR):
            if file.startswith(job_id):
                os.remove(os.path.join(UPLOAD_DIR, file))
        
        # Remove from tracking
        del avatar_api.processing_jobs[job_id]
        
        return {"message": f"Job {job_id} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.post("/generate-stream")
async def generate_avatar_stream(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...)
):
    """
    Generate avatar video with progressive streaming
    Returns frames as they are generated in real-time
    """
    
    # Get file extension
    file_ext = Path(audio_file.filename).suffix.lower() if audio_file.filename else ""
    
    # Validate file type
    valid_audio_extensions = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.wma'}
    valid_content_types = {
        'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/mpeg', 'audio/mp3',
        'audio/mp4', 'audio/m4a', 'audio/x-m4a',
        'audio/aac', 'audio/x-aac',
        'audio/flac', 'audio/x-flac',
        'audio/ogg', 'audio/vorbis',
        'audio/x-ms-wma'
    }
    
    if file_ext not in valid_audio_extensions and not any(ct in audio_file.content_type for ct in valid_content_types):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported audio format. Supported: {', '.join(valid_audio_extensions)}"
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file
        audio_path = os.path.join(UPLOAD_DIR, f"{job_id}_{audio_file.filename}")
        
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Stream generator function
        async def generate_stream():
            try:
                for frame_data in avatar_api.process_audio_to_stream(audio_path, job_id):
                    if "error" in frame_data:
                        yield f"data: {json.dumps(frame_data)}\n\n"
                        break
                    
                    # Convert frame to base64 for streaming
                    if "frame" in frame_data:
                        frame_img = frame_data["frame"]
                        _, buffer = cv2.imencode('.jpg', frame_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        stream_data = {
                            "frame_id": frame_data.get("frame_id", 0),
                            "frame_data": frame_b64,
                            "progress": frame_data.get("progress", 0),
                            "total_frames": frame_data.get("total_frames", 1),
                            "audio_chunk": frame_data.get("audio_chunk"),
                            "audio_chunk_duration": frame_data.get("audio_chunk_duration")
                        }
                        
                        yield f"data: {json.dumps(stream_data)}\n\n"
                    
                    # Handle completion
                    if frame_data.get("completed", False):
                        final_data = {
                            "completed": True,
                            "video_url": f"/download/{job_id}" if "video_path" in frame_data else None
                        }
                        yield f"data: {json.dumps(final_data)}\n\n"
                        
            except Exception as e:
                error_data = {"error": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Content-Type": "text/event-stream"})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "avatar_system": "ready" if avatar_api.avatar else "not_ready",
        "timestamp": datetime.now()
    }

def main():
    """Run the API server"""
    print("[START] Starting Avatar API Server...")
    print("[EMOJI] API Documentation: http://localhost:8000/docs")
    print("[EMOJI] Alternative docs: http://localhost:8000/redoc")
    
    uvicorn.run(
        "avatar_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()