#!/usr/bin/env python3
"""
Real-time Avatar WebSocket Server
Captures audio, transcribes to text, generates avatar video, and streams back
"""

import asyncio
import websockets
import json
import base64
import tempfile
import os
import uuid
import threading
import time
from datetime import datetime
from pathlib import Path
import wave
import io
from urllib.parse import urlparse

# Audio processing
import speech_recognition as sr
import requests
from pydub import AudioSegment
import subprocess

# Web server for webhook
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# WebSocket server
import websockets.server
from websockets.exceptions import ConnectionClosed

def convert_mp3_to_wav(input_mp3_path: str, output_wav_path: str) -> bool:
    """
    Convert MP3 to WAV (16kHz, mono, 16-bit) using ffmpeg - same as avatar_api.py
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', input_mp3_path,
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
        
        print(f"[SUCCESS] MP3 converted to WAV successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] MP3 conversion failed: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] Error in MP3 conversion: {e}")
        return False

class RealTimeAvatarSystem:
    def __init__(self, api_url="http://localhost:8000", n8n_webhook_url="https://automation.maguitech.com/webhook/avatar"):
        self.api_url = api_url
        self.n8n_webhook_url = n8n_webhook_url
        self.recognizer = sr.Recognizer()
        self.temp_dir = tempfile.mkdtemp(prefix="realtime_avatar_")
        self.active_connections = set()
        self.processing_queue = asyncio.Queue()
        
        # Store pending requests to match with webhook responses
        self.pending_requests = {}
        
        # Enhanced concurrency control
        self.processing_lock = asyncio.Lock()  # Prevents concurrent avatar processing
        self.active_jobs = set()  # Track active avatar generation jobs
        self.job_status_cache = {}  # Cache status calls to avoid spam
        self.max_concurrent_jobs = 1  # Only allow 1 concurrent avatar generation
        
        print(f"[INFO] Real-time Avatar System initialized")
        print(f"[INFO] API URL: {self.api_url}")
        print(f"[INFO] n8n Webhook URL: {self.n8n_webhook_url}")
        print(f"[INFO] Temp directory: {self.temp_dir}")
        print(f"[INFO] Max concurrent jobs: {self.max_concurrent_jobs}")
        
    async def cleanup_cache_periodically(self):
        """Clean up old cache entries every 5 minutes"""
        while True:
            await asyncio.sleep(300)  # 5 minutes
            if self.job_status_cache:
                print(f"[CACHE] Cleaning up {len(self.job_status_cache)} cache entries")
                self.job_status_cache.clear()
        
    async def handle_client(self, websocket):
        """Handle WebSocket client connection"""
        client_id = str(uuid.uuid4())[:8]
        self.active_connections.add(websocket)
        
        print(f"[WEBSOCKET] Client {client_id} connected from {websocket.remote_address}")
        
        try:
            await websocket.send(json.dumps({
                "type": "connected",
                "client_id": client_id,
                "message": "Connected to Real-time Avatar System"
            }))
            
            async for message in websocket:
                await self.process_message(websocket, client_id, message)
                
        except ConnectionClosed:
            print(f"[WEBSOCKET] Client {client_id} disconnected")
        except Exception as e:
            print(f"[ERROR] Error with client {client_id}: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": str(e)
            }))
        finally:
            self.active_connections.discard(websocket)
    
    async def process_message(self, websocket, client_id, message):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "audio_chunk":
                await self.handle_audio_chunk(websocket, client_id, data)
            elif message_type == "text_input":
                await self.handle_text_input(websocket, client_id, data)
            elif message_type == "ping":
                await websocket.send(json.dumps({"type": "pong"}))
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": f"Unknown message type: {message_type}"
                }))
                
        except json.JSONDecodeError as e:
            await websocket.send(json.dumps({
                "type": "error", 
                "error": f"Invalid JSON: {e}"
            }))
        except Exception as e:
            await websocket.send(json.dumps({
                "type": "error",
                "error": f"Processing error: {e}"
            }))
    
    async def handle_audio_chunk(self, websocket, client_id, data):
        """Handle audio chunk from client"""
        try:
            # Decode base64 audio data
            audio_data = base64.b64decode(data["audio_data"])
            chunk_id = data.get("chunk_id", str(uuid.uuid4()))
            
            print(f"[AUDIO] Received audio chunk from {client_id}: {len(audio_data)} bytes")
            
            # Send acknowledgment
            await websocket.send(json.dumps({
                "type": "audio_received",
                "chunk_id": chunk_id,
                "status": "processing"
            }))
            
            # Process audio asynchronously
            asyncio.create_task(
                self.process_audio_to_avatar(websocket, client_id, chunk_id, audio_data)
            )
            
        except Exception as e:
            print(f"[ERROR] Error handling audio chunk: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": f"Audio processing error: {e}"
            }))
    
    async def handle_text_input(self, websocket, client_id, data):
        """Handle direct text input from client"""
        try:
            text = data.get("text", "")
            if not text.strip():
                return
            
            print(f"[MESSAGE] Text from {client_id}: {text}")
            
            # Generate TTS audio and then avatar
            await self.generate_avatar_from_text(websocket, client_id, text)
            
        except Exception as e:
            print(f"[ERROR] Error handling text input: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": f"Text processing error: {e}"
            }))
    
    async def process_audio_to_avatar(self, websocket, client_id, chunk_id, audio_data):
        """Process audio chunk: transcribe → generate avatar"""
        try:
            # Save audio chunk to temporary file
            audio_file = os.path.join(self.temp_dir, f"{client_id}_{chunk_id}.wav")
            
            # Convert audio data to WAV if needed
            audio_segment = AudioSegment.from_raw(
                io.BytesIO(audio_data),
                sample_width=2,  # 16-bit
                frame_rate=16000,  # 16kHz
                channels=1  # Mono
            )
            audio_segment.export(audio_file, format="wav")
            
            # Transcribe audio to text
            transcribed_text = await self.transcribe_audio(audio_file)
            
            if transcribed_text:
                print(f"[TEXT] Transcribed: {transcribed_text}")
                
                # Send transcription to client
                await websocket.send(json.dumps({
                    "type": "transcription",
                    "chunk_id": chunk_id,
                    "text": transcribed_text
                }))
                
                # Generate avatar video
                video_url = await self.generate_avatar_video(audio_file)
                
                if video_url:
                    # Send video URL to client
                    await websocket.send(json.dumps({
                        "type": "avatar_video",
                        "chunk_id": chunk_id,
                        "video_url": video_url,
                        "text": transcribed_text
                    }))
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "chunk_id": chunk_id,
                        "error": "Failed to generate avatar video"
                    }))
            else:
                await websocket.send(json.dumps({
                    "type": "transcription",
                    "chunk_id": chunk_id,
                    "text": "",
                    "message": "No speech detected"
                }))
            
            # Clean up
            if os.path.exists(audio_file):
                os.remove(audio_file)
                
        except Exception as e:
            print(f"[ERROR] Error processing audio to avatar: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "chunk_id": chunk_id,
                "error": str(e)
            }))
    
    async def transcribe_audio(self, audio_file):
        """Transcribe audio file to text using speech_recognition"""
        try:
            # Use threading to avoid blocking
            loop = asyncio.get_event_loop()
            
            def transcribe():
                try:
                    with sr.AudioFile(audio_file) as source:
                        audio = self.recognizer.record(source)
                    
                    # Try multiple recognition engines
                    try:
                        # Google Speech Recognition (free tier)
                        text = self.recognizer.recognize_google(audio, language='es-ES')
                        return text
                    except sr.UnknownValueError:
                        return None
                    except sr.RequestError:
                        # Fallback to offline recognition if available
                        try:
                            text = self.recognizer.recognize_sphinx(audio, language='es-ES')
                            return text
                        except:
                            return None
                except Exception as e:
                    print(f"[ERROR] Transcription error: {e}")
                    return None
            
            # Run transcription in thread pool
            text = await loop.run_in_executor(None, transcribe)
            return text
            
        except Exception as e:
            print(f"[ERROR] Error in transcribe_audio: {e}")
            return None
    
    async def generate_avatar_video(self, audio_file):
        """Generate avatar video using the API"""
        try:
            print(f"[API] Sending request to Avatar API: {self.api_url}/generate")
            print(f"[API] Audio file: {audio_file}")
            
            # Send request to avatar API
            with open(audio_file, 'rb') as f:
                files = {'audio_file': f}
                data = {'sync': False}  # Asynchronous processing
                
                print(f"[API] Making POST request...")
                response = requests.post(
                    f"{self.api_url}/generate",
                    files=files,
                    data=data,
                    timeout=30
                )
                print(f"[API] Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                job_id = result['job_id']
                print(f"[API] Job created: {job_id}")
                
                # Wait for completion
                print(f"[API] Waiting for job completion...")
                video_path = await self.wait_for_avatar_completion(job_id)
                print(f"[API] Job completed, video URL: {video_path}")
                return video_path
            else:
                print(f"[ERROR] API request failed: {response.status_code}")
                print(f"[ERROR] Response text: {response.text}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Error generating avatar video: {e}")
            return None
    
    async def wait_for_avatar_completion(self, job_id, max_wait=120):
        """Wait for avatar generation to complete with smart caching"""
        try:
            start_time = time.time()
            check_count = 0
            last_status = None
            
            while time.time() - start_time < max_wait:
                check_count += 1
                
                # Check cache first to avoid redundant calls
                cache_key = f"{job_id}_{check_count}"
                if cache_key in self.job_status_cache:
                    status = self.job_status_cache[cache_key]
                    print(f"[API] Status check #{check_count} (cached): {status['status']}")
                else:
                    response = requests.get(f"{self.api_url}/status/{job_id}")
                    
                    if response.status_code == 200:
                        status = response.json()
                        # Cache the status for a short time
                        self.job_status_cache[cache_key] = status
                        print(f"[API] Status check #{check_count}: {status['status']} (progress: {status.get('progress', 'unknown')}%)")
                    else:
                        print(f"[API] Status check #{check_count} failed: HTTP {response.status_code}")
                        await asyncio.sleep(2)
                        continue
                
                if status['status'] == 'completed':
                    # Clean up cache for this job
                    keys_to_remove = [k for k in self.job_status_cache.keys() if k.startswith(f"{job_id}_")]
                    for key in keys_to_remove:
                        del self.job_status_cache[key]
                    return f"{self.api_url}/download/{job_id}"
                elif status['status'] == 'failed':
                    print(f"[ERROR] Avatar generation failed: {status.get('error')}")
                    # Clean up cache for this job
                    keys_to_remove = [k for k in self.job_status_cache.keys() if k.startswith(f"{job_id}_")]
                    for key in keys_to_remove:
                        del self.job_status_cache[key]
                    return None
                
                # Only log status changes to reduce spam
                if last_status != status['status']:
                    print(f"[API] Job {job_id} status changed: {last_status} -> {status['status']}")
                    last_status = status['status']
                
                # Progressive wait times with longer delays to reduce API spam
                if check_count < 3:
                    await asyncio.sleep(2)  # First 3 checks every 2 seconds
                elif check_count < 8:
                    await asyncio.sleep(4)  # Next 5 checks every 4 seconds
                else:
                    await asyncio.sleep(6)  # Rest every 6 seconds
            
            print(f"[ERROR] Avatar generation timed out for job {job_id} after {check_count} checks")
            # Clean up cache on timeout
            keys_to_remove = [k for k in self.job_status_cache.keys() if k.startswith(f"{job_id}_")]
            for key in keys_to_remove:
                del self.job_status_cache[key]
            return None
            
        except Exception as e:
            print(f"[ERROR] Error waiting for completion: {e}")
            return None
    
    async def generate_avatar_from_text(self, websocket, client_id, text):
        """Generate avatar from text using n8n workflow"""
        # Check if we're already at max concurrent jobs
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            await websocket.send(json.dumps({
                "type": "error", 
                "error": "Sistema ocupado procesando otra solicitud. Intente de nuevo en un momento."
            }))
            print(f"[QUEUE] Request rejected - already processing {len(self.active_jobs)} jobs")
            return
            
        try:
            async with self.processing_lock:
                print(f"[n8n] Processing text via n8n (with lock): {text}")
                print(f"[QUEUE] Active jobs: {len(self.active_jobs)}")
                
                # Send processing message
                await websocket.send(json.dumps({
                    "type": "text_processing",
                    "text": text,
                    "message": "Enviando texto a agente IA..."
                }))
                
                # Generate unique request ID
                request_id = str(uuid.uuid4())
                
                # Add to active jobs tracking
                self.active_jobs.add(request_id)
                print(f"[QUEUE] Added job {request_id} to active jobs (total: {len(self.active_jobs)})")
                
                # Store request info to match with webhook response
                self.pending_requests[request_id] = {
                    "websocket": websocket,
                    "client_id": client_id,
                    "original_text": text,
                    "timestamp": time.time()
                }
                
                # Prepare payload for n8n webhook
                n8n_payload = {
                    "request_id": request_id,
                    "text": text,
                    "client_id": client_id
                }
                
                # Also send request_id in headers to make sure n8n gets it
                n8n_headers = {
                    "Content-Type": "application/json",
                    "X-Request-ID": request_id
                }
                
                print(f"[n8n] Sending to webhook: {self.n8n_webhook_url}")
                print(f"[n8n] Payload: {n8n_payload}")
                
                # Send to n8n webhook and get audio response directly
                try:
                    response = requests.post(
                        self.n8n_webhook_url,
                        json=n8n_payload,
                        timeout=60,  # Increased timeout for AI processing
                        headers=n8n_headers
                    )
                    
                    if response.status_code == 200:
                        print(f"[n8n] Successfully received response from n8n")
                        print(f"[n8n] Response content-type: {response.headers.get('content-type', 'unknown')}")
                        print(f"[n8n] Response size: {len(response.content)} bytes")
                        print(f"[n8n] Response headers: {dict(response.headers)}")
                        
                        # Check if response is JSON (base64) or binary audio
                        content_type = response.headers.get('content-type', '')
                        
                        if 'application/json' in content_type:
                            # n8n is sending JSON with base64 audio
                            print(f"[n8n] Processing JSON response with base64 audio")
                            try:
                                json_data = response.json()
                                print(f"[n8n] JSON response: {type(json_data)} - {str(json_data)[:200]}...")
                                
                                # Handle array format: [{"data":"base64_file"}]
                                if isinstance(json_data, list) and len(json_data) > 0:
                                    audio_base64 = json_data[0].get('data')
                                    if not audio_base64:
                                        print(f"[ERROR] No 'data' field in first array element")
                                        raise Exception("Missing 'data' field in JSON array response")
                                    print(f"[n8n] Found base64 data in array format")
                                else:
                                    # Handle object format: {"data":"base64_file"} or {"audio_data":"base64_file"}
                                    audio_base64 = json_data.get('data') or json_data.get('audio_data')
                                    if not audio_base64:
                                        print(f"[ERROR] No 'data' or 'audio_data' field in JSON response")
                                        print(f"[ERROR] Available keys: {list(json_data.keys()) if isinstance(json_data, dict) else 'Not a dict'}")
                                        raise Exception("Missing audio data in JSON response")
                                    print(f"[n8n] Found base64 data in object format")
                                
                                # Decode base64 to binary
                                import base64
                                audio_data = base64.b64decode(audio_base64)
                                print(f"[n8n] Decoded base64 audio: {len(audio_data)} bytes")
                                
                                # Get AI response text from JSON (if available)
                                if isinstance(json_data, list):
                                    ai_response_text = json_data[0].get('response_text', 'Respuesta del agente IA')
                                else:
                                    ai_response_text = json_data.get('response_text', 'Respuesta del agente IA')
                                
                            except Exception as e:
                                print(f"[ERROR] Failed to parse JSON response: {e}")
                                raise Exception(f"Invalid JSON response from n8n: {e}")
                        else:
                            # n8n is sending binary audio directly
                            print(f"[n8n] Processing binary audio response")
                            audio_data = response.content
                            ai_response_text = response.headers.get('ai-response-text', 'Respuesta del agente IA')
                        
                        # Basic validation - check if we got audio data
                        if len(audio_data) < 1000:  # Too small to be valid audio
                            print(f"[ERROR] Audio data too small: {len(audio_data)} bytes")
                            raise Exception("Audio data from n8n is too small to be valid")
                        
                        print(f"[n8n] Final audio data size: {len(audio_data)} bytes")
                        
                        # Process the audio directly
                        success = await self.handle_n8n_response(request_id, audio_data, ai_response_text)
                        
                        if not success:
                            raise Exception("Failed to process n8n audio response")
                            
                    else:
                        print(f"[n8n] Webhook failed: {response.status_code} - {response.text}")
                        raise Exception(f"n8n webhook failed: {response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    print(f"[n8n] Error sending to webhook: {e}")
                    raise Exception(f"Failed to send to n8n: {e}")
                    
        except Exception as e:
            print(f"[ERROR] Error in n8n text processing: {e}")
            
            # Clean up pending request and active job
            if 'request_id' in locals():
                if request_id in self.pending_requests:
                    del self.pending_requests[request_id]
                if request_id in self.active_jobs:
                    self.active_jobs.remove(request_id)
                    print(f"[QUEUE] Removed failed job {request_id} from active jobs")
            
            await websocket.send(json.dumps({
                "type": "error",
                "error": f"n8n processing error: {e}"
            }))
    
    async def handle_n8n_response(self, request_id: str, audio_data: bytes, ai_response_text: str):
        """Handle response from n8n webhook with audio data"""
        try:
            if request_id not in self.pending_requests:
                print(f"[n8n] No pending request found for ID: {request_id}")
                return False
                
            request_info = self.pending_requests[request_id]
            websocket = request_info["websocket"]
            client_id = request_info["client_id"]
            original_text = request_info["original_text"]
            
            print(f"[n8n] Received audio response for: {original_text}")
            print(f"[n8n] AI Response: {ai_response_text}")
            print(f"[n8n] Audio size: {len(audio_data)} bytes")
            
            # Save audio to temporary file (initially as MP3 from n8n)
            temp_mp3_file = os.path.join(self.temp_dir, f"{client_id}_{request_id}.mp3")
            temp_wav_file = os.path.join(self.temp_dir, f"{client_id}_{request_id}.wav")
            
            with open(temp_mp3_file, 'wb') as f:
                f.write(audio_data)
                
            print(f"[n8n] Saved MP3 audio to: {temp_mp3_file}")
            
            # Convert MP3 to WAV using the same method as avatar_api.py
            print(f"[n8n] Converting MP3 to WAV using ffmpeg...")
            if not convert_mp3_to_wav(temp_mp3_file, temp_wav_file):
                # Cleanup and fail
                if os.path.exists(temp_mp3_file):
                    os.remove(temp_mp3_file)
                raise Exception("Failed to convert MP3 to WAV")
            
            # Send status update to client
            await websocket.send(json.dumps({
                "type": "text_processing",
                "text": ai_response_text,
                "message": "Audio recibido de IA, generando avatar..."
            }))
            
            # Generate avatar video using the converted WAV audio
            print(f"[API] Calling Avatar API with WAV file: {temp_wav_file}")
            print(f"[API] WAV file exists: {os.path.exists(temp_wav_file)}")
            if os.path.exists(temp_wav_file):
                print(f"[API] WAV file size: {os.path.getsize(temp_wav_file)} bytes")
            
            video_url = await self.generate_avatar_video(temp_wav_file)
            print(f"[API] Avatar API returned: {video_url}")
            
            if video_url:
                # Send video URL to client
                await websocket.send(json.dumps({
                    "type": "avatar_video",
                    "video_url": video_url,
                    "text": ai_response_text,
                    "source": "n8n_ai_response"
                }))
                print(f"[SUCCESS] Avatar video generated from n8n response")
            else:
                print(f"[ERROR] Avatar API returned None or empty URL")
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "Failed to generate avatar video from n8n audio"
                }))
            
            # Clean up temporary files
            for temp_file in [temp_mp3_file, temp_wav_file]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"[CLEANUP] Removed: {temp_file}")
                
            # Remove from pending requests and active jobs
            del self.pending_requests[request_id]
            if request_id in self.active_jobs:
                self.active_jobs.remove(request_id)
                print(f"[QUEUE] Completed job {request_id}, removed from active jobs (remaining: {len(self.active_jobs)})")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error handling n8n response: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up on error
            if request_id in self.pending_requests:
                websocket = self.pending_requests[request_id]["websocket"]
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": f"Error processing n8n response: {e}"
                }))
                del self.pending_requests[request_id]
            
            # Also remove from active jobs
            if request_id in self.active_jobs:
                self.active_jobs.remove(request_id)
                print(f"[QUEUE] Removed failed job {request_id} from active jobs on error")
            
            return False

    async def start_server(self, host="localhost", port=8765):
        """Start the WebSocket server"""
        print(f"[WEBSOCKET] Starting WebSocket server on ws://{host}:{port}")
        
        # Start background cache cleanup task
        asyncio.create_task(self.cleanup_cache_periodically())
        
        async with websockets.serve(self.handle_client, host, port):
            print(f"[SUCCESS] WebSocket server running on ws://{host}:{port}")
            await asyncio.Future()  # Run forever

# Global system instance for webhook access
avatar_system = None

def create_webhook_app():
    """Create FastAPI app for handling n8n webhook responses"""
    app = FastAPI(title="Avatar Webhook Server", description="Receives responses from n8n")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.post("/webhook/response")
    async def handle_webhook_response(request: Request):
        """Handle webhook response from n8n with audio file"""
        try:
            content_type = request.headers.get("content-type", "")
            print(f"[WEBHOOK] Received request with content-type: {content_type}")
            
            request_id = None
            ai_response_text = ""
            audio_data = None
            
            # Handle multipart form data (file upload)
            if "multipart/form-data" in content_type:
                from fastapi import Form, File, UploadFile
                # For multipart, we need to handle it differently
                form_data = await request.form()
                
                request_id = form_data.get("request_id")
                ai_response_text = form_data.get("response_text", "Respuesta del agente IA")
                
                # Get audio file
                audio_file = None
                for key, value in form_data.items():
                    if hasattr(value, 'read'):  # It's a file
                        audio_file = value
                        break
                
                if not audio_file:
                    raise HTTPException(status_code=400, detail="No audio file found in form data")
                
                audio_data = await audio_file.read()
                
            # Handle JSON data  
            elif "application/json" in content_type:
                data = await request.json()
                request_id = data.get("request_id")
                ai_response_text = data.get("response_text", "Respuesta del agente IA")
                audio_base64 = data.get("audio_data")
                
                if audio_base64:
                    try:
                        audio_data = base64.b64decode(audio_base64)
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {e}")
                else:
                    raise HTTPException(status_code=400, detail="Missing audio_data in JSON")
            
            # Handle raw audio data
            else:
                # Try to get metadata from headers or query params
                query_params = request.query_params
                headers_dict = dict(request.headers)
                
                # Check for request_id in headers (n8n sends it this way)
                request_id = (
                    headers_dict.get("request-id") or  # n8n header format
                    headers_dict.get("request_id") or  # alternative format
                    query_params.get("request_id")     # query param fallback
                )
                
                ai_response_text = (
                    headers_dict.get("response-text") or
                    query_params.get("response_text") or 
                    "Respuesta del agente IA"
                )
                
                # Get raw body as audio
                audio_data = await request.body()
                
                if not audio_data:
                    raise HTTPException(status_code=400, detail="No audio data received")
            
            print(f"[WEBHOOK] Processing request:")
            print(f"  - Request ID: {request_id}")
            print(f"  - AI Text: {ai_response_text}")
            print(f"  - Audio size: {len(audio_data) if audio_data else 0} bytes")
            print(f"  - Content Type: {content_type}")
            print(f"  - Headers: {dict(request.headers)}")
            print(f"  - Query Params: {dict(request.query_params)}")
            
            if avatar_system:
                print(f"  - Pending requests: {list(avatar_system.pending_requests.keys())}")
            else:
                print(f"  - Avatar system: None")
            
            if not request_id:
                # If no request_id provided, try to match with latest pending request
                if avatar_system and avatar_system.pending_requests:
                    request_id = list(avatar_system.pending_requests.keys())[-1]
                    print(f"[WEBHOOK] No request_id provided, using latest: {request_id}")
                else:
                    raise HTTPException(status_code=400, detail="Missing request_id and no pending requests")
            
            if not audio_data:
                raise HTTPException(status_code=400, detail="No audio data received")
            
            # Process the response
            if avatar_system:
                success = await avatar_system.handle_n8n_response(request_id, audio_data, ai_response_text)
                if success:
                    return {"status": "success", "message": "Audio processed and avatar generated"}
                else:
                    return {"status": "error", "message": "Failed to process audio"}
            else:
                raise HTTPException(status_code=503, detail="Avatar system not available")
                
        except HTTPException:
            raise
        except Exception as e:
            print(f"[WEBHOOK] Error processing webhook: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    
    @app.get("/webhook/health")
    async def webhook_health():
        """Health check for webhook server"""
        return {
            "status": "healthy",
            "system_ready": avatar_system is not None,
            "pending_requests": len(avatar_system.pending_requests) if avatar_system else 0,
            "pending_request_ids": list(avatar_system.pending_requests.keys()) if avatar_system else []
        }
    
    @app.post("/webhook/debug")
    async def debug_webhook(request: Request):
        """Debug endpoint to see what n8n is sending"""
        try:
            content_type = request.headers.get("content-type", "")
            headers = dict(request.headers)
            query_params = dict(request.query_params)
            
            print(f"[DEBUG] Received debug request:")
            print(f"  Content-Type: {content_type}")
            print(f"  Headers: {headers}")
            print(f"  Query Params: {query_params}")
            
            # Try to read body
            body = await request.body()
            print(f"  Body size: {len(body)} bytes")
            print(f"  Body type: {type(body)}")
            
            # Try different parsing methods
            response_data = {
                "content_type": content_type,
                "headers": headers,
                "query_params": query_params,
                "body_size": len(body),
                "body_preview": str(body[:200]) + "..." if len(body) > 200 else str(body)
            }
            
            if "multipart/form-data" in content_type:
                try:
                    # Reset request for form parsing
                    request._body = body  # This might not work, but let's try
                    form_data = await request.form()
                    response_data["form_fields"] = list(form_data.keys())
                    response_data["form_data"] = {k: str(v)[:100] for k, v in form_data.items()}
                except Exception as e:
                    response_data["form_parse_error"] = str(e)
            
            elif "application/json" in content_type:
                try:
                    json_data = json.loads(body)
                    response_data["json_data"] = json_data
                except Exception as e:
                    response_data["json_parse_error"] = str(e)
            
            return response_data
            
        except Exception as e:
            print(f"[DEBUG] Error in debug endpoint: {e}")
            return {"error": str(e)}
    
    return app

def run_webhook_server():
    """Run the webhook server in a separate thread"""
    app = create_webhook_app()
    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="info")

def main():
    """Main function"""
    import argparse
    import threading
    
    parser = argparse.ArgumentParser(description="Real-time Avatar WebSocket Server with n8n Integration")
    parser.add_argument("--host", default="localhost", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Avatar API URL")
    parser.add_argument("--n8n-webhook", default="https://automation.maguitech.com/webhook/avatar", help="n8n webhook URL")
    parser.add_argument("--enable-webhook-debug", action="store_true", help="Enable webhook debug server")
    
    args = parser.parse_args()
    
    global avatar_system
    # Create and start the real-time system
    avatar_system = RealTimeAvatarSystem(api_url=args.api_url, n8n_webhook_url=args.n8n_webhook)
    
    # Optionally start webhook server for debugging
    if args.enable_webhook_debug:
        print("[WEBHOOK] Starting webhook debug server on port 8766...")
        webhook_thread = threading.Thread(target=run_webhook_server, daemon=True)
        webhook_thread.start()
        time.sleep(2)  # Give webhook server time to start
        print(f"[INFO] Webhook debug endpoint: http://localhost:8766/webhook/debug")
        print(f"[INFO] Webhook health: http://localhost:8766/webhook/health")
    else:
        print("[INFO] n8n responses will be processed directly (no separate webhook server)")
    
    print(f"[INFO] n8n will be called at: {args.n8n_webhook}")
    print(f"[INFO] Avatar API: {args.api_url}")
    
    try:
        asyncio.run(avatar_system.start_server(args.host, args.port))
    except KeyboardInterrupt:
        print("\n[GOODBYE] Shutting down...")

if __name__ == "__main__":
    main()