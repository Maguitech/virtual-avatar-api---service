#!/usr/bin/env python3
"""
Enhanced Voice Chat Server with Improved Avatar Management
- Seamless video transitions
- Better audio processing
- Improved n8n integration
- WebRTC-like experience without WebRTC complexity
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

# Web server for enhanced features
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn

# WebSocket server
import websockets.server
from websockets.exceptions import ConnectionClosed

class EnhancedVoiceChatServer:
    def __init__(self, 
                 api_url="http://localhost:8000", 
                 n8n_webhook_url="https://automation.maguitech.com/webhook/avatar"):
        self.api_url = api_url
        self.n8n_webhook_url = n8n_webhook_url
        self.recognizer = sr.Recognizer()
        self.temp_dir = tempfile.mkdtemp(prefix="enhanced_voice_chat_")
        
        # Connection management
        self.active_connections = {}  # client_id -> connection_info
        self.client_sessions = {}     # client_id -> session_data
        
        # Enhanced processing queue
        self.processing_queue = asyncio.Queue()
        self.active_jobs = {}  # job_id -> job_info
        self.job_status_cache = {}
        
        # Video management
        self.video_cache = {}  # video_url -> local_path
        self.background_video_url = f"{self.api_url}/data/preload/bg_video_h264.mp4"
        
        # Configuration
        self.max_concurrent_jobs_per_client = 1
        self.max_audio_chunk_size = 1024 * 1024  # 1MB
        self.transcription_timeout = 10  # seconds
        self.avatar_generation_timeout = 120  # seconds
        
        print(f"[INIT] Enhanced Voice Chat Server initialized")
        print(f"[INIT] API URL: {self.api_url}")
        print(f"[INIT] n8n Webhook: {self.n8n_webhook_url}")
        print(f"[INIT] Temp directory: {self.temp_dir}")
        
    async def handle_client(self, websocket):
        """Handle WebSocket client with enhanced session management"""
        client_id = str(uuid.uuid4())[:8]
        client_info = {
            "websocket": websocket,
            "connected_at": datetime.now(),
            "last_activity": datetime.now(),
            "active_jobs": set(),
            "session_data": {
                "conversation_history": [],
                "preferences": {},
                "current_avatar_video": None
            }
        }
        
        self.active_connections[client_id] = client_info
        
        try:
            print(f"[CLIENT] {client_id} connected from {websocket.remote_address}")
            
            # Send enhanced connection confirmation
            await websocket.send(json.dumps({
                "type": "connected",
                "client_id": client_id,
                "message": "Conectado al asistente de voz avanzado",
                "server_capabilities": {
                    "audio_transcription": True,
                    "avatar_generation": True,
                    "n8n_integration": True,
                    "video_streaming": True,
                    "real_time_processing": True
                },
                "session_info": {
                    "background_video": self.background_video_url,
                    "max_audio_chunk_size": self.max_audio_chunk_size,
                    "supported_audio_formats": ["webm", "wav", "mp3"]
                }
            }))
            
            async for message in websocket:
                await self.process_client_message(client_id, message)
                
        except ConnectionClosed:
            print(f"[CLIENT] {client_id} disconnected")
        except Exception as e:
            print(f"[ERROR] Client {client_id} error: {e}")
            await self.send_error(client_id, f"Connection error: {e}")
        finally:
            await self.cleanup_client(client_id)
    
    async def process_client_message(self, client_id, message):
        """Process incoming message with enhanced routing"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            # Update last activity
            if client_id in self.active_connections:
                self.active_connections[client_id]["last_activity"] = datetime.now()
            
            if message_type == "audio_chunk":
                await self.handle_audio_chunk(client_id, data)
            elif message_type == "text_input":
                await self.handle_text_input(client_id, data)
            elif message_type == "voice_command":
                await self.handle_voice_command(client_id, data)
            elif message_type == "video_ready":
                await self.handle_video_ready(client_id, data)
            elif message_type == "heartbeat":
                await self.send_message(client_id, {"type": "heartbeat_ack"})
            elif message_type == "session_update":
                await self.handle_session_update(client_id, data)
            else:
                await self.send_error(client_id, f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError as e:
            await self.send_error(client_id, f"Invalid JSON: {e}")
        except Exception as e:
            await self.send_error(client_id, f"Message processing error: {e}")
    
    async def handle_audio_chunk(self, client_id, data):
        """Enhanced audio chunk handling with better validation"""
        try:
            # Validate client connection
            if client_id not in self.active_connections:
                return
            
            # Validate audio data
            if "audio_data" not in data:
                await self.send_error(client_id, "Missing audio_data")
                return
            
            audio_b64 = data["audio_data"]
            chunk_id = data.get("chunk_id", str(uuid.uuid4()))
            
            # Decode and validate audio
            try:
                audio_data = base64.b64decode(audio_b64)
            except Exception as e:
                await self.send_error(client_id, f"Invalid base64 audio: {e}")
                return
            
            if len(audio_data) > self.max_audio_chunk_size:
                await self.send_error(client_id, "Audio chunk too large")
                return
            
            if len(audio_data) < 100:  # Too small to be meaningful
                await self.send_message(client_id, {
                    "type": "audio_received",
                    "chunk_id": chunk_id,
                    "status": "too_small"
                })
                return
            
            print(f"[AUDIO] Client {client_id} sent {len(audio_data)} bytes")
            
            # Check if client has too many active jobs
            client_info = self.active_connections[client_id]
            if len(client_info["active_jobs"]) >= self.max_concurrent_jobs_per_client:
                await self.send_message(client_id, {
                    "type": "audio_received",
                    "chunk_id": chunk_id,
                    "status": "queue_full",
                    "message": "Por favor espera a que termine la respuesta anterior"
                })
                return
            
            # Acknowledge receipt
            await self.send_message(client_id, {
                "type": "audio_received",
                "chunk_id": chunk_id,
                "status": "processing",
                "message": "Procesando audio..."
            })
            
            # Process asynchronously
            job_id = f"{client_id}_{chunk_id}"
            client_info["active_jobs"].add(job_id)
            
            asyncio.create_task(
                self.process_audio_to_avatar(client_id, job_id, chunk_id, audio_data)
            )
            
        except Exception as e:
            print(f"[ERROR] Audio chunk error: {e}")
            await self.send_error(client_id, f"Audio processing error: {e}")
    
    async def handle_text_input(self, client_id, data):
        """Enhanced text input handling with conversation context"""
        try:
            text = data.get("text", "").strip()
            if not text:
                return
            
            if client_id not in self.active_connections:
                return
            
            client_info = self.active_connections[client_id]
            
            # Check job limit
            if len(client_info["active_jobs"]) >= self.max_concurrent_jobs_per_client:
                await self.send_message(client_id, {
                    "type": "text_processing",
                    "status": "queue_full",
                    "message": "Por favor espera a que termine la respuesta anterior"
                })
                return
            
            print(f"[TEXT] Client {client_id}: {text}")
            
            # Add to conversation history
            client_info["session_data"]["conversation_history"].append({
                "type": "user",
                "text": text,
                "timestamp": datetime.now().isoformat()
            })
            
            # Process with n8n
            job_id = f"{client_id}_text_{uuid.uuid4().hex[:8]}"
            client_info["active_jobs"].add(job_id)
            
            asyncio.create_task(
                self.process_text_with_n8n(client_id, job_id, text)
            )
            
        except Exception as e:
            print(f"[ERROR] Text input error: {e}")
            await self.send_error(client_id, f"Text processing error: {e}")
    
    async def process_audio_to_avatar(self, client_id, job_id, chunk_id, audio_data):
        """Enhanced audio processing with better error handling"""
        try:
            print(f"[JOB] Starting audio processing job {job_id}")
            
            # Save audio to temp file
            audio_file = os.path.join(self.temp_dir, f"{job_id}.wav")
            
            # Convert audio data to proper WAV format
            try:
                audio_segment = AudioSegment.from_raw(
                    io.BytesIO(audio_data),
                    sample_width=2,
                    frame_rate=16000,
                    channels=1
                )
                audio_segment.export(audio_file, format="wav")
                print(f"[AUDIO] Saved audio file: {audio_file}")
            except Exception as e:
                print(f"[ERROR] Audio conversion failed: {e}")
                await self.send_error(client_id, "Error converting audio format")
                return
            
            # Update client
            await self.send_message(client_id, {
                "type": "processing_status",
                "job_id": job_id,
                "stage": "transcribing",
                "message": "Transcribiendo audio..."
            })
            
            # Transcribe audio
            transcribed_text = await self.transcribe_audio_enhanced(audio_file)
            
            if transcribed_text:
                print(f"[TRANSCRIPT] {job_id}: {transcribed_text}")
                
                # Send transcription
                await self.send_message(client_id, {
                    "type": "transcription",
                    "job_id": job_id,
                    "chunk_id": chunk_id,
                    "text": transcribed_text
                })
                
                # Add to conversation history
                if client_id in self.active_connections:
                    self.active_connections[client_id]["session_data"]["conversation_history"].append({
                        "type": "user",
                        "text": transcribed_text,
                        "source": "voice",
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Generate avatar video
                await self.send_message(client_id, {
                    "type": "processing_status",
                    "job_id": job_id,
                    "stage": "generating_avatar",
                    "message": "Generando respuesta del avatar..."
                })
                
                video_url = await self.generate_avatar_video_enhanced(audio_file, job_id)
                
                if video_url:
                    await self.send_message(client_id, {
                        "type": "avatar_video",
                        "job_id": job_id,
                        "chunk_id": chunk_id,
                        "video_url": video_url,
                        "text": transcribed_text,
                        "source": "voice_transcription",
                        "transition": {
                            "type": "smooth_overlay",
                            "duration": 0.5
                        }
                    })
                    
                    # Update session
                    if client_id in self.active_connections:
                        self.active_connections[client_id]["session_data"]["current_avatar_video"] = video_url
                    
                else:
                    await self.send_error(client_id, "Error generando video del avatar")
            else:
                await self.send_message(client_id, {
                    "type": "transcription",
                    "job_id": job_id,
                    "chunk_id": chunk_id,
                    "text": "",
                    "message": "No se detectó voz clara en el audio"
                })
            
            # Cleanup
            if os.path.exists(audio_file):
                os.remove(audio_file)
                
        except Exception as e:
            print(f"[ERROR] Audio processing job {job_id} failed: {e}")
            await self.send_error(client_id, f"Audio processing failed: {e}")
        finally:
            # Remove job from active jobs
            if client_id in self.active_connections:
                self.active_connections[client_id]["active_jobs"].discard(job_id)
            print(f"[JOB] Completed audio processing job {job_id}")
    
    async def process_text_with_n8n(self, client_id, job_id, text):
        """Enhanced n8n integration with better error handling"""
        try:
            print(f"[N8N] Starting text processing job {job_id}")
            
            await self.send_message(client_id, {
                "type": "processing_status",
                "job_id": job_id,
                "stage": "ai_thinking",
                "message": "Enviando a la IA..."
            })
            
            # Prepare enhanced payload with context
            conversation_history = []
            if client_id in self.active_connections:
                conversation_history = self.active_connections[client_id]["session_data"]["conversation_history"][-5:]  # Last 5 messages
            
            n8n_payload = {
                "request_id": job_id,
                "text": text,
                "client_id": client_id,
                "context": {
                    "conversation_history": conversation_history,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": client_id
                }
            }
            
            n8n_headers = {
                "Content-Type": "application/json",
                "X-Request-ID": job_id,
                "X-Client-ID": client_id
            }
            
            print(f"[N8N] Sending to: {self.n8n_webhook_url}")
            
            # Send to n8n with timeout
            response = requests.post(
                self.n8n_webhook_url,
                json=n8n_payload,
                headers=n8n_headers,
                timeout=60
            )
            
            if response.status_code == 200:
                print(f"[N8N] Success response from n8n for job {job_id}")
                
                # Process n8n response
                success = await self.handle_n8n_response_enhanced(
                    client_id, job_id, response, text
                )
                
                if not success:
                    await self.send_error(client_id, "Error processing AI response")
                    
            else:
                print(f"[N8N] Failed: {response.status_code} - {response.text}")
                await self.send_error(client_id, f"AI service error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            await self.send_error(client_id, "AI service timeout - please try again")
        except Exception as e:
            print(f"[ERROR] n8n processing job {job_id} failed: {e}")
            await self.send_error(client_id, f"AI processing error: {e}")
        finally:
            # Remove job from active jobs
            if client_id in self.active_connections:
                self.active_connections[client_id]["active_jobs"].discard(job_id)
            print(f"[JOB] Completed n8n processing job {job_id}")
    
    async def handle_n8n_response_enhanced(self, client_id, job_id, response, original_text):
        """Enhanced n8n response handling with better audio processing"""
        try:
            content_type = response.headers.get('content-type', '')
            
            # Parse response based on content type
            if 'application/json' in content_type:
                json_data = response.json()
                
                if isinstance(json_data, list) and len(json_data) > 0:
                    audio_base64 = json_data[0].get('data')
                    ai_response_text = json_data[0].get('response_text', 'Respuesta de la IA')
                else:
                    audio_base64 = json_data.get('data') or json_data.get('audio_data')
                    ai_response_text = json_data.get('response_text', 'Respuesta de la IA')
                
                if not audio_base64:
                    raise Exception("No audio data in n8n response")
                
                audio_data = base64.b64decode(audio_base64)
                
            else:
                # Binary response
                audio_data = response.content
                ai_response_text = response.headers.get('ai-response-text', 'Respuesta de la IA')
            
            if len(audio_data) < 1000:
                raise Exception("Audio data too small")
            
            print(f"[N8N] Processing audio response: {len(audio_data)} bytes")
            
            # Save and convert audio
            temp_mp3 = os.path.join(self.temp_dir, f"{job_id}.mp3")
            temp_wav = os.path.join(self.temp_dir, f"{job_id}.wav")
            
            with open(temp_mp3, 'wb') as f:
                f.write(audio_data)
            
            # Convert to WAV
            if not self.convert_mp3_to_wav(temp_mp3, temp_wav):
                raise Exception("Audio conversion failed")
            
            # Update client
            await self.send_message(client_id, {
                "type": "processing_status",
                "job_id": job_id,
                "stage": "generating_avatar",
                "message": "Generando avatar con respuesta de IA..."
            })
            
            # Generate avatar
            video_url = await self.generate_avatar_video_enhanced(temp_wav, job_id)
            
            if video_url:
                # Add to conversation history
                if client_id in self.active_connections:
                    self.active_connections[client_id]["session_data"]["conversation_history"].append({
                        "type": "assistant",
                        "text": ai_response_text,
                        "source": "n8n_ai",
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Send response
                await self.send_message(client_id, {
                    "type": "avatar_video",
                    "job_id": job_id,
                    "video_url": video_url,
                    "text": ai_response_text,
                    "source": "n8n_ai_response",
                    "original_text": original_text,
                    "transition": {
                        "type": "smooth_overlay",
                        "duration": 0.8
                    }
                })
                
                print(f"[SUCCESS] Generated avatar for n8n response")
                return True
            else:
                raise Exception("Avatar generation failed")
            
        except Exception as e:
            print(f"[ERROR] n8n response handling failed: {e}")
            return False
        finally:
            # Cleanup temp files
            for temp_file in [temp_mp3, temp_wav]:
                if 'temp_mp3' in locals() and os.path.exists(temp_mp3):
                    os.remove(temp_mp3)
                if 'temp_wav' in locals() and os.path.exists(temp_wav):
                    os.remove(temp_wav)
    
    async def transcribe_audio_enhanced(self, audio_file):
        """Enhanced transcription with multiple fallbacks"""
        try:
            loop = asyncio.get_event_loop()
            
            def transcribe():
                try:
                    with sr.AudioFile(audio_file) as source:
                        # Adjust for ambient noise
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio = self.recognizer.record(source)
                    
                    # Try Google Speech Recognition first
                    try:
                        text = self.recognizer.recognize_google(audio, language='es-ES')
                        print(f"[TRANSCRIPT] Google: {text}")
                        return text
                    except sr.UnknownValueError:
                        print("[TRANSCRIPT] Google couldn't understand audio")
                    except sr.RequestError as e:
                        print(f"[TRANSCRIPT] Google error: {e}")
                    
                    # Fallback to offline recognition
                    try:
                        text = self.recognizer.recognize_sphinx(audio, language='es-ES')
                        print(f"[TRANSCRIPT] Sphinx: {text}")
                        return text
                    except:
                        print("[TRANSCRIPT] Sphinx failed")
                    
                    return None
                    
                except Exception as e:
                    print(f"[ERROR] Transcription error: {e}")
                    return None
            
            # Run with timeout
            text = await asyncio.wait_for(
                loop.run_in_executor(None, transcribe),
                timeout=self.transcription_timeout
            )
            return text
            
        except asyncio.TimeoutError:
            print("[ERROR] Transcription timeout")
            return None
        except Exception as e:
            print(f"[ERROR] Transcription error: {e}")
            return None
    
    async def generate_avatar_video_enhanced(self, audio_file, job_id):
        """Enhanced avatar generation with better tracking"""
        try:
            print(f"[AVATAR] Generating video for job {job_id}")
            
            with open(audio_file, 'rb') as f:
                files = {'audio_file': f}
                data = {'sync': False}
                
                response = requests.post(
                    f"{self.api_url}/generate",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                api_job_id = result['job_id']
                print(f"[AVATAR] API job created: {api_job_id}")
                
                # Store job mapping
                self.active_jobs[job_id] = {
                    "api_job_id": api_job_id,
                    "status": "processing",
                    "created_at": datetime.now()
                }
                
                # Wait for completion with enhanced polling
                video_url = await self.wait_for_avatar_enhanced(api_job_id, job_id)
                return video_url
            else:
                print(f"[ERROR] Avatar API failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Avatar generation error: {e}")
            return None
    
    async def wait_for_avatar_enhanced(self, api_job_id, internal_job_id, max_wait=120):
        """Enhanced avatar completion waiting with better progress tracking"""
        try:
            start_time = time.time()
            check_count = 0
            last_progress = -1
            
            while time.time() - start_time < max_wait:
                check_count += 1
                
                response = requests.get(f"{self.api_url}/status/{api_job_id}")
                
                if response.status_code == 200:
                    status = response.json()
                    current_progress = status.get('progress', 0)
                    
                    # Only log progress changes
                    if current_progress != last_progress:
                        print(f"[AVATAR] Job {api_job_id} progress: {current_progress}%")
                        last_progress = current_progress
                    
                    if status['status'] == 'completed':
                        video_url = f"{self.api_url}/download/{api_job_id}"
                        print(f"[AVATAR] Completed: {video_url}")
                        return video_url
                    elif status['status'] == 'failed':
                        print(f"[ERROR] Avatar generation failed: {status.get('error')}")
                        return None
                
                # Progressive wait times
                if check_count < 5:
                    await asyncio.sleep(2)
                elif check_count < 15:
                    await asyncio.sleep(3)
                else:
                    await asyncio.sleep(5)
            
            print(f"[ERROR] Avatar generation timeout for {api_job_id}")
            return None
            
        except Exception as e:
            print(f"[ERROR] Avatar waiting error: {e}")
            return None
    
    def convert_mp3_to_wav(self, mp3_path, wav_path):
        """Convert MP3 to WAV using ffmpeg"""
        try:
            cmd = [
                'ffmpeg', '-i', mp3_path,
                '-ar', '16000', '-ac', '1', '-sample_fmt', 's16',
                '-y', wav_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except Exception as e:
            print(f"[ERROR] MP3 to WAV conversion failed: {e}")
            return False
    
    # Utility methods
    async def send_message(self, client_id, message):
        """Send message to specific client"""
        if client_id not in self.active_connections:
            return False
        
        try:
            websocket = self.active_connections[client_id]["websocket"]
            await websocket.send(json.dumps(message))
            return True
        except Exception as e:
            print(f"[ERROR] Failed to send message to {client_id}: {e}")
            return False
    
    async def send_error(self, client_id, error_message):
        """Send error message to client"""
        await self.send_message(client_id, {
            "type": "error",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        })
    
    async def cleanup_client(self, client_id):
        """Clean up client data"""
        if client_id in self.active_connections:
            client_info = self.active_connections[client_id]
            
            # Cancel any active jobs
            for job_id in client_info["active_jobs"]:
                print(f"[CLEANUP] Canceling job {job_id} for client {client_id}")
            
            del self.active_connections[client_id]
        
        print(f"[CLEANUP] Client {client_id} cleaned up")
    
    async def start_server(self, host="localhost", port=8765):
        """Start the enhanced WebSocket server"""
        print(f"[SERVER] Starting Enhanced Voice Chat Server on ws://{host}:{port}")
        
        async with websockets.serve(self.handle_client, host, port):
            print(f"[SUCCESS] Enhanced Voice Chat Server running on ws://{host}:{port}")
            print(f"[INFO] Max concurrent jobs per client: {self.max_concurrent_jobs_per_client}")
            print(f"[INFO] Audio chunk size limit: {self.max_audio_chunk_size / 1024}KB")
            await asyncio.Future()  # Run forever

def main():
    """Main function with enhanced configuration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Voice Chat Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Avatar API URL")
    parser.add_argument("--n8n-webhook", default="https://automation.maguitech.com/webhook/avatar", help="n8n webhook URL")
    
    args = parser.parse_args()
    
    server = EnhancedVoiceChatServer(
        api_url=args.api_url,
        n8n_webhook_url=args.n8n_webhook
    )
    
    try:
        asyncio.run(server.start_server(args.host, args.port))
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Enhanced Voice Chat Server stopped")

if __name__ == "__main__":
    main()