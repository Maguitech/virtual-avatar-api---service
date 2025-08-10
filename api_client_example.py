#!/usr/bin/env python3
"""
Avatar API Client Example
Demonstrates how to use the Avatar API
"""

import requests
import time
import os
from pathlib import Path

class AvatarAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def upload_audio_sync(self, audio_file_path):
        """Upload audio and wait for completion (synchronous)"""
        if not os.path.exists(audio_file_path):
            print(f"❌ Audio file not found: {audio_file_path}")
            return None
        
        print(f"📤 Uploading: {audio_file_path}")
        
        with open(audio_file_path, 'rb') as f:
            files = {'audio_file': f}
            data = {'sync': True}
            
            try:
                response = requests.post(
                    f"{self.base_url}/generate",
                    files=files,
                    data=data,
                    timeout=300  # 5 minutes timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Success! Job ID: {result['job_id']}")
                    return result
                else:
                    print(f"❌ Error: {response.status_code} - {response.text}")
                    return None
                    
            except requests.exceptions.Timeout:
                print("❌ Request timed out")
                return None
            except Exception as e:
                print(f"❌ Error: {e}")
                return None
    
    def upload_audio_async(self, audio_file_path):
        """Upload audio for asynchronous processing"""
        if not os.path.exists(audio_file_path):
            print(f"❌ Audio file not found: {audio_file_path}")
            return None
        
        print(f"📤 Uploading: {audio_file_path}")
        
        with open(audio_file_path, 'rb') as f:
            files = {'audio_file': f}
            data = {'sync': False}
            
            try:
                response = requests.post(
                    f"{self.base_url}/generate",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Upload successful! Job ID: {result['job_id']}")
                    return result
                else:
                    print(f"❌ Error: {response.status_code} - {response.text}")
                    return None
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                return None
    
    def check_status(self, job_id):
        """Check the status of a job"""
        try:
            response = requests.get(f"{self.base_url}/status/{job_id}")
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Error checking status: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def wait_for_completion(self, job_id, check_interval=2):
        """Wait for job completion"""
        print(f"⏳ Waiting for job {job_id} to complete...")
        
        while True:
            status = self.check_status(job_id)
            
            if not status:
                print("❌ Failed to check status")
                return False
            
            print(f"📊 Status: {status['status']}")
            
            if status['status'] == 'completed':
                print("✅ Job completed successfully!")
                return True
            elif status['status'] == 'failed':
                print(f"❌ Job failed: {status.get('error', 'Unknown error')}")
                return False
            
            time.sleep(check_interval)
    
    def download_video(self, job_id, output_path=None):
        """Download the generated video"""
        if not output_path:
            output_path = f"downloaded_avatar_{job_id}.mp4"
        
        try:
            response = requests.get(
                f"{self.base_url}/download/{job_id}",
                stream=True
            )
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"📥 Downloaded: {output_path} ({file_size:.1f} MB)")
                return output_path
            else:
                print(f"❌ Download failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            return None
    
    def list_jobs(self):
        """List all jobs"""
        try:
            response = requests.get(f"{self.base_url}/jobs")
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n📋 Found {result['total_jobs']} jobs:")
                
                for job in result['jobs']:
                    print(f"  🆔 {job['job_id']} - Status: {job['status']}")
                    if 'processing_time_seconds' in job:
                        print(f"     ⏱️  Processing time: {job['processing_time_seconds']:.1f}s")
                
                return result
            else:
                print(f"❌ Error listing jobs: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                health = response.json()
                print(f"🏥 API Status: {health['status']}")
                print(f"🤖 Avatar System: {health['avatar_system']}")
                return health
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ API not reachable: {e}")
            return None

def demo_sync_processing():
    """Demo synchronous processing"""
    client = AvatarAPIClient()
    
    print("\n" + "="*50)
    print("🔄 SYNCHRONOUS PROCESSING DEMO")
    print("="*50)
    
    # Check health
    if not client.health_check():
        print("❌ API not available")
        return
    
    # Upload and process
    audio_file = "./data/preload/asr_example.wav"
    result = client.upload_audio_sync(audio_file)
    
    if result:
        job_id = result['job_id']
        
        # Download the video
        video_path = client.download_video(job_id, f"sync_result_{job_id}.mp4")
        
        if video_path:
            print(f"🎊 Success! Video saved as: {video_path}")
            
            # Open video
            if os.name == 'nt':  # Windows
                os.startfile(video_path)

def demo_async_processing():
    """Demo asynchronous processing"""
    client = AvatarAPIClient()
    
    print("\n" + "="*50)
    print("🔄 ASYNCHRONOUS PROCESSING DEMO")
    print("="*50)
    
    # Check health
    if not client.health_check():
        print("❌ API not available")
        return
    
    # Upload for async processing
    audio_file = "./data/preload/asr_example.wav"
    result = client.upload_audio_async(audio_file)
    
    if result:
        job_id = result['job_id']
        
        # Wait for completion
        if client.wait_for_completion(job_id):
            # Download the video
            video_path = client.download_video(job_id, f"async_result_{job_id}.mp4")
            
            if video_path:
                print(f"🎊 Success! Video saved as: {video_path}")

def main():
    """Main demo function"""
    print("🚀 Avatar API Client Demo")
    
    client = AvatarAPIClient()
    
    # Health check first
    print("\n🏥 Checking API health...")
    if not client.health_check():
        print("❌ Please start the API server first:")
        print("   python avatar_api.py")
        return
    
    print("\nChoose demo mode:")
    print("1. Synchronous processing (wait for completion)")
    print("2. Asynchronous processing (background processing)")
    print("3. List existing jobs")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                demo_sync_processing()
            elif choice == '2':
                demo_async_processing()
            elif choice == '3':
                client.list_jobs()
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()