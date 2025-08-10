#!/usr/bin/env python3
"""
Test script for MP3 support in Avatar API
"""

import requests
import time
import os

def test_mp3_api():
    """Test the API with MP3 file"""
    api_url = "http://localhost:8000"
    
    # Check if API is running
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not running. Start it with: python avatar_api.py")
            return
    except requests.exceptions.RequestException:
        print("❌ API is not reachable. Start it with: python avatar_api.py")
        return
    
    print("✅ API is running")
    
    # Test with different audio formats
    test_files = [
        "./data/preload/asr_example.wav",  # Original WAV
        # Add your MP3 files here for testing
        # "./test_audio.mp3",
        # "./test_audio.m4a",
    ]
    
    for audio_file in test_files:
        if not os.path.exists(audio_file):
            print(f"⚠️  Skipping {audio_file} (not found)")
            continue
            
        print(f"\n🎵 Testing with: {audio_file}")
        
        # Upload file (synchronous mode for testing)
        try:
            with open(audio_file, 'rb') as f:
                files = {'audio_file': f}
                data = {'sync': True}
                
                print("📤 Uploading and processing...")
                start_time = time.time()
                
                response = requests.post(
                    f"{api_url}/generate",
                    files=files,
                    data=data,
                    timeout=300  # 5 minutes
                )
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    job_id = result['job_id']
                    
                    print(f"✅ Processing completed in {processing_time:.1f}s")
                    print(f"🆔 Job ID: {job_id}")
                    
                    # Download the video
                    download_response = requests.get(
                        f"{api_url}/download/{job_id}",
                        stream=True
                    )
                    
                    if download_response.status_code == 200:
                        output_filename = f"api_test_{job_id}.mp4"
                        with open(output_filename, 'wb') as video_file:
                            for chunk in download_response.iter_content(chunk_size=8192):
                                video_file.write(chunk)
                        
                        file_size = os.path.getsize(output_filename) / (1024 * 1024)
                        print(f"📥 Downloaded: {output_filename} ({file_size:.1f} MB)")
                        
                        # Play video
                        play = input("▶️  Play video? (y/n): ").strip().lower()
                        if play == 'y':
                            if os.name == 'nt':  # Windows
                                os.startfile(output_filename)
                    else:
                        print(f"❌ Download failed: {download_response.status_code}")
                        
                else:
                    print(f"❌ Upload failed: {response.status_code}")
                    print(f"Error: {response.text}")
                    
        except requests.exceptions.Timeout:
            print("❌ Request timed out")
        except Exception as e:
            print(f"❌ Error: {e}")

def convert_wav_to_mp3_example():
    """Example to convert the sample WAV to MP3 for testing"""
    input_wav = "./data/preload/asr_example.wav"
    output_mp3 = "./test_example.mp3"
    
    if not os.path.exists(input_wav):
        print(f"❌ Sample WAV not found: {input_wav}")
        return
    
    if os.path.exists(output_mp3):
        print(f"✅ MP3 test file already exists: {output_mp3}")
        return
    
    try:
        import subprocess
        cmd = [
            'ffmpeg',
            '-i', input_wav,
            '-codec:a', 'mp3',
            '-b:a', '128k',
            '-y', output_mp3
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Created MP3 test file: {output_mp3}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create MP3: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Avatar API MP3 Test")
    
    # Create MP3 test file if needed
    convert_wav_to_mp3_example()
    
    # Test the API
    test_mp3_api()