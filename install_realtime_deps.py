#!/usr/bin/env python3
"""
Install dependencies for real-time avatar system
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"[SUCCESS] {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install {package}: {e}")
        return False

def main():
    """Install all required packages"""
    print("[START] Installing Real-time Avatar System Dependencies")
    print("=" * 50)
    
    packages = [
        "websockets>=11.0.0",
        "SpeechRecognition>=3.10.0",
        "pydub>=0.25.1",
        "requests>=2.31.0",
        "pyaudio",  # For microphone access (optional)
        "fastapi>=0.104.0",  # For API server
        "uvicorn>=0.24.0",   # For API server
        "python-multipart>=0.0.6"  # For file uploads and webhook
    ]
    
    successful = 0
    failed = 0
    
    for package in packages:
        print(f"\n[EMOJI] Installing {package}...")
        if install_package(package):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"[INFO] Installation Summary:")
    print(f"[SUCCESS] Successful: {successful}")
    print(f"[ERROR] Failed: {failed}")
    
    if failed == 0:
        print("\n[SUCCESS] All dependencies installed successfully!")
        print("\nNext steps:")
        print("1. Start the Avatar API server:")
        print("   python avatar_api.py")
        print("\n2. Start the WebSocket server:")
        print("   python realtime_websocket_avatar.py")
        print("\n3. Open the client in your browser:")
        print("   realtime_avatar_client.html")
    else:
        print(f"\n[WARNING]  Some packages failed to install. Please install them manually.")
        
    print("\n[TEXT] Note: If pyaudio installation fails on Windows, try:")
    print("   pip install pipwin")
    print("   pipwin install pyaudio")

if __name__ == "__main__":
    main()