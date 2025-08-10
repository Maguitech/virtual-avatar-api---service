#!/usr/bin/env python3
"""
Start the complete real-time avatar system
Launches API server and WebSocket server
"""

import subprocess
import sys
import time
import os
import threading
import webbrowser
from pathlib import Path

def run_server(name, command, cwd=None):
    """Run a server in a separate process"""
    print(f"[START] Starting {name}...")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd or os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        def print_output(stream, prefix):
            for line in iter(stream.readline, ''):
                print(f"[{prefix}] {line.strip()}")
        
        # Start threads to handle output
        threading.Thread(
            target=print_output, 
            args=(process.stdout, name), 
            daemon=True
        ).start()
        
        threading.Thread(
            target=print_output, 
            args=(process.stderr, f"{name}-ERR"), 
            daemon=True
        ).start()
        
        return process
        
    except Exception as e:
        print(f"[ERROR] Failed to start {name}: {e}")
        return None

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['fastapi', 'uvicorn', 'websockets', 'speech_recognition', 'pydub']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("[ERROR] Missing dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nRun: python install_realtime_deps.py")
        return False
    
    return True

def main():
    """Main function to start the system"""
    print("🤖 Real-time Avatar System Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("[ERROR] Please install dependencies first")
        sys.exit(1)
    
    # Check if files exist
    required_files = [
        'avatar_api.py',
        'realtime_websocket_avatar.py',
        'realtime_avatar_client.html',
        'http_server.py'
    ]
    
    for file in required_files:
        if not Path(file).exists():
            print(f"[ERROR] Required file not found: {file}")
            sys.exit(1)
    
    print("[SUCCESS] All dependencies and files found")
    
    processes = []
    
    try:
        # Start Avatar API server
        api_process = run_server(
            "Avatar-API",
            [sys.executable, "avatar_api.py"]
        )
        if api_process:
            processes.append(("Avatar-API", api_process))
            time.sleep(3)  # Wait for API to start
        
        # Start WebSocket server
        ws_process = run_server(
            "WebSocket",
            [sys.executable, "realtime_websocket_avatar.py"]
        )
        if ws_process:
            processes.append(("WebSocket", ws_process))
            time.sleep(2)  # Wait for WebSocket to start
        
        # Start HTTP server for static files
        http_process = run_server(
            "HTTP-Static",
            [sys.executable, "http_server.py"]
        )
        if http_process:
            processes.append(("HTTP-Static", http_process))
            time.sleep(1)  # Wait for HTTP server to start
        
        if processes:
            print("\n" + "=" * 50)
            print("[SUCCESS] System started successfully!")
            print("\n[WEBSOCKET] Services running:")
            print("   - Avatar API: http://localhost:8000")
            print("   - API Docs: http://localhost:8000/docs")
            print("   - WebSocket: ws://localhost:8765")
            print("   - HTTP Static: http://localhost:8080")
            print("\n[AVATAR] Open the client:")
            print("   - URL: http://localhost:8080/realtime_avatar_client.html")
            print("   - File: realtime_avatar_client.html (fallback)")
            
            # Try to open client in browser
            try:
                webbrowser.open("http://localhost:8080/realtime_avatar_client.html")
                print("   - Browser opened automatically")
            except Exception as e:
                print(f"   - Open manually (browser auto-open failed: {e})")
            
            print("\n[WARNING]  Press Ctrl+C to stop all servers")
            
            # Keep the main process alive
            try:
                while True:
                    # Check if any process died
                    for name, process in processes:
                        if process.poll() is not None:
                            print(f"[ERROR] {name} server stopped unexpectedly")
                            return_code = process.poll()
                            print(f"   Return code: {return_code}")
                    
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n\n[SHUTDOWN] Shutting down servers...")
                
        else:
            print("[ERROR] Failed to start any servers")
            
    except Exception as e:
        print(f"[ERROR] Error starting system: {e}")
        
    finally:
        # Clean up processes
        for name, process in processes:
            try:
                print(f"[RELOAD] Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
                print(f"[SUCCESS] {name} stopped")
            except subprocess.TimeoutExpired:
                print(f"[FORCE] Force killing {name}...")
                process.kill()
                process.wait()
                print(f"[KILLED] {name} killed")
            except Exception as e:
                print(f"[ERROR] Error stopping {name}: {e}")
        
        print("\n[GOODBYE] System shutdown complete")

if __name__ == "__main__":
    main()