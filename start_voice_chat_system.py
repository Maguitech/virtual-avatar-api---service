#!/usr/bin/env python3
"""
Complete Voice Chat System Launcher
Starts all components needed for the voice chat system with avatar
"""

import subprocess
import time
import threading
import webbrowser
import os
import sys
import signal
import requests
from pathlib import Path

class VoiceChatSystemLauncher:
    def __init__(self):
        self.processes = []
        self.base_dir = Path(__file__).parent
        self.running = False
        
        # Configuration
        self.config = {
            "avatar_api": {
                "port": 8000,
                "script": "avatar_api.py"
            },
            "voice_chat_server": {
                "port": 8765,
                "script": "enhanced_voice_chat_server.py"  # or realtime_websocket_avatar.py
            },
            "client": {
                "file": "voice_chat_client.html"
            },
            "n8n_webhook": "https://automation.maguitech.com/webhook/avatar"
        }
        
    def check_dependencies(self):
        """Check if all required files exist"""
        print("🔍 Checking dependencies...")
        
        required_files = [
            self.config["avatar_api"]["script"],
            self.config["voice_chat_server"]["script"],
            self.config["client"]["file"],
            "lite_avatar.py",
            "data/preload/bg_video_h264.mp4"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.base_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ Missing required files:")
            for file_path in missing_files:
                print(f"   - {file_path}")
            return False
        
        print("✅ All required files found")
        return True
    
    def wait_for_service(self, name, url, max_wait=30):
        """Wait for a service to become available"""
        print(f"⏳ Waiting for {name} to start...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print(f"✅ {name} is ready!")
                    return True
            except:
                pass
            time.sleep(1)
        
        print(f"❌ {name} failed to start within {max_wait} seconds")
        return False
    
    def start_avatar_api(self):
        """Start the Avatar API server"""
        print("🚀 Starting Avatar API server...")
        
        api_script = self.base_dir / self.config["avatar_api"]["script"]
        
        try:
            process = subprocess.Popen([
                sys.executable, str(api_script)
            ], cwd=str(self.base_dir))
            
            self.processes.append({
                "name": "Avatar API",
                "process": process,
                "port": self.config["avatar_api"]["port"]
            })
            
            # Wait for API to be ready
            if self.wait_for_service(
                "Avatar API", 
                f"http://localhost:{self.config['avatar_api']['port']}/health"
            ):
                return True
            else:
                print("❌ Avatar API failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start Avatar API: {e}")
            return False
    
    def start_voice_chat_server(self):
        """Start the Voice Chat WebSocket server"""
        print("🎤 Starting Voice Chat server...")
        
        server_script = self.base_dir / self.config["voice_chat_server"]["script"]
        
        try:
            process = subprocess.Popen([
                sys.executable, str(server_script),
                "--api-url", f"http://localhost:{self.config['avatar_api']['port']}",
                "--n8n-webhook", self.config["n8n_webhook"]
            ], cwd=str(self.base_dir))
            
            self.processes.append({
                "name": "Voice Chat Server",
                "process": process,
                "port": self.config["voice_chat_server"]["port"]
            })
            
            # Give WebSocket server time to start
            time.sleep(3)
            print("✅ Voice Chat Server started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start Voice Chat server: {e}")
            return False
    
    def start_client(self):
        """Open the voice chat client in browser"""
        print("🌐 Opening Voice Chat client...")
        
        client_file = self.base_dir / self.config["client"]["file"]
        
        try:
            # Try to open in default browser
            webbrowser.open(f"file://{client_file.absolute()}")
            print("✅ Voice Chat client opened in browser")
            return True
        except Exception as e:
            print(f"⚠️  Could not auto-open browser: {e}")
            print(f"📱 Please manually open: {client_file.absolute()}")
            return True
    
    def show_system_status(self):
        """Show the status of all system components"""
        print("\n" + "="*60)
        print("🤖 VOICE CHAT SYSTEM STATUS")
        print("="*60)
        
        for proc_info in self.processes:
            status = "🟢 Running" if proc_info["process"].poll() is None else "🔴 Stopped"
            print(f"{proc_info['name']}: {status} (Port {proc_info['port']})")
        
        print("\n📋 SYSTEM ENDPOINTS:")
        print(f"Avatar API:        http://localhost:{self.config['avatar_api']['port']}")
        print(f"Avatar API Docs:   http://localhost:{self.config['avatar_api']['port']}/docs")
        print(f"WebSocket Server:  ws://localhost:{self.config['voice_chat_server']['port']}")
        print(f"Voice Chat Client: file://{(self.base_dir / self.config['client']['file']).absolute()}")
        print(f"n8n Webhook:       {self.config['n8n_webhook']}")
        
        print("\n🎯 USAGE INSTRUCTIONS:")
        print("1. Open the Voice Chat Client in your browser")
        print("2. Click the 📞 button to connect")
        print("3. Click the 🎤 button to start talking")
        print("4. Watch your avatar respond in real-time!")
        print("5. You can also type messages in the chat panel")
        
        print("\n⚠️  TROUBLESHOOTING:")
        print("- Make sure your microphone permissions are enabled")
        print("- Check that FFmpeg is installed for audio conversion")
        print("- Verify n8n webhook is accessible and configured")
        print("- GPU processing requires CUDA-compatible graphics card")
        
        print("\n💡 PERFORMANCE TIPS:")
        if "batch_processing=True" in open(self.base_dir / "avatar_api.py").read():
            print("- ✅ Batch processing enabled (optimal for GPU)")
            print("- ✅ Expected generation time: ~0.05-0.2 seconds per response")
        else:
            print("- ⚠️  Consider enabling batch_processing=True in avatar_api.py")
            print("- ⚠️  Current generation time: ~0.2+ seconds per response")
        
        print("\n🔧 CONFIGURATION:")
        print(f"- Max concurrent jobs per client: 1")
        print(f"- Audio chunk size limit: 1MB")
        print(f"- Transcription timeout: 10 seconds")
        print(f"- Avatar generation timeout: 120 seconds")
        
        print("="*60)
    
    def monitor_processes(self):
        """Monitor running processes and restart if needed"""
        while self.running:
            for proc_info in self.processes[:]:  # Copy list to avoid modification during iteration
                if proc_info["process"].poll() is not None:
                    print(f"⚠️  {proc_info['name']} has stopped unexpectedly")
                    
                    # Try to restart
                    print(f"🔄 Attempting to restart {proc_info['name']}...")
                    if proc_info["name"] == "Avatar API":
                        self.processes.remove(proc_info)
                        if not self.start_avatar_api():
                            print(f"❌ Failed to restart {proc_info['name']}")
                    elif proc_info["name"] == "Voice Chat Server":
                        self.processes.remove(proc_info)
                        if not self.start_voice_chat_server():
                            print(f"❌ Failed to restart {proc_info['name']}")
            
            time.sleep(5)  # Check every 5 seconds
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutting down Voice Chat System...")
        self.shutdown()
        sys.exit(0)
    
    def shutdown(self):
        """Shutdown all processes"""
        self.running = False
        
        print("🔄 Stopping all services...")
        for proc_info in self.processes:
            try:
                proc_info["process"].terminate()
                print(f"🛑 Stopped {proc_info['name']}")
            except:
                try:
                    proc_info["process"].kill()
                except:
                    pass
        
        # Wait for processes to terminate
        for proc_info in self.processes:
            try:
                proc_info["process"].wait(timeout=5)
            except:
                pass
        
        print("✅ Voice Chat System shutdown complete")
    
    def start(self):
        """Start the complete voice chat system"""
        print("🤖 STARTING VOICE CHAT SYSTEM WITH AVATAR")
        print("="*50)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Check dependencies
        if not self.check_dependencies():
            print("\n❌ Cannot start system due to missing dependencies")
            return False
        
        # Start services in order
        if not self.start_avatar_api():
            print("\n❌ Failed to start Avatar API")
            return False
        
        if not self.start_voice_chat_server():
            print("\n❌ Failed to start Voice Chat server")
            self.shutdown()
            return False
        
        # Start client
        self.start_client()
        
        # Show system status
        self.show_system_status()
        
        # Start monitoring
        self.running = True
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()
        
        return True

def main():
    """Main function"""
    launcher = VoiceChatSystemLauncher()
    
    try:
        success = launcher.start()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ System startup failed: {e}")
        launcher.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()