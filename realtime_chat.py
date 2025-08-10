#!/usr/bin/env python3
"""
Real-time Avatar Chat System
Simple interface for live conversation with avatar
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from lite_avatar import liteAvatar

class RealTimeChatAvatar:
    def __init__(self, data_dir="./data/preload", output_dir="./realtime_output"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.avatar = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("🚀 Initializing avatar system...")
        try:
            self.avatar = liteAvatar(
                data_dir=data_dir,
                num_threads=1,
                generate_offline=True
            )
            print("✅ Avatar system ready!")
        except Exception as e:
            print(f"❌ Error initializing avatar: {e}")
            sys.exit(1)
    
    def process_audio_file(self, audio_file_path):
        """Process an audio file and generate avatar video"""
        if not os.path.exists(audio_file_path):
            print(f"❌ Audio file not found: {audio_file_path}")
            return None
        
        print(f"🎬 Processing: {audio_file_path}")
        
        # Create unique output name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = os.path.splitext(os.path.basename(audio_file_path))[0]
        output_name = f"{basename}_{timestamp}"
        
        try:
            start_time = time.time()
            
            # Use original avatar system
            self.avatar.handle(audio_file_path, self.output_dir)
            
            # The system creates test_demo.mp4, rename it
            original_video = os.path.join(self.output_dir, "test_demo.mp4")
            final_video = os.path.join(self.output_dir, f"{output_name}.mp4")
            
            if os.path.exists(original_video):
                os.rename(original_video, final_video)
                processing_time = time.time() - start_time
                print(f"✅ Video generated in {processing_time:.2f}s: {final_video}")
                return final_video
            else:
                print("❌ Video generation failed")
                return None
                
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            return None
    
    def record_audio(self, duration=5, output_path=None):
        """Record audio using system tools"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"recording_{timestamp}.wav")
        
        print(f"🎤 Recording for {duration} seconds...")
        print("Speak now!")
        
        # Try different recording methods based on OS
        try:
            if os.name == 'nt':  # Windows
                # Try using ffmpeg for recording
                cmd = [
                    'ffmpeg', 
                    '-f', 'dshow',
                    '-i', 'audio=default',
                    '-ar', '16000',
                    '-ac', '1', 
                    '-t', str(duration),
                    '-y', output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            else:  # Linux/Mac
                cmd = [
                    'ffmpeg',
                    '-f', 'pulse',
                    '-i', 'default',
                    '-ar', '16000',
                    '-ac', '1',
                    '-t', str(duration),
                    '-y', output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
            print(f"📁 Audio saved: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Recording failed: {e}")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def interactive_session(self):
        """Interactive chat session"""
        print("\n" + "="*60)
        print("🤖 REAL-TIME AVATAR CHAT SYSTEM")
        print("="*60)
        print("Commands:")
        print("  'record [seconds]' - Record audio (default: 5 seconds)")
        print("  'process <file>'   - Process an audio file")
        print("  'play <video>'     - Play a video file")
        print("  'list'             - List generated videos")
        print("  'help'             - Show this help")
        print("  'quit'             - Exit")
        print("="*60)
        
        while True:
            try:
                command = input("\n🎯 Enter command: ").strip()
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0].lower()
                
                if cmd == 'record':
                    duration = int(parts[1]) if len(parts) > 1 else 5
                    audio_file = self.record_audio(duration)
                    
                    if audio_file:
                        print("\n🎬 Generating avatar video...")
                        video_file = self.process_audio_file(audio_file)
                        
                        if video_file:
                            print(f"🎊 Success! Video: {video_file}")
                            
                            # Ask if user wants to play it
                            play = input("▶️  Play video now? (y/n): ").strip().lower()
                            if play == 'y':
                                self.play_video(video_file)
                
                elif cmd == 'process':
                    if len(parts) < 2:
                        print("❌ Usage: process <audio_file>")
                        continue
                    
                    audio_file = parts[1]
                    video_file = self.process_audio_file(audio_file)
                    
                    if video_file:
                        print(f"🎊 Success! Video: {video_file}")
                
                elif cmd == 'play':
                    if len(parts) < 2:
                        print("❌ Usage: play <video_file>")
                        continue
                    
                    video_file = parts[1]
                    self.play_video(video_file)
                
                elif cmd == 'list':
                    self.list_videos()
                
                elif cmd == 'help':
                    print("\nCommands:")
                    print("  record [seconds] - Record and process audio")
                    print("  process <file>   - Process existing audio file") 
                    print("  play <video>     - Play video file")
                    print("  list             - List all generated videos")
                    print("  quit             - Exit program")
                
                elif cmd in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def play_video(self, video_path):
        """Play video using system default player"""
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return
        
        try:
            if os.name == 'nt':  # Windows
                os.startfile(video_path)
            elif os.name == 'posix':  # Linux/Mac
                subprocess.run(['xdg-open', video_path])
            
            print(f"▶️  Playing: {video_path}")
        except Exception as e:
            print(f"❌ Error playing video: {e}")
    
    def list_videos(self):
        """List all generated videos"""
        video_files = [f for f in os.listdir(self.output_dir) if f.endswith('.mp4')]
        
        if not video_files:
            print("📁 No videos found in output directory")
            return
        
        print(f"\n📹 Videos in {self.output_dir}:")
        for i, video in enumerate(sorted(video_files), 1):
            video_path = os.path.join(self.output_dir, video)
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"  {i}. {video} ({size_mb:.1f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Real-time Avatar Chat System")
    parser.add_argument("--data_dir", default="./data/preload", 
                       help="Avatar data directory")
    parser.add_argument("--output_dir", default="./realtime_output",
                       help="Output directory for generated videos")
    
    args = parser.parse_args()
    
    # Create and start the chat system
    chat_system = RealTimeChatAvatar(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Start interactive session
    chat_system.interactive_session()

if __name__ == "__main__":
    main()