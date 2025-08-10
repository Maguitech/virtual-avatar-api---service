import os
import sys
import time
import threading
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
from datetime import datetime
from lite_avatar import liteAvatar

class RealTimeAvatar:
    def __init__(self, data_dir="./data/preload", result_dir="./result_realtime"):
        """
        Initialize real-time avatar system
        """
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.avatar = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration = 5  # seconds per chunk
        
        # Create result directory
        os.makedirs(result_dir, exist_ok=True)
        
        # Initialize avatar
        print("Initializing avatar system...")
        self.avatar = liteAvatar(
            data_dir=data_dir, 
            num_threads=1, 
            generate_offline=True
        )
        print("Avatar system ready!")
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio recording"""
        if status:
            print(f"Audio status: {status}")
        
        if self.is_recording:
            # Convert to mono if stereo
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata[:, 0]
            
            self.audio_queue.put(audio_data.copy())
    
    def start_recording(self):
        """Start real-time audio recording"""
        self.is_recording = True
        
        # Clear the queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
        
        print("🎤 Recording started... Speak now!")
        print("Press Ctrl+C to stop recording")
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            blocksize=1024
        ):
            try:
                audio_chunks = []
                start_time = time.time()
                
                while self.is_recording:
                    try:
                        # Get audio data with timeout
                        chunk = self.audio_queue.get(timeout=0.1)
                        audio_chunks.append(chunk)
                        
                        # Process chunks every few seconds
                        if time.time() - start_time >= self.chunk_duration:
                            if audio_chunks:
                                self.process_audio_chunk(audio_chunks)
                                audio_chunks = []
                                start_time = time.time()
                                
                    except queue.Empty:
                        continue
                        
            except KeyboardInterrupt:
                print("\\n🛑 Recording stopped by user")
                self.is_recording = False
                
                # Process remaining audio
                if audio_chunks:
                    self.process_audio_chunk(audio_chunks)
    
    def process_audio_chunk(self, audio_chunks):
        """Process audio chunk and generate avatar video"""
        if not audio_chunks:
            return
            
        print("\\n🎬 Processing audio chunk...")
        
        # Combine chunks
        audio_data = np.concatenate(audio_chunks)
        
        # Check if audio has enough content (not just silence)
        if np.max(np.abs(audio_data)) < 0.01:
            print("⚠️  Audio too quiet, skipping...")
            return
        
        # Create temporary audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_audio_path = os.path.join(self.result_dir, f"temp_audio_{timestamp}.wav")
        
        # Save audio file
        sf.write(temp_audio_path, audio_data, self.sample_rate)
        print(f"💾 Saved audio: {temp_audio_path}")
        
        # Generate avatar video
        try:
            output_video = os.path.join(self.result_dir, f"avatar_video_{timestamp}.mp4")
            
            print("🎭 Generating avatar video...")
            start_time = time.time()
            
            # Use the existing lite_avatar system to generate video
            self.generate_avatar_video(temp_audio_path, output_video)
            
            generation_time = time.time() - start_time
            print(f"✅ Avatar video generated in {generation_time:.2f}s: {output_video}")
            
            # Clean up temporary audio file
            try:
                os.remove(temp_audio_path)
            except:
                pass
                
        except Exception as e:
            print(f"❌ Error generating avatar video: {e}")
    
    def generate_avatar_video(self, audio_file_path, output_video_path):
        """Generate avatar video using the original lite_avatar functionality"""
        temp_result_dir = os.path.dirname(output_video_path)
        
        # Use the original handle method to generate frames and video
        self.avatar.handle(audio_file_path, temp_result_dir)
        
        # The original method creates test_demo.mp4, so rename it
        original_output = os.path.join(temp_result_dir, "test_demo.mp4")
        if os.path.exists(original_output):
            import shutil
            shutil.move(original_output, output_video_path)
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
    
    def interactive_mode(self):
        """Interactive conversation mode"""
        print("\\n" + "="*50)
        print("🤖 REAL-TIME AVATAR CHAT")
        print("="*50)
        print("Commands:")
        print("  'start' - Start recording")
        print("  'stop'  - Stop recording")
        print("  'quit'  - Exit program")
        print("="*50)
        
        while True:
            try:
                command = input("\\nEnter command: ").strip().lower()
                
                if command == 'start':
                    if not self.is_recording:
                        # Start recording in a separate thread
                        recording_thread = threading.Thread(target=self.start_recording)
                        recording_thread.daemon = True
                        recording_thread.start()
                    else:
                        print("Already recording!")
                
                elif command == 'stop':
                    if self.is_recording:
                        self.stop_recording()
                        print("Recording stopped.")
                    else:
                        print("Not recording.")
                
                elif command in ['quit', 'exit', 'q']:
                    if self.is_recording:
                        self.stop_recording()
                    print("Goodbye! 👋")
                    break
                    
                else:
                    print("Unknown command. Use 'start', 'stop', or 'quit'")
                    
            except KeyboardInterrupt:
                if self.is_recording:
                    self.stop_recording()
                print("\\nGoodbye! 👋")
                break

def main():
    """Main function"""
    print("🚀 Starting Real-Time Avatar Chat System...")
    
    try:
        # Create real-time avatar system
        rt_avatar = RealTimeAvatar()
        
        # Start interactive mode
        rt_avatar.interactive_mode()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()