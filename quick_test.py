#!/usr/bin/env python3
"""
Quick test script for avatar generation
"""

import os
import sys
from lite_avatar import liteAvatar

def test_avatar():
    """Test avatar generation with sample audio"""
    
    print("🚀 Testing Avatar System...")
    
    # Paths
    data_dir = "./data/preload"
    audio_file = "./data/preload/asr_example.wav"
    result_dir = "./test_result"
    
    # Check if files exist
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    # Create result directory
    os.makedirs(result_dir, exist_ok=True)
    
    try:
        # Initialize avatar
        print("🎭 Initializing avatar...")
        avatar = liteAvatar(
            data_dir=data_dir,
            num_threads=1,
            generate_offline=True
        )
        
        # Generate video
        print("🎬 Generating video...")
        avatar.handle(audio_file, result_dir)
        
        # Check result
        output_video = os.path.join(result_dir, "test_demo.mp4")
        if os.path.exists(output_video):
            size_mb = os.path.getsize(output_video) / (1024 * 1024)
            print(f"✅ Success! Video generated: {output_video} ({size_mb:.1f} MB)")
            
            # Ask to play
            play = input("▶️  Play video? (y/n): ").strip().lower()
            if play == 'y':
                if os.name == 'nt':
                    os.startfile(output_video)
                else:
                    import subprocess
                    subprocess.run(['xdg-open', output_video])
            
            return True
        else:
            print("❌ Video generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_avatar()
    sys.exit(0 if success else 1)