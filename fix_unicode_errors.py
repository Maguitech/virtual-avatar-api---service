#!/usr/bin/env python3
"""
Fix Unicode emoji errors for Windows console
"""

import re
import os

def remove_emojis_from_file(filename):
    """Remove emojis and replace with text equivalents"""
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    
    # Emoji replacements
    emoji_replacements = {
        '🚀': '[START]',
        '✅': '[SUCCESS]',
        '❌': '[ERROR]',
        '🔌': '[CONNECT]',
        '🎤': '[AUDIO]',
        '📝': '[TEXT]',
        '🎭': '[AVATAR]',
        '📊': '[INFO]',
        '🔄': '[PROCESSING]',
        '📤': '[UPLOAD]',
        '📥': '[DOWNLOAD]',
        '💬': '[MESSAGE]',
        '🎬': '[VIDEO]',
        '📁': '[FOLDER]',
        '🌐': '[WEBSOCKET]',
        '⚠️': '[WARNING]',
        '🔇': '[SILENCE]',
        '📋': '[COPY]',
        '⏹️': '[STOP]',
        '🎊': '[CELEBRATION]',
        '🛑': '[SHUTDOWN]',
        '🔨': '[FORCE]',
        '💀': '[KILLED]',
        '👋': '[GOODBYE]',
        '🔄': '[RELOAD]',
        '🎉': '[SUCCESS]',
        '📺': '[PLAY]'
    }
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace emojis
        for emoji, replacement in emoji_replacements.items():
            content = content.replace(emoji, replacement)
        
        # Remove any remaining emojis using regex
        # This regex matches most Unicode emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        content = emoji_pattern.sub('[EMOJI]', content)
        
        if content != original_content:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[SUCCESS] Fixed emojis in {filename}")
        else:
            print(f"[INFO] No emojis found in {filename}")
            
    except Exception as e:
        print(f"[ERROR] Error processing {filename}: {e}")

def main():
    """Fix emoji issues in all Python files"""
    print("[INFO] Fixing Unicode emoji errors...")
    
    files_to_fix = [
        'avatar_api.py',
        'realtime_websocket_avatar.py',
        'start_realtime_system.py',
        'install_realtime_deps.py'
    ]
    
    for filename in files_to_fix:
        remove_emojis_from_file(filename)
    
    print("[SUCCESS] All files processed!")
    print("[INFO] You can now run: python start_realtime_system.py")

if __name__ == "__main__":
    main()