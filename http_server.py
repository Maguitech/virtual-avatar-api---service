#!/usr/bin/env python3
"""
Simple HTTP server to serve static files for the real-time avatar client
Solves CORS issues when loading local video files
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

def main():
    """Start HTTP server for static files"""
    PORT = 8080
    
    # Change to the current directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"[HTTP] Starting HTTP server on port {PORT}")
    print(f"[HTTP] Serving files from: {os.getcwd()}")
    
    try:
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            print(f"[SUCCESS] HTTP server running at http://localhost:{PORT}")
            print(f"[INFO] Open: http://localhost:{PORT}/realtime_avatar_client.html")
            print(f"[INFO] Background video: http://localhost:{PORT}/data/preload/bg_video.mp4")
            print(f"[WARNING] Press Ctrl+C to stop server")
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print(f"\n[SHUTDOWN] HTTP server stopped")
                
    except Exception as e:
        print(f"[ERROR] Failed to start HTTP server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()