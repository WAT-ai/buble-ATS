#!/usr/bin/env python3
"""
Single Song Inference Script
============================

This script demonstrates how to run inference on a single song
using the trained transformer model.
"""

from main import predict_single_song
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py <path_to_audio_file>")
        print("Example: python inference.py 'my_song.mp3'")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found!")
        sys.exit(1)
    
    print(f"Running inference on: {audio_path}")
    print("=" * 50)
    
    try:
        result = predict_single_song(audio_path)
        
        # Print results
        song_name = list(result.keys())[0]
        annotations = result[song_name]["annotations"]
        
        print(f"\n🎵 Song: {song_name}")
        print(f"📊 Found {len(annotations)} segments:")
        print()
        
        for i, segment in enumerate(annotations, 1):
            start_time = float(segment["start_time"])
            end_time = float(segment["end_time"])
            section = segment["section"]
            duration = end_time - start_time
            
            print(f"{i:2d}. {section:8s} | {start_time:6.1f}s - {end_time:6.1f}s | Duration: {duration:5.1f}s")
        
        print(f"\n✅ Inference complete! Found {len(annotations)} structural segments.")
        
    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
