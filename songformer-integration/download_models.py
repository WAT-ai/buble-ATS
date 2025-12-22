#!/usr/bin/env python3
"""
Download SongFormer pretrained models
"""

import os
import sys
from pathlib import Path
import urllib.request
import json

# Model URLs from HuggingFace
MODELS = {
    "SongFormer": {
        "url": "https://huggingface.co/ASLP-lab/SongFormer/resolve/main/SongFormer.safetensors",
        "path": "ckpts/SongFormer.safetensors",
        "size_mb": "~600MB"
    },
    "MusicFM_pretrained": {
        "url": "https://huggingface.co/ASLP-lab/SongFormer/resolve/main/pretrained_msd.pt",
        "path": "ckpts/MusicFM/pretrained_msd.pt",
        "size_mb": "~600MB"
    },
    "MusicFM_stats": {
        "url": "https://huggingface.co/ASLP-lab/SongFormer/resolve/main/msd_stats.json",
        "path": "ckpts/MusicFM/msd_stats.json",
        "size_mb": "~1KB"
    }
}

def download_file(url, output_path, description):
    """Download a file with progress bar"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"✓ {description} already exists: {output_path}")
        return True
    
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")
    
    try:
        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded * 100 / total_size, 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
        
        urllib.request.urlretrieve(url, output_path, progress)
        print()  # New line after progress
        print(f"✓ Downloaded {description}")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to download {description}: {e}")
        return False

def main():
    print("="*60)
    print("SONGFORMER MODEL DOWNLOADER")
    print("="*60)
    print()
    
    # Navigate to SongFormer src directory
    script_dir = Path(__file__).parent
    songformer_src = script_dir.parent / "open-source-models" / "songformer" / "src" / "SongFormer"
    
    if not songformer_src.exists():
        print(f"✗ SongFormer directory not found: {songformer_src}")
        print("\nMake sure you've initialized the git submodule:")
        print("  git submodule update --init --recursive")
        sys.exit(1)
    
    print(f"SongFormer location: {songformer_src}")
    print()
    
    os.chdir(songformer_src)
    
    print("This will download the following models:")
    total_size = 0
    for name, info in MODELS.items():
        print(f"  - {name}: {info['size_mb']}")
    print("\nTotal: ~1.2GB")
    print()
    
    # Ask for confirmation
    response = input("Continue with download? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Download cancelled.")
        sys.exit(0)
    
    print()
    
    # Download each model
    success_count = 0
    for name, info in MODELS.items():
        if download_file(info['url'], info['path'], name):
            success_count += 1
        print()
    
    print("="*60)
    if success_count == len(MODELS):
        print("✓ ALL MODELS DOWNLOADED SUCCESSFULLY!")
        print("="*60)
        print()
        print("You can now run:")
        print("  python run_songformer.py ../music-files -o ../output/songformer_results")
        sys.exit(0)
    else:
        print(f"⚠ PARTIAL SUCCESS: {success_count}/{len(MODELS)} models downloaded")
        print("="*60)
        print()
        print("Some downloads failed. You may need to:")
        print("1. Check your internet connection")
        print("2. Download manually from:")
        print("   https://huggingface.co/ASLP-lab/SongFormer/tree/main")
        sys.exit(1)

if __name__ == "__main__":
    main()

    