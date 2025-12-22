#!/usr/bin/env python3
"""
Diagnostic version of run_songformer.py with full error reporting
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path

# Add SongFormer to path
SONGFORMER_PATH = Path(__file__).parent.parent / "open-source-models" / "songformer" / "src"
sys.path.insert(0, str(SONGFORMER_PATH))

def check_songformer_installation():
    """Verify SongFormer is properly installed"""
    print("="*60)
    print("CHECKING SONGFORMER INSTALLATION")
    print("="*60)
    
    songformer_root = SONGFORMER_PATH.parent
    
    # Check key files
    checks = {
        "Repository root": songformer_root.exists(),
        "src/ directory": SONGFORMER_PATH.exists(),
        "SongFormer/ directory": (SONGFORMER_PATH / "SongFormer").exists(),
        "infer.sh script": (SONGFORMER_PATH / "SongFormer" / "infer.sh").exists(),
    }
    
    all_good = True
    for item, exists in checks.items():
        status = "✓" if exists else "✗"
        print(f"{status} {item}")
        if not exists:
            all_good = False
    
    # Check for model files
    print("\nSearching for model files...")
    model_files = list(songformer_root.glob("**/*.safetensors")) + \
                  list(songformer_root.glob("**/*.pt")) + \
                  list(songformer_root.glob("**/*.pth"))
    
    if model_files:
        print(f"✓ Found {len(model_files)} model file(s):")
        for f in model_files[:5]:
            print(f"  - {f.relative_to(songformer_root)}")
        if len(model_files) > 5:
            print(f"  ... and {len(model_files) - 5} more")
    else:
        print("✗ No model files found!")
        print("  You may need to download pretrained models.")
        all_good = False
    
    print()
    return all_good

def convert_msa_to_json(msa_file, output_json):
    """Convert txt to JSON format."""
    try:
        segments = []
        
        with open(msa_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    timestamp, label = parts[0], parts[1]
                    segments.append({
                        "timestamp": float(timestamp),
                        "label": label
                    })
        
        if not segments:
            print(f"✗ Warning: {msa_file.name} has no valid segments")
            return False
        
        output_data = {
            "file": str(msa_file),
            "segments": segments
        }
        
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Converted {msa_file.name} → {output_json.name} ({len(segments)} segments)")
        return True
        
    except Exception as e:
        print(f"✗ Error converting {msa_file.name}: {e}")
        return False

def run_inference(audio_dir, output_dir, convert_to_json=True):
    """Run SongFormer inference."""
    
    print("\n" + "="*60)
    print("SONGFORMER INFERENCE")
    print("="*60 + "\n")
    
    # Check installation first
    if not check_songformer_installation():
        print("✗ SongFormer installation incomplete!")
        print("\nPlease ensure:")
        print("1. Submodule is initialized: git submodule update --init --recursive")
        print("2. Model checkpoints are downloaded")
        return False
    
    audio_dir = Path(audio_dir).absolute()
    output_dir = Path(output_dir).absolute()
    
    if not audio_dir.exists():
        print(f"✗ Audio directory not found: {audio_dir}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
        audio_files.extend(audio_dir.glob(ext))
    
    if not audio_files:
        print(f"✗ No audio files found in {audio_dir}")
        return False
    
    print(f"✓ Found {len(audio_files)} audio files")
    print(f"✓ Output directory: {output_dir}\n")
    
    # Create input scp file
    scp_file = output_dir / "input_files.scp"
    with open(scp_file, 'w') as f:
        for audio in audio_files:
            f.write(f"{audio}\n")
    
    print(f"✓ Created input list: {scp_file}")
    print(f"  (Contains {len(audio_files)} files)\n")
    
    # Locate the inference script
    infer_script = SONGFORMER_PATH / "SongFormer" / "infer.sh"
    
    if not infer_script.exists():
        print(f"✗ Inference script not found: {infer_script}")
        print("\nSearching for alternative scripts...")
        alternatives = list(SONGFORMER_PATH.glob("**/*infer*"))
        if alternatives:
            print("Found these possibilities:")
            for alt in alternatives:
                print(f"  - {alt}")
        return False
    
    print(f"✓ Found inference script: {infer_script}\n")
    
    # Run SongFormer inference with proper error capture
    print("="*60)
    print("RUNNING SONGFORMER")
    print("="*60 + "\n")
    
    # Change to SongFormer directory
    original_dir = os.getcwd()
    songformer_dir = infer_script.parent
    
    try:
        os.chdir(songformer_dir)
        print(f"Working directory: {songformer_dir}\n")
        
        # Construct command
        cmd = [
            'bash',
            'infer.sh',
            '-i', str(scp_file),
            '-o', str(output_dir),
            '-gn', '1',
            '-tn', '1'
        ]
        
        print(f"Command: {' '.join(cmd)}\n")
        print("-" * 60)
        
        # Run with full output capture
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Print stdout
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        # Print stderr
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print("-" * 60)
        
        if result.returncode != 0:
            print(f"\n✗ SongFormer failed with exit code {result.returncode}")
            return False
        
        print("\n✓ SongFormer completed\n")
        
    except Exception as e:
        print(f"\n✗ Error running SongFormer: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)
    
    # Check for output files
    print("="*60)
    print("CHECKING OUTPUT")
    print("="*60 + "\n")
    
    txt_files = [f for f in output_dir.glob("*.txt") if f.name != "input_files.scp"]
    
    if not txt_files:
        print("✗ No output .txt files were created!")
        print(f"\nExpected output files in: {output_dir}")
        print("\nListing all files in output directory:")
        all_files = list(output_dir.glob("*"))
        if all_files:
            for f in all_files:
                size = f.stat().st_size if f.is_file() else "DIR"
                print(f"  - {f.name} ({size} bytes)" if isinstance(size, int) else f"  - {f.name} (directory)")
        else:
            print("  (empty)")
        
        print("\n✗ Inference failed - no outputs generated")
        return False
    
    print(f"✓ Created {len(txt_files)} output .txt files:")
    for f in txt_files[:5]:
        size = f.stat().st_size
        print(f"  - {f.name} ({size} bytes)")
    if len(txt_files) > 5:
        print(f"  ... and {len(txt_files) - 5} more")
    
    # Convert to JSON
    if convert_to_json:
        print("\n" + "="*60)
        print("CONVERTING TO JSON")
        print("="*60 + "\n")
        
        json_dir = output_dir / "json"
        json_dir.mkdir(exist_ok=True)
        
        converted = 0
        for msa_file in txt_files:
            json_file = json_dir / f"{msa_file.stem}.json"
            if convert_msa_to_json(msa_file, json_file):
                converted += 1
        
        print(f"\n✓ Converted {converted}/{len(txt_files)} files to JSON")
        
        if converted == 0:
            print("✗ JSON conversion failed!")
            return False
    
    print("\n" + "="*60)
    print("✓ SUCCESS!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    if convert_to_json:
        print(f"JSON files: {output_dir / 'json'}")
    print()
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Run SongFormer inference with full diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "audio_dir",
        type=str,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="./songformer_output",
        help="Output directory (default: ./songformer_output)"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip JSON conversion"
    )
    
    args = parser.parse_args()
    
    success = run_inference(
        args.audio_dir,
        args.output_dir,
        convert_to_json=not args.no_json
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
    