# Map SongFormer's generic labels to a EDM-specific structure

import json
from pathlib import Path

EDM_LABEL_MAP = {
    "intro": "intro",
    "verse": "verse",
    "pre-chorus": "buildup",
    "chorus": "chorus",
    "bridge": "breakdown",
    "inst": "breakdown", # could also be a new category like transition but likely not needed
    "outro": "outro",
    "silence": "silence",
    "end": "end",
    "NO_LABEL": "NO_LABEL"
}

def remap_labels(input_json, output_json):
    """Remap SongFormer labels to EDM labels."""
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Remap segments
    for segment in data['segments']:
        original_label = segment['label']
        segment['label'] = EDM_LABEL_MAP.get(original_label, original_label)
        segment['original_label'] = original_label
    
    # Save remapped version
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Remapped {input_json} → {output_json}")

def process_directory(json_dir):
    """Process all JSON files in directory."""
    json_dir = Path(json_dir)
    edm_dir = json_dir.parent / "edm_labels"
    edm_dir.mkdir(exist_ok=True)
    
    for json_file in json_dir.glob("*.json"):
        output_file = edm_dir / json_file.name
        remap_labels(json_file, output_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python edm_label_mapper.py <json_directory>")
        sys.exit(1)
    
    process_directory(sys.argv[1])
