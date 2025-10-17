import json
import re
from typing import List, Tuple, Dict, Any

# Canonical label mapping (should match rest of pipeline)
CANONICAL_LABELS = {
    'Intro': ['intro', 'introduction', 'start'],
    'Verse': ['verse', 'v', 'vs'],
    'Chorus': ['chorus', 'hook', 'refrain'],
    'Bridge': ['bridge', 'middle8', 'break'],
    'Outro': ['outro', 'end', 'ending'],
    'Silence': ['silence'],
    'Solo': ['solo'],
    'Empty': ['empty', 'instrumental', 'break']
}

LABEL_CLASSES = list(CANONICAL_LABELS.keys()) + ['unknown']


def canonicalize_label(label: str) -> str:
    tokens = re.split(r'[(),]', label)
    for token in tokens:
        token = token.strip().lower()
        token_clean = re.sub(r'[^a-z\s]', '', token)
        for canon, syns in CANONICAL_LABELS.items():
            if token_clean in syns:
                return canon
    return 'unknown'


def process_json_annotations(json_path: str) -> Dict[str, List[Tuple[float, float, str]]]:
    """
    Loads combined_data.json and returns a dict:
    { track_id: [ (start, end, canonical_label), ... ] }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    track_segments = {}
    for track_id, track in data.items():
        annots = track['annotations']
        segments = []
        for i, annot in enumerate(annots):
            # Get start/end (assume fields exist, else skip)
            start = annot.get('start_time', None)
            end = annot.get('end_time', None)
            label = annot.get('section', '')  # Use 'section' field instead of 'full_label'
            if start is None or end is None or start == '' or end == '':
                continue
            try:
                start_float = float(start)
                end_float = float(end)
                canon_label = canonicalize_label(label)
                segments.append((start_float, end_float, canon_label))
            except (ValueError, TypeError):
                continue
        track_segments[track_id] = segments
    return track_segments

# Example usage:
# segments_dict = process_json_annotations('training_set/json/combined_data.json')
# print(segments_dict['3'][:5])
