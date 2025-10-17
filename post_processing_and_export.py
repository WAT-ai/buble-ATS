import numpy as np
import json
import csv
from typing import List, Tuple, Dict, Any
from label_alignment import LABEL_CLASSES
from json_preprocessing import CANONICAL_LABELS

def collapse_predictions(y_pred: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Collapse consecutive identical predictions into (start, end, label_idx) segments.
    """
    segments = []
    last_label = y_pred[0]
    start = 0
    for i in range(1, len(y_pred)):
        if y_pred[i] != last_label:
            segments.append((start, i, last_label))
            start = i
            last_label = y_pred[i]
    segments.append((start, len(y_pred), last_label))
    return segments

def segments_to_timestamps(segments: List[Tuple[int, int, int]], hop_length: int, sr: int) -> List[Dict[str, Any]]:
    out = []
    min_duration = 2.0  # seconds
    for start, end, label_idx in segments:
        label = LABEL_CLASSES[label_idx]
        # Only include canonical section labels (not 'unknown', not 'silence' if you want to filter that too)
        if label not in CANONICAL_LABELS:
            continue
        seg_start = float(start * hop_length / sr)
        seg_end = float(end * hop_length / sr)
        if (seg_end - seg_start) < min_duration:
            continue
        out.append({
            "start": seg_start,
            "end": seg_end,
            "label": label
        })
    return out

def export_json(segments: List[Dict[str, Any]], out_path: str):
    with open(out_path, 'w') as f:
        json.dump(segments, f, indent=2)

def export_csv(segments: List[Dict[str, Any]], out_path: str):
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['start', 'end', 'label'])
        writer.writeheader()
        for seg in segments:
            writer.writerow(seg)

def export_html(segments: List[Dict[str, Any]], out_path: str):
    html = '<div role="list">\n'
    for seg in segments:
        html += f'<div role="listitem" aria-label="{seg["label"]}" aria-valuenow="{seg["start"]}" aria-valuemax="{seg["end"]}">{seg["label"]}: {seg["start"]:.2f}–{seg["end"]:.2f}s</div>\n'
    html += '</div>'
    with open(out_path, 'w') as f:
        f.write(html)

def export_salami_format(segments: List[Dict[str, Any]], salami_id: str, out_path: str):
    """
    Export segments in SALAMI format with salami_id and section field.
    """
    output_data = {
        salami_id: {
            "salami_id": int(salami_id),
            "annotations": []
        }
    }
    
    for seg in segments:
        annotation = {
            "start_time": f"{seg['start']:.9f}",
            "end_time": f"{seg['end']:.9f}",
            "section": seg['label']
        }
        output_data[salami_id]["annotations"].append(annotation)
    
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)