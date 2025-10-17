"""
Label Alignment for Music Structure Analysis
===========================================

This module handles the alignment of ground truth segment annotations to
spectrogram frames for training the transformer model.

CRITICAL ISSUE: This module contributes to the class imbalance problem
by defaulting unannotated frames to 'unknown' and having many frames
outside annotated segments.

PROBLEM ANALYSIS:
- 70-80% of frames are labeled as 'unknown' or 'Empty'
- Only 2-10% of frames get actual structural labels
- This creates severe class imbalance that the model cannot overcome

SOLUTIONS NEEDED:
1. Better segment coverage (more comprehensive annotations)
2. Interpolation between segments
3. Confidence weighting for uncertain regions
4. Different labeling strategy (e.g., majority voting)
"""

import numpy as np
from typing import List, Tuple
from json_preprocessing import LABEL_CLASSES


def align_labels_to_frames(
    segments: List[Tuple[float, float, str]], 
    n_frames: int, 
    hop_length: int, 
    sr: int
) -> np.ndarray:
    """
    Assign a label index to each spectrogram frame.
    
    CRITICAL ISSUE: This function creates severe class imbalance by:
    1. Defaulting unannotated frames to 'unknown'
    2. Having many frames outside annotated segments
    3. Not handling gaps between segments properly
    
    This results in 70-80% of frames being labeled as 'unknown' or 'Empty',
    causing the model to learn to predict the majority class.
    
    Args:
        segments: List of (start_time, end_time, canonical_label) tuples
        n_frames: Number of spectrogram frames
        hop_length: Hop length in samples between frames
        sr: Sample rate in Hz
        
    Returns:
        Array of label indices for each frame
        
    IMPROVEMENTS NEEDED:
    1. Interpolate labels between segments
    2. Use majority voting for overlapping segments
    3. Add confidence weighting
    4. Better handling of gaps between segments
    """
    
    # Calculate time for each frame
    frame_times = np.arange(n_frames) * hop_length / sr
    unknown_idx = LABEL_CLASSES.index('unknown')
    
    # Initialize all frames as 'unknown'
    # ISSUE: This creates the class imbalance problem
    y = np.full(n_frames, unknown_idx, dtype=np.int32)
    
    # Assign labels to frames within annotated segments
    for i, frame_time in enumerate(frame_times):
        for start_time, end_time, label in segments:
            if start_time <= frame_time < end_time:
                if label in LABEL_CLASSES:
                    y[i] = LABEL_CLASSES.index(label)
                else:
                    y[i] = unknown_idx
                break  # Take first matching segment
    
    return y


def align_labels_with_interpolation(
    segments: List[Tuple[float, float, str]], 
    n_frames: int, 
    hop_length: int, 
    sr: int
) -> np.ndarray:
    """
    Improved label alignment with interpolation between segments.
    
    This is a placeholder for a better alignment strategy that would:
    1. Interpolate labels between segments
    2. Use majority voting for overlapping segments
    3. Add confidence weighting
    4. Better handle gaps between segments
    
    Args:
        segments: List of (start_time, end_time, canonical_label) tuples
        n_frames: Number of spectrogram frames
        hop_length: Hop length in samples between frames
        sr: Sample rate in Hz
        
    Returns:
        Array of label indices for each frame
    """
    
    # TODO: Implement improved alignment strategy
    # This would include:
    # - Interpolation between segments
    # - Majority voting for overlaps
    # - Confidence weighting
    # - Better gap handling
    
    pass


def get_label_statistics(y: np.ndarray) -> dict:
    """
    Get statistics about label distribution.
    
    Args:
        y: Array of label indices
        
    Returns:
        Dictionary with label statistics
    """
    unique, counts = np.unique(y, return_counts=True)
    total_frames = len(y)
    
    stats = {}
    for label_idx, count in zip(unique, counts):
        label_name = LABEL_CLASSES[label_idx]
        percentage = count / total_frames * 100
        stats[label_name] = {
            'count': int(count),
            'percentage': percentage
        }
    
    return stats