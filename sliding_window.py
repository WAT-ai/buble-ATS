"""
Sliding Window Post-Processing for Music Structure Analysis
=========================================================

This module implements sliding window post-processing to improve temporal consistency
and correct isolated misclassifications in frame-level predictions.

The technique uses a 7-frame sliding window with majority voting to smooth out
predictions and reduce fragmented segments that cause poor F1 scores.

Based on the reference image showing three correction strategies:
1. Simple predecessor correction: i = i-1
2. Window-based correction with Y function: Y(i+1,i+2,i+3,i+4) = i-1  
3. Window-based correction with E function: E(i+1,i+2,i+3,i+4) = i-1

This implementation uses majority voting within a 7-frame window.
"""

import numpy as np
from typing import List, Tuple
from collections import Counter


def sliding_window_majority_vote(predictions: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Apply sliding window majority voting to frame-level predictions.
    
    This function corrects isolated misclassifications by considering the local
    context within a sliding window and applying majority voting.
    
    Args:
        predictions: Array of predicted label indices for each frame
        window_size: Size of the sliding window (default: 7)
        
    Returns:
        Smoothed predictions with corrected isolated misclassifications
        
    Example:
        Original:  [0, 0, 1, 2, 1, 1, 1]  # 2 is isolated between 1s
        Corrected: [0, 0, 1, 1, 1, 1, 1]  # 2 corrected to 1 by majority vote
    """
    
    if len(predictions) < window_size:
        # If sequence is shorter than window, return original
        return predictions.copy()
    
    smoothed = predictions.copy()
    half_window = window_size // 2
    
    # Apply sliding window majority voting
    for i in range(len(predictions)):
        # Define window boundaries
        start = max(0, i - half_window)
        end = min(len(predictions), i + half_window + 1)
        
        # Extract window
        window = predictions[start:end]
        
        # Apply majority voting
        if len(window) > 0:
            # Count occurrences of each label in the window
            label_counts = Counter(window)
            # Get the most common label
            majority_label = label_counts.most_common(1)[0][0]
            
            # Only change if the current prediction differs from majority
            # and the majority has at least 2 votes (not just a tie)
            if predictions[i] != majority_label and label_counts[majority_label] >= 2:
                smoothed[i] = majority_label
    
    return smoothed


def apply_temporal_smoothing(predictions: np.ndarray, 
                           window_size: int = 7,
                           min_segment_length: int = 3) -> np.ndarray:
    """
    Apply comprehensive temporal smoothing to predictions.
    
    This function combines sliding window majority voting with additional
    temporal consistency checks to create smoother, more coherent segments.
    
    Args:
        predictions: Array of predicted label indices for each frame
        window_size: Size of the sliding window for majority voting
        min_segment_length: Minimum length for a segment to be considered valid
        
    Returns:
        Temporally smoothed predictions
    """
    
    # Step 1: Apply sliding window majority voting
    smoothed = sliding_window_majority_vote(predictions, window_size)
    
    # Step 2: Remove very short segments (less than min_segment_length)
    smoothed = remove_short_segments(smoothed, min_segment_length)
    
    # Step 3: Apply additional smoothing for edge cases
    smoothed = edge_case_smoothing(smoothed)
    
    return smoothed


def remove_short_segments(predictions: np.ndarray, min_length: int = 3) -> np.ndarray:
    """
    Remove segments that are shorter than the minimum length.
    
    Short segments are replaced with the label of their longer neighbor.
    
    Args:
        predictions: Array of predicted label indices
        min_length: Minimum segment length to keep
        
    Returns:
        Predictions with short segments removed
    """
    
    if len(predictions) < min_length:
        return predictions.copy()
    
    smoothed = predictions.copy()
    
    # Find segment boundaries
    segment_starts = [0]
    segment_labels = [predictions[0]]
    
    for i in range(1, len(predictions)):
        if predictions[i] != predictions[i-1]:
            segment_starts.append(i)
            segment_labels.append(predictions[i])
    
    # Process each segment
    for i in range(len(segment_starts)):
        start = segment_starts[i]
        end = segment_starts[i+1] if i+1 < len(segment_starts) else len(predictions)
        length = end - start
        
        # If segment is too short, replace with neighbor
        if length < min_length:
            # Choose the label of the longer neighbor
            prev_length = start - (segment_starts[i-1] if i > 0 else 0)
            next_length = (segment_starts[i+1] if i+1 < len(segment_starts) else len(predictions)) - end
            
            if prev_length > next_length and i > 0:
                # Use previous segment's label
                replacement_label = segment_labels[i-1]
            elif i+1 < len(segment_labels):
                # Use next segment's label
                replacement_label = segment_labels[i+1]
            else:
                # Use previous segment's label as fallback
                replacement_label = segment_labels[i-1] if i > 0 else predictions[start]
            
            # Replace the short segment
            smoothed[start:end] = replacement_label
    
    return smoothed


def edge_case_smoothing(predictions: np.ndarray) -> np.ndarray:
    """
    Handle edge cases in temporal smoothing.
    
    This function addresses specific edge cases that might not be caught
    by the sliding window approach.
    
    Args:
        predictions: Array of predicted label indices
        
    Returns:
        Predictions with edge cases smoothed
    """
    
    smoothed = predictions.copy()
    
    # Handle single-frame differences at segment boundaries
    for i in range(1, len(predictions) - 1):
        # If current frame is different from both neighbors
        if (smoothed[i] != smoothed[i-1] and 
            smoothed[i] != smoothed[i+1] and 
            smoothed[i-1] == smoothed[i+1]):
            # Replace with the common neighbor label
            smoothed[i] = smoothed[i-1]
    
    return smoothed


def analyze_smoothing_improvement(original: np.ndarray, 
                                smoothed: np.ndarray,
                                label_names: List[str]) -> dict:
    """
    Analyze the improvement from temporal smoothing.
    
    Args:
        original: Original predictions
        smoothed: Smoothed predictions
        label_names: List of label names for reporting
        
    Returns:
        Dictionary with smoothing statistics
    """
    
    # Count changes
    changes = np.sum(original != smoothed)
    total_frames = len(original)
    change_percentage = (changes / total_frames) * 100
    
    # Count segments before and after
    original_segments = count_segments(original)
    smoothed_segments = count_segments(smoothed)
    
    # Calculate average segment length
    original_avg_length = total_frames / original_segments if original_segments > 0 else 0
    smoothed_avg_length = total_frames / smoothed_segments if smoothed_segments > 0 else 0
    
    stats = {
        'total_frames': total_frames,
        'frames_changed': changes,
        'change_percentage': change_percentage,
        'original_segments': original_segments,
        'smoothed_segments': smoothed_segments,
        'segment_reduction': original_segments - smoothed_segments,
        'original_avg_length': original_avg_length,
        'smoothed_avg_length': smoothed_avg_length,
        'length_improvement': smoothed_avg_length - original_avg_length
    }
    
    return stats


def count_segments(predictions: np.ndarray) -> int:
    """
    Count the number of segments in a prediction sequence.
    
    Args:
        predictions: Array of predicted label indices
        
    Returns:
        Number of segments
    """
    
    if len(predictions) == 0:
        return 0
    
    segments = 1
    for i in range(1, len(predictions)):
        if predictions[i] != predictions[i-1]:
            segments += 1
    
    return segments


def print_smoothing_stats(stats: dict):
    """
    Print smoothing statistics in a readable format.
    
    Args:
        stats: Statistics dictionary from analyze_smoothing_improvement
    """
    
    print("\n=== Temporal Smoothing Results ===")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Frames changed: {stats['frames_changed']} ({stats['change_percentage']:.1f}%)")
    print(f"Original segments: {stats['original_segments']}")
    print(f"Smoothed segments: {stats['smoothed_segments']}")
    print(f"Segment reduction: {stats['segment_reduction']}")
    print(f"Original avg length: {stats['original_avg_length']:.1f} frames")
    print(f"Smoothed avg length: {stats['smoothed_avg_length']:.1f} frames")
    print(f"Length improvement: {stats['length_improvement']:.1f} frames")
    print("=" * 40)
