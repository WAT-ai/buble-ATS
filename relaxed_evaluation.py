"""
Relaxed F1 Evaluation for Music Structure Analysis
=================================================

This module implements relaxed F1 evaluation with timing tolerance to address
the strict timing requirements that cause 0.000 F1 scores.

The standard F1 evaluation requires exact time overlap, which is too strict
for continuous audio segmentation where small timing differences are acceptable.

This implementation provides:
- Tolerance-based overlap calculation (±5 seconds configurable)
- Relaxed precision, recall, and F1 calculation
- Detailed evaluation statistics
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from configs import F1_TOLERANCE_SECONDS


def calculate_relaxed_f1(predicted_segments: List[Dict[str, Any]], 
                        ground_truth_segments: List[Dict[str, Any]], 
                        tolerance: float = F1_TOLERANCE_SECONDS) -> Dict[str, float]:
    """
    Calculate relaxed F1 score with timing tolerance.
    
    Args:
        predicted_segments: List of predicted segments with 'start', 'end', 'label'
        ground_truth_segments: List of ground truth segments with 'start', 'end', 'label'
        tolerance: Time tolerance in seconds (default: 5.0)
        
    Returns:
        Dictionary with precision, recall, F1, and detailed statistics
    """
    
    if not predicted_segments and not ground_truth_segments:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
    
    if not predicted_segments:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(ground_truth_segments)}
    
    if not ground_truth_segments:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(predicted_segments), 'fn': 0}
    
    # Calculate overlaps with tolerance
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    # Track which segments have been matched
    gt_matched = [False] * len(ground_truth_segments)
    
    # For each predicted segment, find best matching ground truth
    for pred_seg in predicted_segments:
        best_overlap = 0
        best_gt_idx = -1
        
        for gt_idx, gt_seg in enumerate(ground_truth_segments):
            if gt_matched[gt_idx]:
                continue
                
            # Calculate relaxed overlap
            overlap = calculate_relaxed_overlap(pred_seg, gt_seg, tolerance)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_gt_idx = gt_idx
        
        # If we found a good match (overlap > 0)
        if best_overlap > 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    # Count unmatched ground truth segments as false negatives
    fn = sum(1 for matched in gt_matched if not matched)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tolerance': tolerance
    }


def calculate_relaxed_overlap(pred_seg: Dict[str, Any], 
                            gt_seg: Dict[str, Any], 
                            tolerance: float) -> float:
    """
    Calculate overlap between predicted and ground truth segments with tolerance.
    
    Args:
        pred_seg: Predicted segment with 'start', 'end', 'label'
        gt_seg: Ground truth segment with 'start', 'end', 'label'
        tolerance: Time tolerance in seconds
        
    Returns:
        Overlap score (0.0 to 1.0)
    """
    
    # Extract times
    pred_start = pred_seg['start']
    pred_end = pred_seg['end']
    gt_start = gt_seg['start']
    gt_end = gt_seg['end']
    
    # Apply tolerance to predicted segment
    relaxed_pred_start = pred_start - tolerance
    relaxed_pred_end = pred_end + tolerance
    
    # Calculate overlap
    overlap_start = max(relaxed_pred_start, gt_start)
    overlap_end = min(relaxed_pred_end, gt_end)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    # Calculate overlap ratio
    overlap_duration = overlap_end - overlap_start
    gt_duration = gt_end - gt_start
    
    if gt_duration <= 0:
        return 0.0
    
    # Check label match
    label_match = pred_seg['label'] == gt_seg['label']
    
    # Return overlap ratio weighted by label match
    overlap_ratio = overlap_duration / gt_duration
    
    if label_match:
        return overlap_ratio
    else:
        # Reduce score for label mismatch
        return overlap_ratio * 0.5


def calculate_strict_f1(predicted_segments: List[Dict[str, Any]], 
                       ground_truth_segments: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate strict F1 score (original evaluation method).
    
    Args:
        predicted_segments: List of predicted segments
        ground_truth_segments: List of ground truth segments
        
    Returns:
        Dictionary with strict F1 metrics
    """
    
    if not predicted_segments and not ground_truth_segments:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
    
    if not predicted_segments:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(ground_truth_segments)}
    
    if not ground_truth_segments:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(predicted_segments), 'fn': 0}
    
    # Calculate strict overlaps (no tolerance)
    tp = 0
    fp = 0
    fn = 0
    
    gt_matched = [False] * len(ground_truth_segments)
    
    for pred_seg in predicted_segments:
        best_overlap = 0
        best_gt_idx = -1
        
        for gt_idx, gt_seg in enumerate(ground_truth_segments):
            if gt_matched[gt_idx]:
                continue
                
            # Calculate strict overlap (no tolerance)
            overlap = calculate_strict_overlap(pred_seg, gt_seg)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_gt_idx = gt_idx
        
        if best_overlap > 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    fn = sum(1 for matched in gt_matched if not matched)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tolerance': 0.0
    }


def calculate_strict_overlap(pred_seg: Dict[str, Any], gt_seg: Dict[str, Any]) -> float:
    """
    Calculate strict overlap between segments (no tolerance).
    
    Args:
        pred_seg: Predicted segment
        gt_seg: Ground truth segment
        
    Returns:
        Overlap score (0.0 to 1.0)
    """
    
    pred_start = pred_seg['start']
    pred_end = pred_seg['end']
    gt_start = gt_seg['start']
    gt_end = gt_seg['end']
    
    # Calculate overlap
    overlap_start = max(pred_start, gt_start)
    overlap_end = min(pred_end, gt_end)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_duration = overlap_end - overlap_start
    gt_duration = gt_end - gt_start
    
    if gt_duration <= 0:
        return 0.0
    
    # Check label match
    label_match = pred_seg['label'] == gt_seg['label']
    
    overlap_ratio = overlap_duration / gt_duration
    
    if label_match:
        return overlap_ratio
    else:
        return overlap_ratio * 0.5


def compare_evaluation_methods(predicted_segments: List[Dict[str, Any]], 
                            ground_truth_segments: List[Dict[str, Any]], 
                            tolerance: float = F1_TOLERANCE_SECONDS) -> Dict[str, Any]:
    """
    Compare strict vs relaxed F1 evaluation methods.
    
    Args:
        predicted_segments: List of predicted segments
        ground_truth_segments: List of ground truth segments
        tolerance: Time tolerance for relaxed evaluation
        
    Returns:
        Dictionary with comparison results
    """
    
    strict_results = calculate_strict_f1(predicted_segments, ground_truth_segments)
    relaxed_results = calculate_relaxed_f1(predicted_segments, ground_truth_segments, tolerance)
    
    return {
        'strict': strict_results,
        'relaxed': relaxed_results,
        'improvement': {
            'f1_delta': relaxed_results['f1'] - strict_results['f1'],
            'precision_delta': relaxed_results['precision'] - strict_results['precision'],
            'recall_delta': relaxed_results['recall'] - strict_results['recall']
        }
    }


def print_evaluation_comparison(results: Dict[str, Any]):
    """
    Print comparison between strict and relaxed evaluation methods.
    
    Args:
        results: Results from compare_evaluation_methods
    """
    
    strict = results['strict']
    relaxed = results['relaxed']
    improvement = results['improvement']
    
    print(f"\n=== F1 Evaluation Comparison ===")
    print(f"Tolerance: ±{relaxed['tolerance']:.1f} seconds")
    print(f"")
    print(f"Strict F1:    {strict['f1']:.3f} (P: {strict['precision']:.3f}, R: {strict['recall']:.3f})")
    print(f"Relaxed F1:   {relaxed['f1']:.3f} (P: {relaxed['precision']:.3f}, R: {relaxed['recall']:.3f})")
    print(f"")
    print(f"Improvement:  F1: {improvement['f1_delta']:+.3f}, P: {improvement['precision_delta']:+.3f}, R: {improvement['recall_delta']:+.3f}")
    print(f"")
    print(f"Strict:  TP={strict['tp']}, FP={strict['fp']}, FN={strict['fn']}")
    print(f"Relaxed: TP={relaxed['tp']}, FP={relaxed['fp']}, FN={relaxed['fn']}")
    print("=" * 40)
