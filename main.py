"""
Music Structure Analysis using Transformer Neural Networks
=========================================================

This module implements a transformer-based approach for automatic music structure analysis,
specifically for segmenting songs into structural components like Intro, Verse, Chorus, etc.

PROBLEM ANALYSIS:
================
The current implementation has several critical issues preventing successful training:

1. SEVERE CLASS IMBALANCE:
   - 70-80% of frames are labeled as "Empty" 
   - Only 2-10% are actual structural labels (Intro, Verse, Chorus)
   - Model learns to predict majority class (Empty) to minimize loss

2. TIMING MISALIGNMENT:
   - Predicted segments: 2-3 seconds each
   - Ground truth segments: 8+ seconds each
   - F1 calculation requires exact timing matches → 0.000 F1 scores

3. SEGMENT GRANULARITY MISMATCH:
   - Model predicts many small segments
   - Ground truth has fewer, longer segments
   - Different temporal resolution between prediction and ground truth

4. EVALUATION METRIC ISSUES:
   - F1 score requires exact time overlap
   - No tolerance for timing differences
   - Binary evaluation too strict for continuous audio

SOLUTIONS NEEDED:
================
1. Class balancing (weighted loss, data augmentation, undersampling)
2. Temporal smoothing/post-processing
3. Relaxed evaluation metrics (tolerance windows)
4. Better feature engineering
5. Different model architecture (CRF, CTC loss)

USAGE:
======
python main.py  # Run test mode (2 epochs, 2 iterations)
"""

import os
import numpy as np
import glob
import re
import json
from configs import (
    COMBINED_JSON_PATH,
    AUDIO_GLOB,
    MODEL_PATH,
    PREDICTED_OUTPUT_PATH,
    SAMPLE_RATE,
    N_MELS,
    HOP_LENGTH,
    N_FFT,
    TEST_MODE,
    EPOCHS_TEST,
    EPOCHS_FULL,
    BATCH_TEST,
    BATCH_FULL,
    LEARNING_RATE,
    LABEL_SMOOTHING,
    SMOOTH_WINDOW,
    MIN_SEGMENT_FRAMES,
    MAX_ITERS_TEST,
    MAX_ITERS_FULL,
    TARGET_F1_TEST,
    TARGET_F1_FULL,
    F1_TOLERANCE_SECONDS,
)
from typing import List, Tuple, Dict, Any

# Core modules
from audio_preprocessing import load_audio_mono, compute_logmel, normalize_features
from label_alignment import align_labels_to_frames
from json_preprocessing import process_json_annotations, LABEL_CLASSES, CANONICAL_LABELS
from train import train_model
from post_processing_and_export import collapse_predictions, segments_to_timestamps
from relaxed_evaluation import calculate_relaxed_f1
from sliding_window import apply_temporal_smoothing, analyze_smoothing_improvement, print_smoothing_stats


def load_training_data():
    """
    Load all 20 songs and their annotations for training.
    
    Returns:
        Tuple containing training data and metadata
    """
    print("Loading training data...")
    
    # Configuration
    sample_rate = SAMPLE_RATE
    n_mels = N_MELS
    hop_length = HOP_LENGTH
    
    # Load all annotated segments from combined_data.json
    segments_dict = process_json_annotations(COMBINED_JSON_PATH)
    
    # Discover available audio files and map to track IDs
    audio_files = sorted(glob.glob(AUDIO_GLOB))
    id_to_path = {}
    for audio_path in audio_files:
        match = re.search(r'salami_id_(\d+)\.mp3$', audio_path)
        if match:
            id_to_path[match.group(1)] = audio_path
    
    # Build dataset across all matching tracks
    X_list = []  # Audio features
    y_list = []  # Frame labels
    track_ids = []
    track_names = []
    
    for track_id, segments in segments_dict.items():
        if track_id not in id_to_path:
            print(f"Warning: No audio file found for track {track_id}")
            continue
            
        audio_path = id_to_path[track_id]
        track_name = f"song_{track_id}"  # Generic name for product use
        
        # Load and process audio
        y_audio = load_audio_mono(audio_path, sr=sample_rate)
        X_feat = compute_logmel(y_audio, sr=sample_rate, n_fft=N_FFT, 
                               hop_length=hop_length, n_mels=n_mels)
        X_feat = normalize_features(X_feat)
        
        # Align labels to spectrogram frames
        frame_labels = align_labels_to_frames(segments, X_feat.shape[0], 
                                            hop_length, sample_rate)
        
        # Debug: Show label distribution for first few tracks
        if len(track_ids) < 3:
            unique, counts = np.unique(frame_labels, return_counts=True)
            print(f"Track {track_id} label distribution:")
            for label_idx, count in zip(unique, counts):
                label_name = LABEL_CLASSES[label_idx]
                percentage = count/len(frame_labels)*100
                print(f"  {label_name}: {count} frames ({percentage:.1f}%)")
        
        X_list.append(X_feat)
        y_list.append(frame_labels)
        track_ids.append(track_id)
        track_names.append(track_name)
    
    return X_list, y_list, track_ids, track_names, segments_dict, sample_rate, hop_length


def train_transformer_model(test_mode=True):
    """
    Train the transformer model on all 20 songs.
    
    Args:
        test_mode: If True, use reduced parameters for quick testing
        
    Returns:
        Trained model and training data
    """
    print("Loading training data...")
    X_list, y_list, track_ids, track_names, segments_dict, sample_rate, hop_length = load_training_data()
    
    if len(X_list) == 0:
        raise RuntimeError('No matching audio files found for training.')
    
    print(f"Training on {len(X_list)} songs...")
    
    # Pad sequences to common length for batching
    max_len = max(x.shape[0] for x in X_list)
    n_features = X_list[0].shape[1]
    
    X_batch = np.zeros((len(X_list), max_len, n_features), dtype=np.float32)
    y_batch = np.full((len(y_list), max_len), fill_value=LABEL_CLASSES.index('unknown'), dtype=np.int32)
    
    for i, (xf, yl) in enumerate(zip(X_list, y_list)):
        seq_len = xf.shape[0]
        X_batch[i, :seq_len, :] = xf
        y_batch[i, :seq_len] = yl
    
    # Train model with appropriate parameters
    epochs = EPOCHS_TEST if test_mode else EPOCHS_FULL
    batch_size = BATCH_TEST if test_mode else BATCH_FULL
    model = train_model(X_batch, y_batch, batch_size=batch_size, epochs=epochs)
    
    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved as '{MODEL_PATH}'")
    
    # Verify file was created
    import os
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        print(f"Model file verified: {file_size} bytes")
    else:
        print("ERROR: Model file was not created!")
    
    return model, X_list, y_list, track_ids, track_names, segments_dict, sample_rate, hop_length


def predict_and_compare(model, X_list, track_ids, track_names, segments_dict, sample_rate, hop_length):
    """
    Predict on all songs and compare with ground truth.
    
    NOTE: This function currently produces 0.000 F1 scores due to:
    1. Severe class imbalance (70-80% Empty labels)
    2. Timing misalignment between predictions and ground truth
    3. Different segment granularity (model: 2-3s, GT: 8+s)
    4. Strict F1 evaluation requiring exact time overlap
    
    Args:
        model: Trained transformer model
        X_list: List of audio features for each track
        track_ids: List of track IDs
        track_names: List of track names
        segments_dict: Ground truth segments
        sample_rate: Audio sample rate
        hop_length: Spectrogram hop length
        
    Returns:
        Tuple of (predicted_data, mean_f1_score)
    """
    print("Making predictions...")
    
    predicted_data = {}
    total_f1 = 0
    valid_tracks = 0
    
    for i, (track_id, track_name) in enumerate(zip(track_ids, track_names)):
        print(f"Processing track {i+1}/{len(track_ids)}: {track_id}")
        
        # Get single song features
        xf = X_list[i]
        seq_len = xf.shape[0]
        
        # Predict single song
        X_single = xf[None, ...]  # Add batch dimension
        y_pred_raw = model.predict(X_single, verbose=0)[0][:seq_len].argmax(axis=-1)
        
        # Apply sliding window temporal smoothing (7-frame majority voting)
        print(f"  Applying 7-frame sliding window smoothing...")
        y_pred = apply_temporal_smoothing(y_pred_raw, window_size=7, min_segment_length=3)
        
        # Analyze smoothing improvement
        smoothing_stats = analyze_smoothing_improvement(y_pred_raw, y_pred, LABEL_CLASSES)
        print(f"  Smoothing: {smoothing_stats['frames_changed']} frames changed ({smoothing_stats['change_percentage']:.1f}%)")
        print(f"  Segments: {smoothing_stats['original_segments']} → {smoothing_stats['smoothed_segments']} (reduced by {smoothing_stats['segment_reduction']})")
        
        # Convert to segments
        pred_segments = collapse_predictions(y_pred)
        pred_json = segments_to_timestamps(pred_segments, hop_length, sample_rate)
        
        # Format as combined_data.json structure
        predicted_data[track_name] = {
            "salami_id": int(track_id),
            "annotations": []
        }
        
        for seg in pred_json:
            annotation = {
                "start_time": f"{seg['start']:.9f}",
                "end_time": f"{seg['end']:.9f}",
                "section": seg['label']
            }
            predicted_data[track_name]["annotations"].append(annotation)
        
        # Compare with ground truth
        gt_segments = []
        for start, end, label in segments_dict[track_id]:
            if label in CANONICAL_LABELS and (end - start) >= 2.0:
                gt_segments.append({"start": start, "end": end, "label": label})
        
        if len(gt_segments) > 0:
            # Calculate F1 score with tolerance
            f1_results = calculate_relaxed_f1(pred_json, gt_segments)
            f1_score = f1_results['f1']
            
            print(f"  F1 Score: {f1_score:.3f} (P: {f1_results['precision']:.3f}, R: {f1_results['recall']:.3f})")
            
            total_f1 += f1_score
            valid_tracks += 1
            print(f"    Predicted segments: {len(pred_json)}")
            print(f"    Ground truth segments: {len(gt_segments)}")
            if len(pred_json) > 0:
                print(f"    Sample prediction: {pred_json[0]}")
            if len(gt_segments) > 0:
                print(f"    Sample ground truth: {gt_segments[0]}")
        else:
            print(f"  Track {track_id}: No valid ground truth segments")
    
    # Save predicted output in combined_data.json format
    with open('predicted_combined_data.json', 'w') as f:
        json.dump(predicted_data, f, indent=2)
    
    mean_f1 = total_f1 / valid_tracks if valid_tracks > 0 else 0
    print(f"Mean F1 across all tracks: {mean_f1:.3f}")
    
    return predicted_data, mean_f1


def train_until_convergence(max_iterations=2, target_f1=0.3, test_mode=True):
    """
    Train the model iteratively until it matches the ground truth.
    
    WARNING: This function currently does not converge due to fundamental issues
    in the training setup (class imbalance, timing misalignment, etc.)
    
    Args:
        max_iterations: Maximum number of training iterations
        target_f1: Target F1 score for convergence
        test_mode: If True, use reduced parameters
        
    Returns:
        Tuple of (best_model, best_f1_score)
    """
    best_f1 = 0
    best_model = None
    
    if test_mode:
        print("🧪 TEST MODE: Using reduced parameters for quick testing")
        print(f"   - Max iterations: {max_iterations}")
        print(f"   - Target F1: {target_f1}")
        print(f"   - Epochs per iteration: 2 (instead of 50)")
        print(f"   - Batch size: 2 (instead of 4)")
    
    for iteration in range(max_iterations):
        print(f"\n=== Training Iteration {iteration + 1} ===")
        
        # Try to load existing model first (for any iteration)
        try:
            import tensorflow as tf
            import os
            
            # Check if model file exists
            if not os.path.exists(MODEL_PATH):
                print(f"Model file '{MODEL_PATH}' not found, training from scratch...")
                raise FileNotFoundError("Model file not found")
            
            print(f"Loading existing model from '{MODEL_PATH}'...")
            # Import the custom layer to register it
            from transformer import PositionalEncoding
            # Load model with custom objects
            model = tf.keras.models.load_model(MODEL_PATH, 
                                             custom_objects={'PositionalEncoding': PositionalEncoding})
            print("Successfully loaded existing model!")
            
            # Load training data for prediction/comparison
            X_list, y_list, track_ids, track_names, segments_dict, sample_rate, hop_length = load_training_data()
            
        except Exception as e:
            print(f"Could not load existing model: {str(e)}")
            print("Training from scratch...")
            model, X_list, y_list, track_ids, track_names, segments_dict, sample_rate, hop_length = train_transformer_model(test_mode=test_mode)
        
        # Predict and compare
        predicted_data, mean_f1 = predict_and_compare(model, X_list, track_ids, track_names, segments_dict, sample_rate, hop_length)
        
        print(f"Iteration {iteration + 1} - Mean F1: {mean_f1:.3f}")
        
        # Save best model
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_model = model
            model.save(MODEL_PATH)
            print(f"New best F1: {best_f1:.3f}")
        
        # Check convergence
        if mean_f1 >= target_f1:
            print(f"Target F1 of {target_f1} reached!")
            break
            
        # Continue training if not converged and not using pre-existing model
        if iteration < max_iterations - 1:
            print("Continuing training...")
            # Additional training epochs
            X_batch, y_batch = prepare_batch_data(X_list, y_list)
            epochs = 2 if test_mode else 10
            # Recompile model for continued training
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                metrics=['accuracy']
            )
            model.fit(X_batch, y_batch, batch_size=2 if test_mode else 4, epochs=epochs, verbose=1)
            model.save(MODEL_PATH)
            print("Model saved for next iteration...")
    
    return best_model, best_f1


def prepare_batch_data(X_list, y_list):
    """
    Prepare batched data for training.
    
    Args:
        X_list: List of audio features
        y_list: List of frame labels
        
    Returns:
        Tuple of (X_batch, y_categorical)
    """
    import tensorflow as tf
    
    max_len = max(x.shape[0] for x in X_list)
    n_features = X_list[0].shape[1]
    
    X_batch = np.zeros((len(X_list), max_len, n_features), dtype=np.float32)
    y_batch = np.full((len(y_list), max_len), fill_value=LABEL_CLASSES.index('unknown'), dtype=np.int32)
    
    for i, (xf, yl) in enumerate(zip(X_list, y_list)):
        seq_len = xf.shape[0]
        X_batch[i, :seq_len, :] = xf
        y_batch[i, :seq_len] = yl
    
    # Convert labels to categorical format for training
    n_classes = len(LABEL_CLASSES)
    y_categorical = tf.keras.utils.to_categorical(y_batch, num_classes=n_classes)
    
    return X_batch, y_categorical


def predict_single_song(audio_path, model_path='best_transformer_model.h5'):
    """
    Predict segments for a single song (for product use).
    
    Args:
        audio_path: Path to audio file
        model_path: Path to trained model
        
    Returns:
        Dictionary in combined_data.json format
    """
    import tensorflow as tf
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load and process audio
    sample_rate = 8000
    n_mels = 128
    hop_length = 512
    
    y_audio = load_audio_mono(audio_path, sr=sample_rate)
    X_feat = compute_logmel(y_audio, sr=sample_rate, n_fft=2048, hop_length=hop_length, n_mels=n_mels)
    X_feat = normalize_features(X_feat)
    
    # Predict
    X_input = X_feat[None, ...]  # Add batch dimension
    y_pred_raw = model.predict(X_input)[0].argmax(axis=-1)
    
    # Apply sliding window temporal smoothing
    y_pred = apply_temporal_smoothing(y_pred_raw, window_size=7, min_segment_length=3)
    
    # Convert to segments
    pred_segments = collapse_predictions(y_pred)
    pred_json = segments_to_timestamps(pred_segments, hop_length, sample_rate)

    # Format as product output
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    result = {
        song_name: {
            "annotations": []
        }
    }
    
    for seg in pred_json:
        annotation = {
            "start_time": f"{seg['start']:.9f}",
            "end_time": f"{seg['end']:.9f}",
            "section": seg['label']
        }
        result[song_name]["annotations"].append(annotation)
    
    return result


def main():
    """
    Main training and evaluation pipeline.
    
    CURRENT STATUS: This pipeline does not work due to fundamental issues:
    1. Severe class imbalance (70-80% Empty labels)
    2. Timing misalignment between predictions and ground truth
    3. Different segment granularity
    4. Strict F1 evaluation requiring exact time overlap
    
    The model learns to predict "Empty" for most frames to minimize loss,
    resulting in 0.000 F1 scores across all tracks.
    """
    print("Starting iterative training until convergence...")
    print("WARNING: Current implementation has fundamental issues preventing convergence.")
    
    # Test mode: quick validation
    print("🧪 Running in TEST MODE for quick validation...")
    best_model, best_f1 = train_until_convergence(max_iterations=2, target_f1=0.3, test_mode=True)
    
    print(f"\n=== Test Results ===")
    print(f"Best F1 achieved: {best_f1:.3f}")
    print("Best model saved as 'best_transformer_model.h5'")
    print("Predicted data saved as 'predicted_combined_data.json'")
    
    print("\n=== MODEL FEATURES ===")
    print("✅ Sliding window temporal smoothing (7-frame majority voting)")
    print("✅ Frame-level correction before segment collapse")
    print("✅ Short segment removal and edge case smoothing")
    print("✅ Relaxed F1 evaluation with ±5 second tolerance")
    
    print("\n=== Example Inference ===")
    print("To predict segments for a new song, use:")
    print("result = predict_single_song('path/to/your/song.mp3')")
    print("This will return the same format as combined_data.json")


if __name__ == '__main__':
    main()