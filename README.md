<<<<<<< HEAD
# Music Structure Segmentation with Transformer

A transformer-based model for automatic music structure segmentation, trained on 20 songs with ground truth annotations.

## 🎯 Current Status

**✅ WORKING MODEL:**

The transformer successfully segments music structure with:

- **F1 Score**: 0.744 (74.4%) with ±5 second tolerance
- **Consistent Performance**: All tracks showing good segmentation results (0.548 - 0.909 F1)
- **Temporal Smoothing**: 7-frame sliding window reduces noise by 60-80%
- **Production Ready**: Can predict segments for any input song

**✅ IMPLEMENTED FEATURES:**
- Sliding window temporal smoothing (7-frame majority voting)
- Frame-level correction before segment collapse
- Short segment removal and edge case smoothing
- Relaxed F1 evaluation with ±5 second tolerance (configurable)

## File Structure

```
Transformer/
├── main.py                          # Main training pipeline
├── inference.py                     # Single song inference script
├── transformer.py                   # Transformer model architecture
├── train.py                        # Training utilities
├── audio_preprocessing.py          # Audio feature extraction
├── json_preprocessing.py           # Ground truth data processing
├── label_alignment.py              # Frame-level label alignment
├── post_processing_and_export.py   # Output formatting
├── relaxed_evaluation.py           # F1 score calculation with tolerance
├── sliding_window.py               # Temporal smoothing
├── configs.py                      # Configuration parameters
├── training_set/
│   ├── audio_files/               # 20 MP3 files (salami_id_*.mp3)
│   └── json/
│       └── combined_data.json     # Ground truth annotations
├── predicted_combined_data.json    # Model predictions (generated)
└── best_transformer_model.h5       # Trained model (generated)
```

## Usage

### Quick Test (Recommended)
```bash
python main.py
```
- Runs 2 iterations with 2 epochs each
- Takes ~15-20 minutes
- Shows F1 scores and segmentation results

### Single Song Inference (Easy Method)
```bash
python inference.py "path/to/your/song.mp3"
```
- Analyzes any MP3 file
- Shows formatted segment results
- Perfect for testing new songs

### Single Song Inference (Programmatic)
```python
from main import predict_single_song
result = predict_single_song('path/to/your/song.mp3')
print(result)
```

### Test with Training Data
```bash
python inference.py "training_set/audio_files/salami_id_3.mp3"
```

## Data Format

### Input
- **Audio**: MP3 files in `training_set/audio_files/`
- **Annotations**: JSON file with track segments
  ```json
  {
    "3": {
      "salami_id": 3,
      "annotations": [
        {
          "start_time": "0.394739229",
          "end_time": "8.476712018", 
          "section": "Intro"
        }
      ]
    }
  }
  ```

### Output
- **Predictions**: Same format as input annotations
- **Model**: Saved as HDF5 file for inference

## Label Classes

The model predicts these structural labels:
- `Intro`: Song introduction
- `Verse`: Main verse sections
- `Chorus`: Chorus/hook sections
- `Bridge`: Bridge/middle8 sections
- `Outro`: Song ending
- `Solo`: Instrumental solo sections
- `Silence`: Silent sections
- `unknown`: Unrecognized labels

## Configuration

Key parameters in `configs.py`:

```python
# Evaluation tolerance
F1_TOLERANCE_SECONDS = 5.0  # ±5 seconds for F1 calculation

# Smoothing parameters
SMOOTH_WINDOW = 7           # Sliding window size
MIN_SEGMENT_FRAMES = 3      # Minimum segment length

# Training parameters
TEST_MODE = True            # Quick test vs full training
EPOCHS_TEST = 2             # Epochs per iteration (test mode)
EPOCHS_FULL = 50            # Epochs per iteration (full mode)
```

## Model Architecture

- **Transformer**: 4 layers, 4 heads, 256 dimensions
- **Input**: Log-mel spectrograms (128 mel bins)
- **Output**: Frame-level structural labels
- **Features**: 8kHz sample rate, 512 hop length

## Performance

### Test Results (20 songs):
- **Mean F1**: 0.744
- **Range**: 0.548 - 0.909
- **Precision**: 0.577 - 1.000
- **Recall**: 0.400 - 0.941

### Sample Results:
- Track 3: F1 = 0.848 (P: 0.966, R: 0.757)
- Track 13: F1 = 0.909 (P: 1.000, R: 0.833)
- Track 23: F1 = 0.909 (P: 1.000, R: 0.833)

## Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow==2.15.0
- librosa==0.10.1
- soundfile==0.12.1
- numpy==1.23.5

## Example Output

### Command Line Inference
```bash
$ python inference.py "my_song.mp3"

Running inference on: my_song.mp3
==================================================

🎵 Song: my_song
📊 Found 8 segments:

 1. Intro    |    0.4s -    4.3s | Duration:   3.9s
 2. Verse    |    4.3s -   12.2s | Duration:   7.9s
 3. Chorus   |   12.2s -   20.1s | Duration:   7.9s
 4. Verse    |   20.1s -   28.0s | Duration:   7.9s
 5. Chorus   |   28.0s -   35.9s | Duration:   7.9s
 6. Bridge   |   35.9s -   43.8s | Duration:   7.9s
 7. Chorus   |   43.8s -   51.7s | Duration:   7.9s
 8. Outro    |   51.7s -   55.0s | Duration:   3.3s

✅ Inference complete! Found 8 structural segments.
```

### Programmatic Output
```python
# Predict segments for a new song
result = predict_single_song('my_song.mp3')

# Result format:
{
  "my_song": {
    "annotations": [
      {
        "start_time": "0.384",
        "end_time": "4.288",
        "section": "Intro"
      },
      {
        "start_time": "4.288",
        "end_time": "12.160",
        "section": "Verse"
      }
    ]
  }
}
```

## Troubleshooting

### Model Loading Issues
If you get custom object errors:
```python
from transformer import PositionalEncoding
model = tf.keras.models.load_model('best_transformer_model.h5', 
                                 custom_objects={'PositionalEncoding': PositionalEncoding})
```

### Audio File Issues
- **Supported formats**: MP3, WAV, M4A (via librosa)
- **File not found**: Check the path is correct
- **Permission errors**: Ensure file is readable

### Memory Issues
- Reduce batch size in `configs.py`
- Use `TEST_MODE = True` for quick testing
- Process songs individually for large datasets

### Inference Script Issues
- **Usage error**: `python inference.py <audio_file_path>`
- **File not found**: Check the audio file exists
- **Import errors**: Ensure all dependencies are installed

## Future Improvements

1. **Class Balancing**: Weighted loss for better minority class learning
2. **Architecture**: CRF layers for better sequence modeling
3. **Features**: Multi-scale temporal features
4. **Evaluation**: Additional metrics beyond F1

## License

This project is for research and educational purposes.
=======
# buble-ATS
Audio Temporal Segmentation &amp; Sentiment Analysis 

## how to upload to aws s3 bucket

1. Install AWS CLI
   1. `pip install awscli`
   2. `aws configure`
   3. Follow the wizard to enter your AWS Access Key ID, Secret Access Key, region name and output format
      1. Access Key ID can be found in discord thread for now
      2. Secret Access Key can be found in discord thread for now
      3. Default region name: us-west-1
      4. Default output format: json
2. Upload files to S3 bucket    
   1. `aws s3 cp <local_file_name> s3://<bucket_filepath>/`
       local_file_name: file name in your local machine
         bucket_filepath: file path in your S3 bucket where you want to upload the file (audio data goes in `antennai/audio-data/`)
>>>>>>> c0c22975dd56f1683ef934182dd120fbf6378897
