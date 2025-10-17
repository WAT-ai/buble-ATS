# Central configuration for the project

# Paths
COMBINED_JSON_PATH = 'training_set/json/combined_data.json'
AUDIO_GLOB = 'training_set/audio_files/salami_id_*.mp3'
MODEL_PATH = 'best_transformer_model.h5'
PREDICTED_OUTPUT_PATH = 'predicted_combined_data.json'

# Audio/feature params
SAMPLE_RATE = 8000
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# Training params
TEST_MODE = True
EPOCHS_TEST = 2
EPOCHS_FULL = 50
BATCH_TEST = 2
BATCH_FULL = 4
LEARNING_RATE = 1e-4
LABEL_SMOOTHING = 0.1
VAL_SPLIT = 0.2

# Smoothing params
SMOOTH_WINDOW = 7
MIN_SEGMENT_FRAMES = 3

# Iterative training
MAX_ITERS_TEST = 2
MAX_ITERS_FULL = 10
TARGET_F1_TEST = 0.3
TARGET_F1_FULL = 0.8

# Evaluation params
F1_TOLERANCE_SECONDS = 5.0  # Tolerance for relaxed F1 evaluation
