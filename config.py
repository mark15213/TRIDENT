# config.py
import torch
import os # Make sure os is imported

# --- Paths ---
BASE_OUTPUT_DIR = "wsi_classification_output_breast/" # Optional: Changed output dir name
CSV_PATH = "/mnt/rj200t/downstream/breast-cancer_vs_healthy.csv"  # <--- UPDATE THIS
# WSI_BASE_DIR = "..." # No longer strictly needed for path resolution

EXTRACTED_FEATURES_DIR = os.path.join(BASE_OUTPUT_DIR, "extracted_features")
TRAINED_MODELS_DIR = os.path.join(BASE_OUTPUT_DIR, "trained_models")
RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, "results")

# --- Device ---
DEVICE_STR = f"cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

# --- Feature Extraction (ConvNeXt V2 Specific) ---
CONVNEXT_MODEL_NAME_IN_FACTORY = 'convnextv2l' # As defined in your trident factory
CONVNEXT_WEIGHTS_PATH = '/mnt/rj200t/ckpt/checkpoint-2.pth' # Your SSL pretrained weights
CONVNEXT_INPUT_SIZE = 224
CONVNEXT_PRECISION = torch.float16 if DEVICE_STR.startswith("cuda") else torch.float32
TARGET_MAG_SEGMENTATION = 10
TARGET_MAG_PATCHING = 20
PATCH_SIZE = 224

# --- Classifier Training ---
NUM_CLASSES = 2 # Binary classification: cancer vs healthy <--- ENSURE THIS IS 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 4 # Adjust based on GPU memory for classifier training
AGGREGATION_METHOD = "mean" # "mean", "max", "attention"
SPLIT_RATIO = [0.7, 0.15, 0.15] # Train, Val, Test split

# --- Other ---
RANDOM_SEED = 42