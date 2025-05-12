import os
import torch
from PIL import Image
from huggingface_hub import snapshot_download
import h5py
import timm

from trident.patch_encoder_models import encoder_factory

# Assuming trident imports are standard
from trident import OpenSlideWSI
from trident.segmentation_models import segmentation_model_factory

# --- Global Configuration ---
OUTPUT_DIR = "tutorial_convnext_extraction_output/"
DEVICE_STR = f"cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR) # torch.device object for operations outside trident if any
WSI_FNAME = '394140.svs' # Example WSI filename
WSI_FULL_PATH = "/mnt/warm/SenseCare-PathCloud/single/storage/rj/section_files/20240419/0880014d8f3afd247f3af16fea1c11d1/2024-014590#2#1.sdpc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Custom ConvNeXt V2 Configuration ---
CONVNEXT_MODEL_NAME_IN_FACTORY = 'convnextv2l'
# !!! IMPORTANT: SET THE PATH TO YOUR PRETRAINED WEIGHTS !!!
CONVNEXT_WEIGHTS_PATH = '/mnt/rj200t/ckpt/checkpoint-2.pth' # <--- MODIFY THIS
CONVNEXT_INPUT_SIZE = 224
CONVNEXT_PRECISION = torch.float16 if DEVICE_STR.startswith("cuda") else torch.float32

TARGET_MAG_SEGMENTATION = 10
TARGET_MAG_PATCHING = 20
PATCH_SIZE = 224

# --- Helper function for HDF5 inspection ---
def print_attrs(name, obj):
    print(f"Object: {name}")
    if hasattr(obj, 'attrs'):
        for key, value in obj.attrs.items():
            print(f"  Attribute - {key}: {value}")

# --- Main Feature Extraction Pipeline ---
def run_feature_extraction():
    print(f"Using device: {DEVICE_STR}")

    # Pre-flight check for weights
    if not os.path.exists(CONVNEXT_WEIGHTS_PATH):
        print(f"ERROR: ConvNeXt V2 weights not found at '{CONVNEXT_WEIGHTS_PATH}'.")
        if CONVNEXT_WEIGHTS_PATH == '/mnt/rj200t/ckpt/checkpoint-2.pth':
            print("Attempting to create dummy weights for demonstration as placeholder path is used...")
            try:
                dummy_model_temp = timm.create_model('convnextv2_large', pretrained=False, num_classes=0)
                os.makedirs(os.path.dirname(CONVNEXT_WEIGHTS_PATH), exist_ok=True)
                torch.save(dummy_model_temp.state_dict(), CONVNEXT_WEIGHTS_PATH)
                print(f"Dummy weights created at {CONVNEXT_WEIGHTS_PATH}")
            except Exception as e_dummy:
                print(f"Failed to create dummy weights: {e_dummy}. Please set a valid weights path.")
                exit()
        else:
            exit() # Exit if specific path given but not found

    # 1. Download WSI
    print(f"Downloading WSI: {WSI_FNAME}...")
    local_wsi_dir = snapshot_download(
        repo_id="MahmoodLab/unit-testing",
        repo_type='dataset',
        local_dir=os.path.join(OUTPUT_DIR, 'wsis'),
        allow_patterns=[WSI_FNAME]
    )
    wsi_path = os.path.join(local_dir, WSI_FNAME)

    # 2. Create OpenSlideWSI object
    print("Creating OpenSlideWSI object...")
    # print(wsi_path)
    slide = OpenSlideWSI(slide_path=wsi_path, lazy_init=False)

    # 3. Run tissue segmentation
    print("Running tissue segmentation...")
    # Assuming segmentation_model_factory correctly handles device or the model does.
    segmentation_model_instance = segmentation_model_factory("hest", device=DEVICE_STR)
    # segmentation_model_instance.model.to(DEVICE) # If factory doesn't move it

    geojson_contours_path = slide.segment_tissue(
        segmentation_model=segmentation_model_instance,
        target_mag=TARGET_MAG_SEGMENTATION,
        job_dir=OUTPUT_DIR,
        device=DEVICE_STR
    )
    print(f"Tissue contours saved to: {geojson_contours_path}")

    # 4. Extract patch coordinates
    print("Extracting patch coordinates...")
    coords_h5_path = slide.extract_tissue_coords(
        target_mag=TARGET_MAG_PATCHING,
        patch_size=PATCH_SIZE,
        save_coords=OUTPUT_DIR
    )
    print(f"Patch coordinates saved to: {coords_h5_path}")

    viz_coords_dir = os.path.join(OUTPUT_DIR, "visualization_coords")
    os.makedirs(viz_coords_dir, exist_ok=True)
    viz_coords_image_path = slide.visualize_coords(
        coords_path=coords_h5_path,
        save_patch_viz=viz_coords_dir
    )

    print("\nInspecting patch coordinates HDF5 file:")
    with h5py.File(coords_h5_path, 'r') as h5_file:
        h5_file.visititems(print_attrs)

    # 5. Instantiate Custom ConvNeXt V2 Encoder using the factory from load.py
    print(f"\nInstantiating patch encoder: {CONVNEXT_MODEL_NAME_IN_FACTORY}...")
    patch_encoder = encoder_factory( # This now calls the factory from load.py
        model_name=CONVNEXT_MODEL_NAME_IN_FACTORY,
        weights_path=CONVNEXT_WEIGHTS_PATH,
        input_size=CONVNEXT_INPUT_SIZE,
        precision=CONVNEXT_PRECISION,
        device_str=DEVICE_STR # Pass device_str for MyConvNeXtV2LargeSSLEncoder
    )
    print(f"Patch encoder '{patch_encoder.enc_name}' instantiated.")
    if patch_encoder.model: # Check if model was built
        print(f"  Encoder model device: {next(patch_encoder.model.parameters()).device}")
        print(f"  Encoder model dtype: {next(patch_encoder.model.parameters()).dtype}")
    else:
        print("  Encoder model not available (might be an issue with BasePatchEncoder or _build).")


    # 6. Run Patch Feature Extraction
    print("Running patch feature extraction...")
    features_dir_name = f"features_{patch_encoder.enc_name.replace('/', '_') if patch_encoder.enc_name else 'custom_encoder'}"
    features_dir = os.path.join(OUTPUT_DIR, features_dir_name)
    os.makedirs(features_dir, exist_ok=True)

    features_h5_path = slide.extract_patch_features(
        patch_encoder=patch_encoder, # The encoder with model already on device
        coords_path=coords_h5_path,
        save_features=features_dir,
        device=DEVICE_STR # trident's extract_patch_features might use this for its own Dataloader/workers
    )
    print(f"Patch features saved to: {features_h5_path}")

    # 7. Inspect features HDF5 file
    print("\nInspecting patch features HDF5 file:")
    with h5py.File(features_h5_path, 'r') as h5_file:
        h5_file.visititems(print_attrs)
        if 'features' in h5_file:
            print(f"  Shape of 'features' dataset: {h5_file['features'].shape}")
            print(f"  Dtype of 'features' dataset: {h5_file['features'].dtype}")

    print("\nFeature extraction pipeline complete.")
    print(f"All outputs saved in: {OUTPUT_DIR}")

if __name__ == '__main__':
    if CONVNEXT_WEIGHTS_PATH == 'path/to/your/custom_convnextv2_large_weights.pth' and \
       not os.path.exists(CONVNEXT_WEIGHTS_PATH):
        print("*"*80)
        print("WARNING: The script is using a placeholder path for ConvNeXt V2 weights.")
        print(f"         A DUMMY weights file will be created at '{CONVNEXT_WEIGHTS_PATH}' IF the path is the default placeholder.")
        print("         For meaningful results, please update CONVNEXT_WEIGHTS_PATH with your actual weights.")
        print("*"*80)

    run_feature_extraction()