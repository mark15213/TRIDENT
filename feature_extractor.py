# feature_extractor.py
import os
import torch
import h5py
import timm
from trident.patch_encoder_models import encoder_factory
from trident import OpenSlideWSI
from trident.segmentation_models import segmentation_model_factory
import pandas as pd
import config # Import your config
import re # For sanitizing filenames
import traceback

def sanitize_path_for_filename(path):
    """Creates a relatively safe directory/filename component from a path."""
    # Remove protocol if present (like file://)
    path = re.sub(r'^file://', '', path)
    # Remove leading slash to prevent absolute paths interpretation
    if path.startswith('/'):
        path = path[1:]
    # Replace directory separators and other problematic characters with underscores
    sanitized = re.sub(r'[\\/:\*\?"<>\| ]', '_', path)
    # Replace multiple consecutive underscores with a single one
    sanitized = re.sub(r'_+', '_', sanitized)
    # Optional: truncate if too long (adjust max_len as needed)
    max_len = 200
    if len(sanitized) > max_len:
        # Simple truncation, consider hashing for very long/colliding paths if needed
        sanitized = sanitized[-(max_len):]
    return sanitized

# --- get_wsi_path_from_slide_id function is REMOVED ---

# --- extract_features_for_wsi function ---
# (Logic inside remains similar, focus on input args and output naming)
def extract_features_for_wsi(wsi_full_path, unique_output_identifier, output_base_dir):
    """
    Extracts features for a single WSI.
    Args:
        wsi_full_path (str): Full path to the WSI file.
        unique_output_identifier (str): A unique ID derived from the sanitized path for naming outputs.
        output_base_dir (str): Base directory where feature subdirs are stored (e.g., config.EXTRACTED_FEATURES_DIR).
    Returns:
        str: Path to the HDF5 file containing extracted features, or None if failed.
    """
    print(f"Processing WSI: {wsi_full_path}")
    # Output directory for this specific slide's intermediate files (seg, coords) and final features
    # Place this directory *inside* the main output_base_dir
    slide_output_dir = os.path.join(output_base_dir, unique_output_identifier)
    os.makedirs(slide_output_dir, exist_ok=True)

    # Define the final desired path for the feature file *before* extraction
    final_features_path = os.path.join(slide_output_dir, f"{unique_output_identifier}_features.h5")

    # --- Weight check ---
    if not os.path.exists(config.CONVNEXT_WEIGHTS_PATH):
        print(f"ERROR: ConvNeXt V2 weights not found at '{config.CONVNEXT_WEIGHTS_PATH}'.")
        return None

    try:
        # 1. Create OpenSlideWSI object
        print("  Creating OpenSlideWSI object...")
        # Set lazy_init to True if memory is an issue during batch processing,
        # but False is okay if opening one slide at a time is fine.
        slide = OpenSlideWSI(slide_path=wsi_full_path, lazy_init=False)

        # 2. Run tissue segmentation
        print("  Running tissue segmentation...")
        segmentation_model_instance = segmentation_model_factory("hest")
        _ = slide.segment_tissue( # We don't necessarily need the geojson path later
            segmentation_model=segmentation_model_instance,
            target_mag=config.TARGET_MAG_SEGMENTATION,
            job_dir=slide_output_dir, # Save segmentation outputs per slide
            device=config.DEVICE_STR
        )

        # 3. Extract patch coordinates
        print("  Extracting patch coordinates...")
        coords_h5_path = slide.extract_tissue_coords(
            target_mag=config.TARGET_MAG_PATCHING,
            patch_size=config.PATCH_SIZE,
            save_coords=slide_output_dir # Save coords per slide
        )
        # Check if coordinates were actually extracted
        with h5py.File(coords_h5_path, 'r') as hf_coords:
             if 'coords' not in hf_coords or hf_coords['coords'].shape[0] == 0:
                 print(f"  Warning: No coordinates extracted for {wsi_full_path}. Skipping feature extraction.")
                 # Optionally save an empty feature file or handle later
                 # Saving empty HDF5 to mark as processed but with no features:
                 with h5py.File(final_features_path, 'w') as hf_feat:
                    hf_feat.create_dataset('features', shape=(0,0), dtype=np.float32) # Use known dtype if possible
                    hf_feat.create_dataset('coords', shape=(0,2), dtype=np.int64) # Match coord dtype
                 print(f"  Saved empty feature file: {final_features_path}")
                 return final_features_path # Return path even if empty

        # 4. Instantiate Custom ConvNeXt V2 Encoder
        print(f"  Instantiating patch encoder: {config.CONVNEXT_MODEL_NAME_IN_FACTORY}...")
        # Ensure the encoder registration happens if needed (see previous comments)
        patch_encoder = encoder_factory(
            model_name=config.CONVNEXT_MODEL_NAME_IN_FACTORY,
            weights_path=config.CONVNEXT_WEIGHTS_PATH,
            input_size=config.CONVNEXT_INPUT_SIZE,
            precision=config.CONVNEXT_PRECISION,
            device_str=config.DEVICE_STR # Pass device_str
        )
        if not patch_encoder.model:
            print("  ERROR: Patch encoder model not built. Aborting for this slide.")
            # Clean up coordinates file? Optional.
            # os.remove(coords_h5_path)
            return None

        # 5. Run Patch Feature Extraction
        print("  Running patch feature extraction...")
        # save_features expects a directory. Trident will create a subdir based on WSI name within it.
        _ = slide.extract_patch_features(
            patch_encoder=patch_encoder,
            coords_path=coords_h5_path,
            save_features=slide_output_dir, # Save into the slide-specific directory
            device=config.DEVICE_STR
        )

        # --- Find and Rename the HDF5 file ---
        wsi_basename = os.path.splitext(os.path.basename(wsi_full_path))[0]
        potential_trident_h5_path = os.path.join(slide_output_dir, wsi_basename, "features.h5")

        if os.path.exists(potential_trident_h5_path):
            os.rename(potential_trident_h5_path, final_features_path)
            try:
                os.rmdir(os.path.join(slide_output_dir, wsi_basename))
            except OSError: pass # Ignore error if dir not empty or other issue
        elif os.path.exists(os.path.join(slide_output_dir, "features.h5")):
            os.rename(os.path.join(slide_output_dir, "features.h5"), final_features_path)
        elif os.path.exists(final_features_path):
            print(f"  Feature file already exists at target path: {final_features_path}")
        else:
            print(f"  ERROR: Could not locate the features HDF5 file after extraction. Looked for patterns like {potential_trident_h5_path} and features.h5 in {slide_output_dir}")
            return None

        print(f"  Patch features saved to: {final_features_path}")
        return final_features_path

    except Exception as e:
        print(f"  ERROR processing WSI {wsi_full_path}: {e}")
        traceback.print_exc()
        # Clean up potentially partially created files? Optional.
        return None
    finally:
        # Explicitly close the slide object if lazy_init=False was used
        if 'slide' in locals() and hasattr(slide, 'close'):
            slide.close()
        # Release GPU memory if patch_encoder was loaded here
        if 'patch_encoder' in locals():
            del patch_encoder
        if config.DEVICE_STR.startswith('cuda'):
            torch.cuda.empty_cache()


def run_batch_feature_extraction():
    """
    Reads the NEW CSV format (path, label), extracts features for each WSI path.
    """
    os.makedirs(config.EXTRACTED_FEATURES_DIR, exist_ok=True)
    try:
        df = pd.read_csv(config.CSV_PATH)
        # Basic validation of CSV format
        if 'path' not in df.columns or 'label' not in df.columns:
             raise ValueError("CSV file must contain 'path' and 'label' columns.")
        print(f"Found {len(df)} entries in {config.CSV_PATH}")
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {config.CSV_PATH}")
        return
    except Exception as e:
        print(f"ERROR reading CSV file {config.CSV_PATH}: {e}")
        return

    processed_list_path = os.path.join(config.EXTRACTED_FEATURES_DIR, "processed_slides.txt")
    processed_paths = set()
    if os.path.exists(processed_list_path):
        try:
            with open(processed_list_path, 'r') as f:
                # Store the original WSI path as the identifier
                processed_paths = set(line.strip() for line in f)
            print(f"Loaded {len(processed_paths)} already processed slide paths.")
        except Exception as e:
            print(f"Warning: Could not read processed slides list {processed_list_path}: {e}")


    feature_file_map = {} # Maps original WSI path -> path_to_features.h5

    # Load existing map if it exists to potentially resume/update
    map_file_path = os.path.join(config.EXTRACTED_FEATURES_DIR, "feature_file_map.csv")
    if os.path.exists(map_file_path):
        try:
            df_map_existing = pd.read_csv(map_file_path)
            if 'wsi_path' in df_map_existing.columns and 'feature_path' in df_map_existing.columns:
                feature_file_map = pd.Series(df_map_existing.feature_path.values, index=df_map_existing.wsi_path).to_dict()
                print(f"Loaded existing feature map with {len(feature_file_map)} entries.")
            else:
                print(f"Warning: Existing map file {map_file_path} has incorrect columns. Will create a new one.")
        except Exception as e:
             print(f"Warning: Could not load existing feature map {map_file_path}: {e}. Will create a new one.")


    for index, row in df.iterrows():
        wsi_full_path = row['path']

        # Use the original path as the primary identifier for skipping checks
        if wsi_full_path in processed_paths:
            print(f"Skipping already processed: {wsi_full_path}")
            # Ensure it's in the map if we loaded an existing one
            if wsi_full_path not in feature_file_map:
                 # Try to reconstruct the expected path if skipping
                 sanitized_identifier = sanitize_path_for_filename(wsi_full_path)
                 expected_h5_path = os.path.join(config.EXTRACTED_FEATURES_DIR, sanitized_identifier, f"{sanitized_identifier}_features.h5")
                 if os.path.exists(expected_h5_path):
                      feature_file_map[wsi_full_path] = expected_h5_path
                 else:
                     print(f"Warning: Processed flag found for {wsi_full_path} but feature file {expected_h5_path} missing. Will attempt re-processing.")
                     processed_paths.remove(wsi_full_path) # Remove from processed set to force re-run
                     # Continue to processing block
            else:
                continue # Already processed and in map, skip to next row

        # --- Proceed if not skipped ---
        print(f"\nProcessing {index+1}/{len(df)}: {wsi_full_path}")

        # Check if WSI file exists before proceeding
        if not os.path.exists(wsi_full_path):
            print(f"  ERROR: WSI file not found at '{wsi_full_path}'. Skipping.")
            continue

        # Create the unique identifier for output files/dirs using sanitized path
        sanitized_identifier = sanitize_path_for_filename(wsi_full_path)

        # Call the extraction function
        features_h5_path = extract_features_for_wsi(
            wsi_full_path=wsi_full_path,
            unique_output_identifier=sanitized_identifier,
            output_base_dir=config.EXTRACTED_FEATURES_DIR # Pass the main features directory
        )

        if features_h5_path and os.path.exists(features_h5_path):
            print(f"  Successfully extracted features for {wsi_full_path} to {features_h5_path}")
            feature_file_map[wsi_full_path] = features_h5_path # Map original path to H5 path
            # Add original path to processed list
            if wsi_full_path not in processed_paths:
                 try:
                     with open(processed_list_path, 'a') as f:
                         f.write(wsi_full_path + '\n')
                     processed_paths.add(wsi_full_path) # Update in-memory set
                 except Exception as e:
                     print(f"Warning: Could not write to processed list {processed_list_path}: {e}")
        else:
            print(f"  Failed to extract features for {wsi_full_path}")
            # Optionally remove from map if it failed but was previously mapped
            if wsi_full_path in feature_file_map:
                del feature_file_map[wsi_full_path]


    # Save the final map of WSI path to feature file path
    try:
        # Check if map is empty before saving
        if feature_file_map:
             map_df_to_save = pd.DataFrame(list(feature_file_map.items()), columns=['wsi_path', 'feature_path'])
             map_df_to_save.to_csv(map_file_path, index=False)
             print(f"\nFeature extraction complete. Map saved to {map_file_path}")
        else:
             print("\nFeature extraction complete, but no features were successfully processed or mapped.")
             # Save an empty map file? Or just leave it? Let's save empty with headers.
             pd.DataFrame(columns=['wsi_path', 'feature_path']).to_csv(map_file_path, index=False)


    except Exception as e:
        print(f"ERROR saving feature map to {map_file_path}: {e}")

    print(f"All extracted features subdirectories are under: {config.EXTRACTED_FEATURES_DIR}")


if __name__ == '__main__':
    # Ensure the factory for your custom ConvNeXt is registered if needed
    # ... (registration logic if not handled elsewhere) ...

    run_batch_feature_extraction()