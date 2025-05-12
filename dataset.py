# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import h5py
import numpy as np
import os
import config
from sklearn.model_selection import train_test_split

class WSIFeatureDataset(Dataset):
    def __init__(self, df_subset, feature_file_map, label_map, aggregation_fn=None, transform=None):
        """
        Args:
            df_subset (pd.DataFrame): DataFrame subset (train, val, or test) with 'path' and 'label'.
            feature_file_map (dict): Dictionary mapping WSI path (from CSV 'path' column) to HDF5 feature file path.
            label_map (dict): Dictionary mapping string labels ('cancer', 'healthy') to integer indices.
            aggregation_fn (callable, optional): Function/Module to aggregate patch features.
            transform (callable, optional): Optional transform.
        """
        self.df_subset = df_subset
        self.feature_file_map = feature_file_map
        self.label_map = label_map
        self.aggregation_fn = aggregation_fn
        self.transform = transform

        # Filter out entries for which feature files are missing or invalid in the map
        self.valid_indices = []
        original_count = len(self.df_subset)
        for idx in self.df_subset.index: # Iterate using original index
            row = self.df_subset.loc[idx]
            wsi_path = row['path'] # Use the 'path' column as the key
            if wsi_path in self.feature_file_map and os.path.exists(self.feature_file_map[wsi_path]):
                self.valid_indices.append(idx)
            else:
                 # Provide more specific warning
                 if wsi_path not in self.feature_file_map:
                     print(f"Warning: WSI path '{wsi_path}' not found in feature map. Skipping.")
                 else: # Path is in map, but file doesn't exist
                     print(f"Warning: Feature file '{self.feature_file_map[wsi_path]}' for WSI '{wsi_path}' not found on disk. Skipping.")

        if not self.valid_indices:
            raise RuntimeError("No valid feature files found for the provided DataFrame subset and map.")

        self.df_subset = self.df_subset.loc[self.valid_indices].reset_index(drop=True)
        print(f"WSIFeatureDataset: Initialized with {len(self.df_subset)} valid samples out of {original_count} provided.")


    def __len__(self):
        return len(self.df_subset)

    def __getitem__(self, idx):
        row = self.df_subset.iloc[idx]
        wsi_path = row['path'] # Get the original WSI path
        label_str = row['label']
        label_int = self.label_map[label_str]

        feature_h5_path = self.feature_file_map[wsi_path] # Look up HDF5 path using WSI path

        try:
            with h5py.File(feature_h5_path, 'r') as hf:
                if 'features' not in hf:
                    print(f"Warning: 'features' dataset not found in {feature_h5_path} for {wsi_path}. Returning zero tensor.")
                    # Need to know feature_dim for zero tensor
                    # This requires feature_dim to be passed or inferred robustly
                    feature_dim = 1024 # Replace with actual or inferred dim
                    patch_features = np.zeros((0, feature_dim), dtype=np.float32) # Empty array with correct second dim
                else:
                    patch_features = hf['features'][:] # Load features
                # coords = hf['coords'][:] # Load coords if needed

                # Ensure features are at least 2D (N_patches, Feature_dim), even if N_patches is 0
                if patch_features.ndim == 1:
                    # This case indicates an issue with saving or an unexpected format.
                    # Treat as no features found.
                    print(f"Warning: Features in {feature_h5_path} are 1D. Treating as empty.")
                    feature_dim = patch_features.shape[0] if patch_features.shape[0] > 0 else 1024 # Get dim or fallback
                    patch_features = np.zeros((0, feature_dim), dtype=np.float32)
                elif patch_features.ndim == 0: # Scalar? Also an error.
                    print(f"Warning: Features in {feature_h5_path} are scalar. Treating as empty.")
                    feature_dim = 1024 # Fallback
                    patch_features = np.zeros((0, feature_dim), dtype=np.float32)
                elif patch_features.shape[0] > 0 and patch_features.shape[1] == 0:
                    # N_patches > 0 but Feature_dim is 0? Error.
                    print(f"Warning: Features in {feature_h5_path} have 0 dimension. Treating as empty.")
                    feature_dim = 1024 # Fallback
                    patch_features = np.zeros((0, feature_dim), dtype=np.float32)


        except Exception as e:
            print(f"Error loading features for {wsi_path} from {feature_h5_path}: {e}. Returning zero tensor.")
            # Create a dummy empty array if loading fails
            feature_dim = 1024 # Replace with actual or inferred dim
            patch_features = np.zeros((0, feature_dim), dtype=np.float32)


        # Convert to tensor before aggregation
        patch_features_tensor = torch.from_numpy(patch_features).float()

        # Aggregate features (handles empty patch_features_tensor correctly if aggregation_fn does)
        if self.aggregation_fn:
            # If aggregation_fn is an nn.Module, call its forward method
            if isinstance(self.aggregation_fn, torch.nn.Module):
                 with torch.no_grad(): # Ensure no gradients calculated here if it's a module like attention
                      wsi_feature = self.aggregation_fn(patch_features_tensor)
            else: # If it's a simple function (mean, max)
                 wsi_feature = self.aggregation_fn(patch_features_tensor)

        else:
            # If no aggregation, return all patch features (requires custom collate_fn)
            wsi_feature = patch_features_tensor


        sample = {'wsi_feature': wsi_feature, 'label': torch.tensor(label_int, dtype=torch.long), 'wsi_path': wsi_path}

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_data_loaders(csv_path, feature_file_map_path, label_map, aggregation_fn, batch_size, split_ratio, random_seed):
    df_all = pd.read_csv(csv_path)

    # Create feature_file_map from the saved CSV
    if not os.path.exists(feature_file_map_path):
        raise FileNotFoundError(f"Feature file map not found at {feature_file_map_path}. Run feature extraction first.")
    try:
        df_map = pd.read_csv(feature_file_map_path)
        if 'wsi_path' not in df_map.columns or 'feature_path' not in df_map.columns:
             raise ValueError("Feature map CSV must contain 'wsi_path' and 'feature_path' columns.")
        # Use wsi_path (original path from input CSV) as the key
        feature_file_map = pd.Series(df_map.feature_path.values, index=df_map.wsi_path).to_dict()
    except Exception as e:
        print(f"Error loading feature map file {feature_file_map_path}: {e}")
        raise

    # Split data using the main DataFrame
    # Ensure stratification works correctly with string labels
    if df_all['label'].nunique() < 2:
        print("Warning: Only one class present in the dataset. Stratification may not work as expected.")
        # Simple split without stratification if only one class
        train_df, temp_df = train_test_split(df_all, test_size=(1-split_ratio[0]), random_state=random_seed)
    else:
        train_df, temp_df = train_test_split(df_all, test_size=(1-split_ratio[0]), random_state=random_seed, stratify=df_all['label'])

    # Adjust split ratio for val/test from the remainder
    if split_ratio[1] + split_ratio[2] > 1e-6: # Avoid division by zero if only train split needed
        val_test_ratio = split_ratio[2] / (split_ratio[1] + split_ratio[2])
        if temp_df['label'].nunique() < 2:
             print("Warning: Only one class present in the temp dataset for val/test split. Stratification may not work.")
             val_df, test_df = train_test_split(temp_df, test_size=val_test_ratio, random_state=random_seed)
        else:
             val_df, test_df = train_test_split(temp_df, test_size=val_test_ratio, random_state=random_seed, stratify=temp_df['label'])
    else: # No validation or test set needed based on ratio
        val_df = pd.DataFrame(columns=df_all.columns)
        test_df = pd.DataFrame(columns=df_all.columns)


    print(f"Data Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Create datasets
    train_dataset = WSIFeatureDataset(train_df.reset_index(drop=True), feature_file_map, label_map, aggregation_fn)
    val_dataset = WSIFeatureDataset(val_df.reset_index(drop=True), feature_file_map, label_map, aggregation_fn)
    test_dataset = WSIFeatureDataset(test_df.reset_index(drop=True), feature_file_map, label_map, aggregation_fn)

    # Check if datasets are empty after filtering
    if len(train_dataset) == 0: print("Warning: Training dataset is empty after filtering for valid features.")
    if len(val_dataset) == 0: print("Warning: Validation dataset is empty after filtering for valid features.")
    if len(test_dataset) == 0: print("Warning: Test dataset is empty after filtering for valid features.")


    # Define collate_fn (usually None if aggregation happens in Dataset)
    collate_fn = None
    if aggregation_fn is None:
        print("Warning: aggregation_fn is None. Model must handle bags of features. Implement custom collate_fn if needed.")
        # Define collate_fn here if needed

    # Create DataLoaders (handle empty datasets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn) if len(train_dataset) > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn) if len(test_dataset) > 0 else None

    return train_loader, val_loader, test_loader