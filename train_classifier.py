# train_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np

import config
from dataset import get_data_loaders # WSIFeatureDataset implicitly used here
from aggregation import get_aggregation_function # For selecting aggregation method

# --- Define a simple classifier model ---
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_epoch(model, dataloader, criterion, optimizer, device, aggregation_module=None):
    model.train()
    if aggregation_module: # If attention aggregator is an nn.Module
        aggregation_module.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, batch in enumerate(dataloader):
        # WSIFeatureDataset already returns aggregated 'wsi_feature' if aggregation_fn is set,
        # or it returns bag of features if aggregation_fn is None.
        # If using AttentionGated as the model itself (MIL style):
        # features_bag = batch['wsi_feature'].to(device) # This would be a list of tensors or padded tensor
        # labels = batch['label'].to(device)
        # aggregated_feature, attention_scores = model(features_bag) # Model itself does aggregation
        # logits = model.classifier_head(aggregated_feature) # If classifier is separate
        
        # If aggregation is done in Dataset or by a separate module before classifier:
        wsi_features = batch['wsi_feature'].to(device) # Should be (batch_size, feature_dim)
        labels = batch['label'].to(device)

        if aggregation_module and isinstance(aggregation_module, nn.Module) and config.AGGREGATION_METHOD == "attention":
            # This path is if 'wsi_feature' from dataset is still per-patch and needs aggregation by the module
            # However, WSIFeatureDataset with an aggregation_fn should already provide aggregated features.
            # This logic needs to be clear: either Dataset aggregates, or the model does.
            # Let's assume WSIFeatureDataset provides aggregated features if config.AGGREGATION_METHOD is 'mean' or 'max'.
            # If 'attention', WSIFeatureDataset might pass the nn.Module which then needs to be applied.
            # For simplicity here, assume WSIFeatureDataset's aggregation_fn handles it.
            pass


        optimizer.zero_grad()
        outputs = model(wsi_features) # (batch_size, num_classes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * wsi_features.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device, aggregation_module=None):
    model.eval()
    if aggregation_module:
        aggregation_module.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = [] # For AUC

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            wsi_features = batch['wsi_feature'].to(device)
            labels = batch['label'].to(device)
            
            # Similar logic for aggregation as in train_epoch if needed

            outputs = model(wsi_features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * wsi_features.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if probs.shape[1] == 2: # Binary classification
                all_probs.extend(probs[:, 1].cpu().numpy()) # Prob of positive class
            else: # Multiclass
                all_probs.extend(probs.cpu().numpy())


    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    auc = -1
    if config.NUM_CLASSES == 2 and len(all_probs) > 0 and len(np.unique(all_labels)) > 1 :
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError as e:
            print(f"Could not calculate AUC: {e}") # e.g. only one class present in y_true
            auc = -1 # Or some other placeholder
    elif config.NUM_CLASSES > 2 and len(all_probs) > 0 and len(np.unique(all_labels)) > 1:
        try:
            # For multiclass, use one-vs-rest or one-vs-one, average='weighted' or 'macro'
            # Ensure all_labels are 0,1,...N-1 and all_probs is (samples, N_classes)
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        except ValueError as e:
            print(f"Could not calculate AUC for multiclass: {e}")
            auc = -1


    f1 = f1_score(all_labels, all_preds, average='weighted' if config.NUM_CLASSES > 2 else 'binary')

    return epoch_loss, epoch_acc, auc, f1


def main():
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if config.DEVICE_STR.startswith("cuda"):
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    os.makedirs(config.TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # --- 1. Prepare Data ---
    try:
        df_all_labels = pd.read_csv(config.CSV_PATH)
        if 'label' not in df_all_labels.columns:
             raise ValueError("CSV must contain 'label' column.")
    except FileNotFoundError:
        print(f"Error: Input CSV {config.CSV_PATH} not found.")
        return
    except Exception as e:
        print(f"Error reading input CSV {config.CSV_PATH}: {e}")
        return

    # Automatically create label map for 'cancer', 'healthy' or any other labels found
    unique_labels = sorted(df_all_labels['label'].astype(str).unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    actual_num_classes = len(unique_labels)

    # Verify NUM_CLASSES in config matches data
    if actual_num_classes == 0:
        print("Error: No labels found in the CSV file.")
        return
    elif actual_num_classes != config.NUM_CLASSES:
        print(f"Warning: config.NUM_CLASSES is {config.NUM_CLASSES} but found {actual_num_classes} unique labels in CSV ({unique_labels}). Using {actual_num_classes}.")
    current_num_classes = actual_num_classes # Use the actual number found

    print("Using Label map:", label_map)

    # Determine feature dimension from HDF5 files
    feature_dim = None
    expected_feature_dim = 1536
    feature_map_path = os.path.join(config.EXTRACTED_FEATURES_DIR, "feature_file_map.csv")
    if not os.path.exists(feature_map_path):
        print(f"ERROR: Feature map file not found at {feature_map_path}. Run feature extraction first.")
        return
    try:
        df_map = pd.read_csv(feature_map_path)
        if not df_map.empty and 'feature_path' in df_map.columns:
            # Find the first valid feature file path in the map
            first_valid_feature_file = None
            for fpath in df_map['feature_path']:
                if isinstance(fpath, str) and os.path.exists(fpath):
                    first_valid_feature_file = fpath
                    break

            if first_valid_feature_file:
                try:
                    with h5py.File(first_valid_feature_file, 'r') as hf:
                        if 'features' in hf:
                             # Handle potentially empty features dataset when determining dim
                             if hf['features'].shape[0] > 0:
                                 feature_dim = hf['features'].shape[1]
                             elif hf['features'].ndim > 1 : # Shape is (0, D)
                                 feature_dim = hf['features'].shape[1]

                    if feature_dim is not None and feature_dim > 0:
                        print(f"Determined feature dimension from HDF5: {feature_dim}")
                    else:
                        print(f"Warning: Could not determine feature dimension from dataset 'features' in {first_valid_feature_file}. Shape: {hf['features'].shape if 'features' in hf else 'Not found'}")

                except Exception as e:
                    print(f"Could not read features from {first_valid_feature_file} to determine dimension: {e}")
            else:
                print("Warning: No valid feature file paths found in the feature map.")

        else:
            print("Warning: Feature map file is empty or missing 'feature_path' column.")

    except Exception as e:
        print(f"Error processing feature map file {feature_map_path}: {e}")


    # Fallback or error if dimension couldn't be determined
    if feature_dim is None or feature_dim <= 0:
        # Check your ConvNeXtV2LargeSSLEncoder output dimension
        # Typical values for ConvNeXt-L are 1024 or 1536
        feature_dim = 1024 # Example fallback - **VERIFY THIS IS CORRECT**
        print(f"Could not reliably determine feature dimension. Using fallback: {feature_dim}")
        # Alternatively, exit:
        # print("ERROR: Could not determine feature dimension. Exiting.")
        # return


    # Get aggregation function/module
    aggregation_method_instance = get_aggregation_function(
        config.AGGREGATION_METHOD,
        feature_dim=feature_dim,
        hidden_dim=128, # Example for attention
        num_classes=current_num_classes
    )

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_data_loaders(
        csv_path=config.CSV_PATH,
        feature_file_map_path=feature_map_path, # Use variable defined above
        label_map=label_map,
        aggregation_fn=aggregation_method_instance, # Pass the instance/function
        batch_size=config.BATCH_SIZE,
        split_ratio=config.SPLIT_RATIO,
        random_seed=config.RANDOM_SEED
    )

    # Check if dataloaders are usable
    if train_loader is None:
        print("Error: Training dataloader could not be created (likely no valid training data). Aborting.")
        return

    # --- 2. Define Model, Loss, Optimizer ---
    print(f"Initializing classifier with input_dim={feature_dim}, num_classes={current_num_classes}")
    classifier_model = SimpleClassifier(input_dim=feature_dim, num_classes=current_num_classes).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True) # Monitor val_loss

    # --- 3. Training Loop ---
    best_val_metric = float('inf') # Use loss as primary metric
    best_val_auc = 0.0 # Keep track of AUC for binary cases

    print("\nStarting training...")
    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_epoch(classifier_model, train_loader, criterion, optimizer, config.DEVICE)

        val_loss, val_acc, val_auc, val_f1 = -1, -1, -1, -1 # Default values
        if val_loader: # Only validate if validation set exists
            val_loss, val_acc, val_auc, val_f1 = validate_epoch(classifier_model, val_loader, criterion, config.DEVICE)
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")
            scheduler.step(val_loss) # Step scheduler based on validation loss
        else:
            # If no validation loader, just print training stats
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"(No validation set)")
            # How to save best model without validation? Save based on train loss? Or just save last?
            # Saving based on train loss is usually not recommended. Save last epoch is an option.
            # Let's stick to saving based on validation metric if val_loader exists.


        # Save best model based on validation loss (if val_loader exists)
        if val_loader and val_loss < best_val_metric:
            best_val_metric = val_loss
            best_val_auc = val_auc # Store corresponding AUC
            print(f"  New best validation loss: {best_val_metric:.4f}")
            model_save_path = os.path.join(config.TRAINED_MODELS_DIR, f"best_classifier_{config.AGGREGATION_METHOD}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_metric,
                'auc': best_val_auc,
                'label_map': label_map,
                'feature_dim': feature_dim,
                'aggregation_method': config.AGGREGATION_METHOD
            }, model_save_path)
            print(f"  Model saved to {model_save_path}")

    # --- 4. Final Evaluation on Test Set ---
    print("\nLoading best model for final test evaluation...")
    best_model_path = os.path.join(config.TRAINED_MODELS_DIR, f"best_classifier_{config.AGGREGATION_METHOD}.pth")
    if os.path.exists(best_model_path) and test_loader: # Check if best model was saved and test set exists
        checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
        loaded_feature_dim = checkpoint.get('feature_dim', feature_dim) # Use saved dim
        final_model = SimpleClassifier(input_dim=loaded_feature_dim, num_classes=current_num_classes).to(config.DEVICE)
        final_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model loaded from epoch {checkpoint.get('epoch', -1)+1} with Val Loss: {checkpoint.get('loss', -1):.4f}, Val AUC: {checkpoint.get('auc', -1):.4f}")

        test_loss, test_acc, test_auc, test_f1 = validate_epoch(final_model, test_loader, criterion, config.DEVICE)
        print(f"\nTest Set Performance:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Test F1-score: {test_f1:.4f}")

        # Save test results
        results_df = pd.DataFrame([{
            'aggregation': config.AGGREGATION_METHOD,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_auc': test_auc,
            'test_f1': test_f1,
            'best_val_loss_at_save': checkpoint.get('loss', -1),
            'best_val_auc_at_save': checkpoint.get('auc', -1),
            'num_epochs_run': config.NUM_EPOCHS, # Or checkpoint['epoch']+1 if using early stopping
            'label_map': label_map,
        }])
        results_path = os.path.join(config.RESULTS_DIR, f"test_results_{config.AGGREGATION_METHOD}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Test results saved to {results_path}")

    elif not os.path.exists(best_model_path):
         print("Skipping test evaluation: Best model checkpoint not found.")
    elif not test_loader:
         print("Skipping test evaluation: No test data loader created.")


if __name__ == '__main__':
    # --- Check/Run Feature Extraction ---
    feature_map_file = os.path.join(config.EXTRACTED_FEATURES_DIR, "feature_file_map.csv")
    run_extraction = False
    if not os.path.exists(feature_map_file):
        print("Feature map file not found. Feature extraction will be run.")
        run_extraction = True
    else:
        try:
            # Check if map is valid and non-empty
            df_map_check = pd.read_csv(feature_map_file)
            if df_map_check.empty or 'wsi_path' not in df_map_check.columns or 'feature_path' not in df_map_check.columns:
                 print("Feature map file is empty or invalid. Feature extraction will be run.")
                 run_extraction = True
            else:
                 print(f"Found existing feature map: {feature_map_file}. Verifying file existence...")
                 # Optional: Add a check here to verify a few feature files actually exist
                 # num_to_check = 5
                 # files_exist = True
                 # for fpath in df_map_check['feature_path'].head(num_to_check):
                 #    if not os.path.exists(fpath):
                 #        print(f"Feature file {fpath} listed in map does not exist. Re-running extraction.")
                 #        files_exist = False
                 #        run_extraction = True
                 #        break
                 # if files_exist:
                 #    print("Feature files seem consistent with map. Skipping feature extraction.")
                 print("Assuming feature map is valid. Skipping feature extraction.")


        except Exception as e:
            print(f"Error reading feature map file ({e}). Feature extraction will be run.")
            run_extraction = True

    if run_extraction:
        print("Running batch feature extraction...")
        from feature_extractor import run_batch_feature_extraction # Ensure import works
        run_batch_feature_extraction()
        # Check again if map was created successfully
        if not os.path.exists(feature_map_file):
            print("ERROR: Feature extraction ran but did not produce a feature map. Aborting.")
            exit()
        else:
             try:
                 df_map_check = pd.read_csv(feature_map_file)
                 if df_map_check.empty:
                      print("ERROR: Feature extraction ran but the resulting feature map is empty. Aborting.")
                      exit()
             except Exception:
                  print("ERROR: Feature extraction ran but could not read the resulting feature map. Aborting.")
                  exit()

    # --- Run Training ---
    main()