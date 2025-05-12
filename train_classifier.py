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
    df_all_labels = pd.read_csv(config.CSV_PATH)
    unique_labels = sorted(df_all_labels['label'].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    actual_num_classes = len(unique_labels)
    if actual_num_classes != config.NUM_CLASSES:
        print(f"Warning: config.NUM_CLASSES is {config.NUM_CLASSES} but found {actual_num_classes} unique labels in CSV. Using {actual_num_classes}.")
        current_num_classes = actual_num_classes
    else:
        current_num_classes = config.NUM_CLASSES

    print("Label map:", label_map)

    # Get feature dimension - inspect one HDF5 file
    # This is important for defining the classifier's input_dim
    feature_dim = None
    df_map = pd.read_csv(os.path.join(config.EXTRACTED_FEATURES_DIR, "feature_file_map.csv"))
    if not df_map.empty:
        first_feature_file = df_map['feature_path'].iloc[0]
        try:
            with h5py.File(first_feature_file, 'r') as hf:
                feature_dim = hf['features'].shape[1] # (N_patches, Feature_dim)
            print(f"Determined feature dimension from HDF5: {feature_dim}")
        except Exception as e:
            print(f"Could not determine feature dimension from {first_feature_file}: {e}")
            # Fallback to a known dimension from your ConvNeXt model if error
            # For ConvNeXt-Large, output is often 1024 or 1536 before head
            # Check your specific SSL pretraining output dimension
            feature_dim = 1024 # Example, verify this for your convnextv2l SSL model
            print(f"Using fallback feature dimension: {feature_dim}")
    else:
        print("ERROR: feature_file_map.csv is empty. Cannot determine feature dimension. Run feature extraction.")
        return

    if feature_dim is None:
        print("ERROR: Could not determine feature dimension. Exiting.")
        return


    aggregation_method_instance = get_aggregation_function(
        config.AGGREGATION_METHOD,
        feature_dim=feature_dim,
        hidden_dim=128, # Example for attention, tune this
        num_classes=current_num_classes # For attention if it has internal classifier bits
    )

    train_loader, val_loader, test_loader = get_data_loaders(
        csv_path=config.CSV_PATH,
        feature_file_map_path=os.path.join(config.EXTRACTED_FEATURES_DIR, "feature_file_map.csv"),
        label_map=label_map,
        aggregation_fn=aggregation_method_instance if isinstance(aggregation_method_instance, nn.Module) else aggregation_method_instance,
        batch_size=config.BATCH_SIZE,
        split_ratio=config.SPLIT_RATIO,
        random_seed=config.RANDOM_SEED
    )

    # --- 2. Define Model, Loss, Optimizer ---
    # If aggregation is a simple function (mean, max), classifier takes aggregated features
    # If aggregation is an nn.Module (like AttentionGated used as an aggregator),
    # WSIFeatureDataset should apply its forward method.
    
    # The input_dim for SimpleClassifier should be the dimension of the *aggregated* WSI feature.
    # For mean/max/attention (as implemented), this is still `feature_dim`.
    classifier_model = SimpleClassifier(input_dim=feature_dim, num_classes=current_num_classes).to(config.DEVICE)
    
    # If AttentionGated is used as the *entire model* (MIL style):
    # mil_model = AttentionGated(feature_dim, hidden_dim=128, num_classes=current_num_classes).to(config.DEVICE)
    # In this case, WSIFeatureDataset should return bags of features, and mil_model handles aggregation + classification.
    # The current setup assumes WSIFeatureDataset provides *already aggregated* features to SimpleClassifier.

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)


    # --- 3. Training Loop ---
    best_val_auc = 0.0 # Or best_val_loss if AUC is not primary
    best_val_loss = float('inf')

    aggregation_module_for_training = None
    if isinstance(aggregation_method_instance, nn.Module):
        # If the aggregation itself has trainable parameters (e.g., attention)
        # And it's NOT part of the main classifier_model (i.e., used in Dataset)
        # This scenario is less common if classifier_model expects aggregated features.
        # More likely: attention IS the model, or part of the model.
        # For now, let's assume `classifier_model` is the only thing trained by `optimizer`.
        # If AttentionGated is used in the dataset, its parameters won't be trained by this optimizer
        # unless you add them: optimizer = optim.Adam(list(classifier_model.parameters()) + list(aggregation_method_instance.parameters()), ...)
        # This gets complex. Simpler: have AttentionGated BE the model, or SimpleClassifier operate on pre-aggregated.
        # Sticking to: SimpleClassifier on pre-aggregated features.
        pass


    print("\nStarting training...")
    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_epoch(classifier_model, train_loader, criterion, optimizer, config.DEVICE, aggregation_module_for_training)
        val_loss, val_acc, val_auc, val_f1 = validate_epoch(classifier_model, val_loader, criterion, config.DEVICE, aggregation_module_for_training)

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")

        scheduler.step(val_loss)

        # Save best model (e.g., based on validation AUC or loss)
        current_metric_is_better = False
        if config.NUM_CLASSES == 2: # Use AUC for binary
            if val_auc > best_val_auc :
                best_val_auc = val_auc
                current_metric_is_better = True
                print(f"  New best validation AUC: {best_val_auc:.4f}")
        else: # Use validation loss for multiclass or if AUC is problematic
             if val_loss < best_val_loss:
                best_val_loss = val_loss
                current_metric_is_better = True
                print(f"  New best validation loss: {best_val_loss:.4f}")

        if current_metric_is_better:
            model_save_path = os.path.join(config.TRAINED_MODELS_DIR, f"best_classifier_{config.AGGREGATION_METHOD}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'auc': val_auc, # Store AUC even if not primary for multiclass
                'label_map': label_map,
                'feature_dim': feature_dim,
                'aggregation_method': config.AGGREGATION_METHOD
            }, model_save_path)
            print(f"  Model saved to {model_save_path}")

    # --- 4. Final Evaluation on Test Set ---
    print("\nLoading best model for final test evaluation...")
    best_model_path = os.path.join(config.TRAINED_MODELS_DIR, f"best_classifier_{config.AGGREGATION_METHOD}.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
        # Re-initialize model to ensure correct architecture if params changed
        loaded_feature_dim = checkpoint.get('feature_dim', feature_dim) # Get saved dim
        final_model = SimpleClassifier(input_dim=loaded_feature_dim, num_classes=current_num_classes).to(config.DEVICE)
        final_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model loaded from epoch {checkpoint['epoch']+1} with Val AUC: {checkpoint.get('auc', -1):.4f}, Val Loss: {checkpoint.get('loss', -1):.4f}")

        test_loss, test_acc, test_auc, test_f1 = validate_epoch(final_model, test_loader, criterion, config.DEVICE)
        print(f"\nTest Set Performance:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Test F1-score: {test_f1:.4f}")

        results_df = pd.DataFrame([{
            'aggregation': config.AGGREGATION_METHOD,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_auc': test_auc,
            'test_f1': test_f1,
            'best_val_auc_at_save': checkpoint.get('auc', -1),
            'best_val_loss_at_save': checkpoint.get('loss', -1),
            'num_epochs_run': config.NUM_EPOCHS,
            'label_map': label_map,
        }])
        results_df.to_csv(os.path.join(config.RESULTS_DIR, f"test_results_{config.AGGREGATION_METHOD}.csv"), index=False)
        print(f"Test results saved to {os.path.join(config.RESULTS_DIR, f'test_results_{config.AGGREGATION_METHOD}.csv')}")

    else:
        print(f"ERROR: Best model not found at {best_model_path}. Skipping test evaluation.")


if __name__ == '__main__':
    # Ensure Trident's factory knows your custom encoder if it's not globally registered
    # This is crucial if `feature_extractor.py` isn't run as a separate step first,
    # or if the registration is not in an __init__.py that gets loaded.
    # from trident.patch_encoder_models.load import MyConvNeXtV2LargeSSLEncoder # Assuming this class definition exists
    # from trident.patch_encoder_models import encoder_factory # Ensure this is the same factory instance
    # if 'convnextv2l' not in encoder_factory.encoder_zoo: # Check if already registered
    #     encoder_factory.register_encoder('convnextv2l', MyConvNeXtV2LargeSSLEncoder)
    #     print("Registered MyConvNeXtV2LargeSSLEncoder with factory as 'convnextv2l'")

    # First, ensure features are extracted if they don't exist
    feature_map_file = os.path.join(config.EXTRACTED_FEATURES_DIR, "feature_file_map.csv")
    if not os.path.exists(feature_map_file) or os.path.getsize(feature_map_file) == 0 :
        print("Feature map not found or empty. Running batch feature extraction first...")
        from feature_extractor import run_batch_feature_extraction # Make sure this can be imported
        run_batch_feature_extraction()
        if not os.path.exists(feature_map_file) or os.path.getsize(feature_map_file) == 0:
            print("ERROR: Feature extraction did not produce a feature map. Aborting.")
            exit()
    else:
        print(f"Found existing feature map: {feature_map_file}. Skipping feature extraction.")

    main()