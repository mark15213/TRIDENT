# aggregation.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def mean_aggregation(patch_features):
    """patch_features: (N_patches, Feature_dim)"""
    if patch_features.nelement() == 0 or patch_features.shape[0] == 0 : # handles tensor([]) or tensor of shape (0, D)
        # Determine feature_dim, this is a bit tricky if empty.
        # If patch_features is (0, D), patch_features.shape[1] gives D.
        # If patch_features is tensor([]), it has no shape[1].
        # Assume a default or get from elsewhere if truly empty.
        feature_dim = patch_features.shape[1] if patch_features.ndim > 1 and patch_features.shape[1] > 0 else 1024 # Fallback
        return torch.zeros(feature_dim, device=patch_features.device, dtype=patch_features.dtype)
    return torch.mean(patch_features, dim=0)

def max_aggregation(patch_features):
    """patch_features: (N_patches, Feature_dim)"""
    if patch_features.nelement() == 0 or patch_features.shape[0] == 0 :
        feature_dim = patch_features.shape[1] if patch_features.ndim > 1 and patch_features.shape[1] > 0 else 1024
        return torch.zeros(feature_dim, device=patch_features.device, dtype=patch_features.dtype)
    return torch.max(patch_features, dim=0)[0]


# --- CLAM-like Attention (Simplified Example) ---
class AttentionGated(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes, dropout_rate=0.25):
        super(AttentionGated, self).__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)
        # Classifier part is separate in CLAM, here we integrate for simplicity if used as aggregator
        # For a pure aggregator, you might not need the classifier_head here.
        # self.classifier_head = nn.Linear(feature_dim, num_classes) # Or hidden_dim if using attention_V output
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Patch features, shape (N_patches, feature_dim)
        Returns:
            torch.Tensor: Aggregated WSI feature, shape (feature_dim)
            torch.Tensor: Attention scores, shape (N_patches, 1) - for visualization/interpretation
        """
        if x.nelement() == 0 or x.shape[0] == 0: # No patches
            feature_dim = x.shape[1] if x.ndim > 1 and x.shape[1] > 0 else self.attention_V[0].in_features # Try to get in_features
            return torch.zeros(feature_dim, device=x.device, dtype=x.dtype) # , torch.empty(0, 1, device=x.device, dtype=x.dtype)

        # x is (num_patches, feature_dim)
        A_V = self.attention_V(x)  # (num_patches, hidden_dim)
        A_U = self.attention_U(x)  # (num_patches, hidden_dim)
        A = self.attention_weights(A_V * A_U) # (num_patches, 1) Element-wise multiplication
        A = F.softmax(A, dim=0)  # Normalize attention scores over patches

        # Weighted sum of features
        # x is (num_patches, feature_dim), A is (num_patches, 1)
        # We want to multiply each feature vector by its attention score and sum up.
        # (1, num_patches) x (num_patches, feature_dim) -> (1, feature_dim)
        m = torch.mm(A.T, x) # (1, feature_dim)
        
        # m = self.dropout(m) # Apply dropout to the aggregated feature
        # If using a classifier head directly here:
        # logits = self.classifier_head(m) # (1, num_classes)
        # return logits.squeeze(0), A # Return logits and attention scores

        return m.squeeze(0) # Return aggregated feature, attention scores can be returned too if needed

# Factory for aggregation methods
def get_aggregation_function(method_name, feature_dim=None, hidden_dim=None, num_classes=None):
    if method_name == "mean":
        return mean_aggregation
    elif method_name == "max":
        return max_aggregation
    elif method_name == "attention":
        if feature_dim is None or hidden_dim is None: # num_classes not strictly needed for aggregator
            raise ValueError("feature_dim and hidden_dim must be provided for attention aggregation.")
        # This returns an nn.Module instance, so it needs to be instantiated.
        # The dataset loader will call its forward method.
        return AttentionGated(feature_dim, hidden_dim, num_classes if num_classes else 2) # Pass num_classes for internal classifier if any
    else:
        raise ValueError(f"Unknown aggregation method: {method_name}")