# File: trident/patch_encoder_models/convnextv2_model.py
# (Or whatever path you choose within your inference framework)

import torch
import torch.nn as nn
import torch.nn.functional as F # F is used by the LayerNorm you provided
from timm.models.layers import trunc_normal_, DropPath

# --- BEGIN: Copied and integrated LayerNorm and GRN from your utils.py ---
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, ) # This should be an int or a tuple of ints

        # If normalized_shape is an int (common case for LayerNorm on the last dim)
        # then self.normalized_shape should be (normalized_shape,)
        # If it's already a tuple, it's fine.
        # PyTorch's nn.LayerNorm takes normalized_shape as int or list/tuple.
        # F.layer_norm expects normalized_shape to be a tuple of ints for the shape of the normalized dims.

    def forward(self, x):
        if self.data_format == "channels_last":
            # For F.layer_norm, normalized_shape is the shape of the last D dimensions.
            # If x is (N, H, W, C) and normalized_shape is C (an int),
            # then F.layer_norm expects (C,) for its normalized_shape argument.
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # x is (N, C, H, W)
            # We want to normalize over C.
            # Permute to (N, H, W, C) for standard LayerNorm application if needed,
            # or apply directly if LayerNorm is designed for channels_first.
            # Your original channels_first implementation:
            u = x.mean(1, keepdim=True) # Mean over C
            s = (x - u).pow(2).mean(1, keepdim=True) # Variance over C
            x = (x - u) / torch.sqrt(s + self.eps)
            # self.weight and self.bias are (C,). Need to be broadcastable.
            # (C,) reshaped to (1, C, 1, 1) for (N,C,H,W) or (C, None, None) for weight below
            x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        # For (N, H, W, C) input, dim is C. gamma/beta are (1,1,1,C)
        # For (N, C, H, W) input after permute, dim is C.
        # The GRN in ConvNeXtV2 is applied after permuting to (N, H, W, C).
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # x is expected to be (N, H, W, C) by the original ConvNeXtV2 block before GRN
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True) # Norm over H, W. Gx shape (N, 1, 1, C)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6) # Mean over C. Nx shape (N, 1, 1, C)
        return self.gamma * (x * Nx) + self.beta + x
# --- END: Copied and integrated LayerNorm and GRN ---


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        # LayerNorm is applied after permute to (N, H, W, C), so data_format="channels_last"
        # and normalized_shape is `dim` (the channel dimension)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim) # GRN takes the expanded dimension
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_tensor = x # Renamed to avoid conflict with 'input' keyword
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input_tensor + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.dims = dims # Store dims for access by patch encoder wrapper
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first") # LayerNorm on C dimension
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"), # LayerNorm on C dimension
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Final LayerNorm after stages, before GAP. This is applied on the C dimension of (N,C,H,W)
        # Or, if features are permuted to (N, H, W, C) then mean, then normalized over C.
        # The original forward_features: x.mean([-2, -1]) -> (N, C), then self.norm (nn.LayerNorm(dims[-1]))
        # This nn.LayerNorm will normalize over the last dimension, which is C. This is correct.
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer for features
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        if num_classes > 0 : # Only initialize head if it's a real classification head
            if hasattr(self.head, 'weight') and self.head.weight is not None:
                 self.head.weight.data.mul_(head_init_scale)
            if hasattr(self.head, 'bias') and self.head.bias is not None:
                 self.head.bias.data.mul_(head_init_scale)
        elif isinstance(self.head, nn.Linear) and num_classes == 0: # common practice for feature extraction
            self.head = nn.Identity()


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: # Check if bias exists
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x) # x is (N, C, H, W)
        # global average pooling, (N, C, H, W) -> (N, C)
        # then norm. self.norm is nn.LayerNorm(dims[-1])
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# Factory functions
def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs): # Name from your original code
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model