import traceback
from abc import abstractmethod
from typing import Literal, Optional, Tuple, Callable, Dict, Any # Added Dict, Any, Callable, Tuple
import torch
import torch.nn as nn # Added nn
import os

# --- BEGIN: Import your ConvNeXtV2 model and its factory functions ---
# Adjust this import based on your project structure.
# For example, if convnextv2_model.py is in a 'custom_models' subdirectory
# of where 'trident' is, you might use:
# from trident.custom_models.convnextv2_model import (
try:
    # Assuming convnextv2_model.py is in the same directory or a findable path
    # This is a common pattern in libraries like timm
    # If your convnextv2_model.py is part of the trident.patch_encoder_models structure:
    # from .convnextv2_model import (
    # Or if it's a top-level module in your project:
    # from my_project_convnext_models.convnextv2_model import (

    # For demonstration, assuming it's accessible like this:
    from trident.patch_encoder_models.convnextv2_model import ( # YOU WILL LIKELY NEED TO CHANGE THIS IMPORT PATH
        ConvNeXtV2, convnextv2_atto, convnextv2_femto, convnext_pico,
        convnextv2_nano, convnextv2_tiny, convnextv2_base, convnextv2_large, convnextv2_huge
    )
except ImportError:
    print("Warning: Could not import ConvNeXtV2 model. Please ensure convnextv2_model.py is in the correct path.")
    print("Falling back to dummy ConvNeXtV2 for structure demonstration.")
    # --- Dummy definitions for demonstration if import fails ---
    class GRN(torch.nn.Module):
        def __init__(self, dim): super().__init__(); self.gamma = torch.nn.Parameter(torch.zeros(1, 1, 1, dim)); self.beta = torch.nn.Parameter(torch.zeros(1, 1, 1, dim))
        def forward(self, x): gx = torch.norm(x, p=2, dim=(1,2), keepdim=True); nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6); return self.gamma * (x * nx) + self.beta + x
    class LayerNorm(torch.nn.Module):
        def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
            super().__init__(); self.normalized_shape = (normalized_shape,); self.eps = eps; self.data_format = data_format
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape)); self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        def forward(self, x):
            if self.data_format == "channels_last": u = x.mean(-1, keepdim=True); s = (x - u).pow(2).mean(-1, keepdim=True); x = (x - u) / torch.sqrt(s + self.eps); x = self.weight * x + self.bias; return x
            elif self.data_format == "channels_first": x_perm = x.permute(0, 2, 3, 1); u = x_perm.mean(-1, keepdim=True); s = (x_perm - u).pow(2).mean(-1, keepdim=True); x_perm = (x_perm - u) / torch.sqrt(s + self.eps); x_perm = self.weight * x_perm + self.bias; return x_perm.permute(0, 3, 1, 2)
    class Block(torch.nn.Module):
        def __init__(self, dim, drop_path=0.): super().__init__(); self.dwconv = torch.nn.Conv2d(dim, dim, 7, 3, groups=dim); self.norm = LayerNorm(dim); self.pwconv1 = torch.nn.Linear(dim, 4*dim); self.act = torch.nn.GELU(); self.grn = GRN(4*dim); self.pwconv2 = torch.nn.Linear(4*dim, dim); self.drop_path = torch.nn.Identity()
        def forward(self, x): inp = x; x = self.dwconv(x); x = x.permute(0,2,3,1); x=self.norm(x); x=self.pwconv1(x); x=self.act(x); x=self.grn(x); x=self.pwconv2(x); x=x.permute(0,3,1,2); x=inp+self.drop_path(x); return x
    class ConvNeXtV2(torch.nn.Module):
        def __init__(self, in_chans=3, num_classes=1000, depths=[1], dims=[40], drop_path_rate=0., head_init_scale=1., **kwargs): # Added missing args
            super().__init__(); self.dims = dims; self.downsample_layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(in_chans, dims[0], 4, 4), LayerNorm(dims[0], data_format="channels_first"))])
            self.stages = torch.nn.ModuleList([torch.nn.Sequential(*[Block(dim=dims[0]) for _ in range(depths[0])])])
            self.norm = torch.nn.LayerNorm(dims[-1]); self.head = torch.nn.Linear(dims[-1], num_classes)
            self.apply(self._init_weights) # Added apply
            if hasattr(self.head, 'weight'): self.head.weight.data.mul_(head_init_scale) # Added safety check
            if hasattr(self.head, 'bias') and self.head.bias is not None: self.head.bias.data.mul_(head_init_scale)
        def forward_features(self, x): x = self.downsample_layers[0](x); x = self.stages[0](x); return self.norm(x.mean([-2,-1]))
        def forward(self, x): x = self.forward_features(x); return self.head(x)
        def _init_weights(self,m): # simplified
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # Corrected trunc_normal_ import or simplified init
                torch.nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def convnextv2_atto(**kwargs): return ConvNeXtV2(depths=[2,2,6,2], dims=[40,80,160,320], **kwargs)
    def convnextv2_femto(**kwargs): return ConvNeXtV2(depths=[2,2,6,2], dims=[48,96,192,384], **kwargs)
    def convnext_pico(**kwargs): return ConvNeXtV2(depths=[2,2,6,2], dims=[64,128,256,512], **kwargs) # Corrected name
    def convnextv2_nano(**kwargs): return ConvNeXtV2(depths=[2,2,8,2], dims=[80,160,320,640], **kwargs)
    def convnextv2_tiny(**kwargs): return ConvNeXtV2(depths=[3,3,9,3], dims=[96,192,384,768], **kwargs)
    def convnextv2_base(**kwargs): return ConvNeXtV2(depths=[3,3,27,3], dims=[128,256,512,1024], **kwargs)
    def convnextv2_large(**kwargs): return ConvNeXtV2(depths=[3,3,27,3], dims=[192,384,768,1536], **kwargs)
    def convnextv2_huge(**kwargs): return ConvNeXtV2(depths=[3,3,27,3], dims=[352,704,1408,2816], **kwargs)
# --- END: Import your ConvNeXtV2 model ---


from trident.patch_encoder_models.utils.constants import get_constants
from trident.patch_encoder_models.utils.transform_utils import get_eval_transforms
from trident.IO import get_weights_path, has_internet_connection


"""
This file contains an assortment of pretrained patch encoders, all loadable via the encoder_factory() function.
"""

def encoder_factory(model_name: str, **kwargs) -> torch.nn.Module: # Added return type hint
    """
    Instantiate a patch encoder model by name.
    This factory function returns a pre-configured encoder model class based on the provided
    `model_name`. Each encoder is designed for extracting representations from image patches
    using specific backbones or pretraining strategies.

    Args:
        model_name (str): Name of the encoder to instantiate. Must be one of the following:
            - "conch_v1"
            - "conch_v15"
            - "uni_v1"
            - "uni_v2"
            - "ctranspath"
            - "phikon"
            - "phikon_v2"
            - "resnet50"
            - "gigapath"
            - "virchow"
            - "virchow2"
            - "hoptimus0"
            - "hoptimus1"
            - "musk"
            - "hibou_l"
            - "kaiko-vitb8"
            - "kaiko-vitb16"
            - "kaiko-vits8"
            - "kaiko-vits16"
            - "kaiko-vitl14"
            - "lunit-vits8"

        **kwargs: Optional keyword arguments passed directly to the encoder constructor. These
            may include parameters such as:
            - weights_path (str): Path to a local checkpoint (optional)
            - normalize (bool): Whether to normalize output embeddings (default: False)
            - with_proj (bool): Whether to apply the projection head (default: True)
            - any model-specific configuration parameters

    Returns:
        torch.nn.Module: An instance of the specified encoder model.

    Raises:
        ValueError: If `model_name` is not among the recognized encoder names.
    """
    enc_class: Optional[type[BasePatchEncoder]] = None

    if model_name == 'conch_v1':
        enc_class = Conchv1InferenceEncoder
    elif model_name == 'conch_v15':
        enc_class = Conchv15InferenceEncoder
    elif model_name == 'uni_v1':
        enc_class = UNIInferenceEncoder
    elif model_name == 'uni_v2':
        enc_class = UNIv2InferenceEncoder
    elif model_name == 'ctranspath':
        enc_class = CTransPathInferenceEncoder
    elif model_name == 'phikon':
        enc_class = PhikonInferenceEncoder
    elif model_name == 'resnet50':
        enc_class = ResNet50InferenceEncoder
    elif model_name == 'gigapath':
        enc_class = GigaPathInferenceEncoder
    elif model_name == 'virchow':
        enc_class = VirchowInferenceEncoder
    elif model_name == 'virchow2':
        enc_class = Virchow2InferenceEncoder
    elif model_name == 'hoptimus0':
        enc_class = HOptimus0InferenceEncoder
    elif model_name == 'hoptimus1':
        enc_class = HOptimus1InferenceEncoder
    elif model_name == 'phikon_v2':
        enc_class = Phikonv2InferenceEncoder
    elif model_name == 'musk':
        enc_class = MuskInferenceEncoder
    elif model_name == 'hibou_l':
        enc_class = HibouLInferenceEncoder
    elif model_name == 'kaiko-vitb8':
        enc_class = KaikoB8InferenceEncoder
    elif model_name == 'kaiko-vitb16':
        enc_class = KaikoB16InferenceEncoder
    elif model_name == 'kaiko-vits8':
        enc_class = KaikoS8InferenceEncoder
    elif model_name == 'kaiko-vits16':
        enc_class = KaikoS16InferenceEncoder
    elif model_name == 'kaiko-vitl14':
        enc_class = KaikoL14InferenceEncoder
    elif model_name == 'lunit-vits8':
        enc_class = LunitS8InferenceEncoder
    elif model_name == 'midnight12k':
        enc_class = Midnight12kInferenceEncoder
    # --- BEGIN: Add ConvNeXtV2 variants ---
    elif model_name.startswith('convnextv2_') or model_name == 'convnext_pico':
        enc_class = ConvNeXtV2PatchEncoder
        # Pass the specific variant name to the constructor
        # The ConvNeXtV2PatchEncoder's _build method will use this.
        variant = model_name.split('_')[-1] if model_name != 'convnext_pico' else 'pico'
        kwargs['model_variant'] = variant
    # --- END: Add ConvNeXtV2 variants ---
    else:
        raise ValueError(f"Unknown encoder name {model_name}")

    if enc_class is None: # Should not happen if logic is correct
        raise ValueError(f"Encoder class not assigned for model_name: {model_name}")
        
    return enc_class(**kwargs)


class BasePatchEncoder(torch.nn.Module):

    _has_internet = has_internet_connection()

    def __init__(self, weights_path: Optional[str] = None, **build_kwargs: Any): # Added Any for build_kwargs
        """
        Initialize BasePatchEncoder.
        Args:
            weights_path (Optional[str]): 
                Optional path to local model weights. If None, the model is loaded from the model registry or downloaded from Hugging Face Hub.
            **build_kwargs: 
                Additional keyword arguments passed to the `_build()` method to customize model creation.

        Attributes:
            enc_name (Optional[str]): Name of the encoder architecture (set during `_build()`).
            weights_path (Optional[str]): Path to local model weights (if provided).
            model (nn.Module): The instantiated encoder model.
            eval_transforms (Callable): Evaluation-time preprocessing transforms.
            precision (torch.dtype): Precision used for inference.
        """
        super().__init__()
        self.enc_name: Optional[str] = None
        self.weights_path: Optional[str] = weights_path
        # Model, transforms, and precision are tuples, so type hint accordingly
        _build_result = self._build(**build_kwargs)
        if not (isinstance(_build_result, tuple) and len(_build_result) == 3):
            raise ValueError(f"_build method for {self.__class__.__name__} must return a tuple of (model, eval_transforms, precision). Got: {_build_result}")
        self.model: Optional[torch.nn.Module]
        self.eval_transforms: Optional[Callable]
        self.precision: Optional[torch.dtype]
        self.model, self.eval_transforms, self.precision = _build_result


    def ensure_valid_weights_path(self, weights_path: Optional[str]): # Added Optional and type hint
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Expected checkpoint at '{weights_path}', but the file was not found.")

    def ensure_has_internet(self, enc_name: str): # Added type hint
        if not BasePatchEncoder._has_internet:
            raise ConnectionError( # Changed to ConnectionError for more specific type
                f"Internet connection does not seem available. Auto checkpoint download is disabled for {enc_name}.\n"
                f"To proceed, please manually download the weights for: {enc_name},\n"
                f"and place it in the model registry (e.g., `trident/patch_encoder_models/local_ckpts.json`) or provide the path via `weights_path`."
            )

    def _get_weights_path(self) -> Optional[str]: # Added return type hint
        """
        If self.weights_path is provided, use it.
        If not provided, check the model registry using self.enc_name.
        """
        if self.weights_path:
            self.ensure_valid_weights_path(self.weights_path)
            return self.weights_path
        else:
            if self.enc_name is None:
                # This case should ideally not happen if enc_name is set in _build before calling _get_weights_path
                # However, if a subclass calls _get_weights_path before self.enc_name is set,
                # it means it cannot look up in the registry.
                print(f"Warning: self.enc_name is None for {self.__class__.__name__}. Cannot look up weights in registry. "
                      "Ensure self.enc_name is set in _build, or provide weights_path directly.")
                return None # No specific path, implies HuggingFace download or no weights.

            weights_path_from_registry = get_weights_path('patch', self.enc_name) # type: ignore

            # get_weights_path might return None or an empty string if not found or not configured
            if weights_path_from_registry and os.path.isfile(weights_path_from_registry):
                return weights_path_from_registry
            elif weights_path_from_registry: # Path is configured but file doesn't exist
                 print(f"Warning: Weights path '{weights_path_from_registry}' for '{self.enc_name}' found in registry but file does not exist.")
                 return None # Or raise error, depending on desired behavior
            return None # Not in registry or path is empty/invalid

    def forward(self, x: torch.Tensor) -> torch.Tensor: # Added type hints
        """
        Can be overwritten if model requires special forward pass.
        """
        if self.model is None:
            raise RuntimeError(f"Model not built for {self.enc_name or self.__class__.__name__}. Call _build first.")
        z = self.model(x)
        return z

    @abstractmethod
    def _build(self, **build_kwargs: Any) -> Tuple[Optional[torch.nn.Module], Optional[Callable], Optional[torch.dtype]]: # Added Any, return type hint
        pass


# --- BEGIN: Add ConvNeXtV2PatchEncoder class ---
class ConvNeXtV2PatchEncoder(BasePatchEncoder):
    """
    Patch encoder using ConvNeXtV2 variants.
    """
    def __init__(self, model_variant: str = 'tiny', weights_path: Optional[str] = None, **build_kwargs: Any):
        """
        Initialize ConvNeXtV2PatchEncoder.

        Args:
            model_variant (str): Specific ConvNeXtV2 variant (e.g., 'tiny', 'base', 'pico').
            weights_path (Optional[str]): Path to local model weights.
            **build_kwargs: Additional arguments for _build.
        """
        # model_variant is crucial for _build, so pass it along.
        # BasePatchEncoder.__init__ will call self._build(**build_kwargs_for_super)
        # where build_kwargs_for_super will include model_variant.
        build_kwargs['model_variant'] = model_variant
        super().__init__(weights_path=weights_path, **build_kwargs)

    def _build(self,
                 model_variant: str = 'tiny',
                 in_chans: int = 3,
                 img_size: int = 224, # Default image size for transforms
                 drop_path_rate: float = 0.0,
                 head_init_scale: float = 1.0,
                 # pretrained_flag_for_timm: bool = True, # Use this if your ConvNeXtV2 factory takes 'pretrained'
                 **kwargs: Any
                ) -> Tuple[torch.nn.Module, Callable, torch.dtype]:

        self.enc_name = f'convnextv2_{model_variant}'
        if model_variant == "pico":
            self.enc_name = 'convnext_pico' # Match your specific naming

        _model_factories: Dict[str, Callable[..., ConvNeXtV2]] = {
            'atto': convnextv2_atto,
            'femto': convnextv2_femto,
            'pico': convnext_pico,
            'nano': convnextv2_nano,
            'tiny': convnextv2_tiny,
            'base': convnextv2_base,
            'large': convnextv2_large,
            'huge': convnextv2_huge,
        }

        if model_variant not in _model_factories:
            raise ValueError(f"Unsupported ConvNeXtV2 variant: {model_variant}. Available: {list(_model_factories.keys())}")

        # Instantiate the ConvNeXtV2 model
        # The num_classes is often set to 0 or a dummy value for feature extraction
        # Your ConvNeXtV2 definition needs to handle this (e.g., by having forward_features)
        model = _model_factories[model_variant](
            in_chans=in_chans,
            num_classes=0, # Or some other value if your model requires it, but head is usually ignored for patch features
            drop_path_rate=drop_path_rate,
            head_init_scale=head_init_scale
            #pretrained=pretrained_flag_for_timm # If your ConvNeXtV2 factory supports a pretrained flag for its own remote weights
        )

        # Load custom pretrained weights if a path is determined
        # self.weights_path is set by BasePatchEncoder.__init__
        # _get_weights_path() resolves it or gets from registry
        resolved_weights_path = self._get_weights_path() # This uses self.enc_name set above

        if resolved_weights_path:
            print(f"Loading weights for {self.enc_name} from: {resolved_weights_path}")
            try:
                checkpoint = torch.load(resolved_weights_path, map_location='cpu')
                # Adapt this based on how your weights are saved
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint # Assume it's the state_dict itself

                # Remove "module." prefix if present (from DataParallel/DDP)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                
                # Load into the model.
                # If you are only using forward_features, strict=False might be safer
                # if the checkpoint contains a head that model doesn't have (e.g. num_classes=0).
                missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                if missing_keys:
                    print(f"Warning: Missing keys for {self.enc_name}: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys for {self.enc_name}: {unexpected_keys}")
                print(f"Successfully loaded weights for {self.enc_name} from {resolved_weights_path}")

            except Exception as e:
                print(f"Error loading weights for {self.enc_name} from {resolved_weights_path}: {e}")
                traceback.print_exc()
                raise e
        elif self.weights_path: # User provided a path but it was invalid (handled by ensure_valid_weights_path)
             print(f"Warning: Provided weights_path '{self.weights_path}' for {self.enc_name} was invalid. Using initial weights.")
        else:
            # No weights_path provided by user, and not found in registry.
            # This implies either using timm's pretrained (if ConvNeXtV2 factory supports it)
            # or using randomly initialized weights.
            # If your _model_factories[model_variant] call already handles timm's pretrained, this is fine.
            # Otherwise, it's randomly initialized.
            print(f"No local or registry weights path found for {self.enc_name}. Model will use its initial/default weights.")


        # Standard ImageNet transforms for ConvNeXt typically
        mean, std = get_constants('imagenet') # Or specific constants if your model was trained differently
        from torchvision.transforms import InterpolationMode # Ensure import
        eval_transforms = get_eval_transforms(
            mean=mean,
            std=std,
            target_img_size=img_size, # ConvNeXtV2 is often 224
            center_crop=True, # Common practice
            interpolation=InterpolationMode.BICUBIC, # Common for ViTs/ConvNeXts
            antialias=True
        )

        precision = torch.float32 # Default, can be changed to float16 if model supports it well

        return model, eval_transforms, precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts features using the ConvNeXtV2 model's forward_features method.
        """
        if self.model is None:
             raise RuntimeError(f"Model not built for {self.enc_name}. Call _build first.")
        # Assuming your ConvNeXtV2 model has a `forward_features` method
        # that returns the desired patch embeddings (N, C)
        return self.model.forward_features(x) # type: ignore

class CustomInferenceEncoder(BasePatchEncoder):
    def __init__(self, enc_name: str, model: torch.nn.Module, transforms: Callable, precision: torch.dtype): # Added type hints
        """
        Initialize a CustomInferenceEncoder from user-defined components.
        ... (docstring unchanged) ...
        """
        # super().__init__() # Call super AFTER setting attributes needed by _build if _build isn't overridden properly
        self.enc_name = enc_name # Set enc_name before super().__init__ if _get_weights_path is called by it indirectly
        self._custom_model = model
        self._custom_transforms = transforms
        self._custom_precision = precision
        super().__init__(weights_path=None) # weights_path is not used here as model is pre-loaded

    def _build(self, **build_kwargs: Any) -> Tuple[torch.nn.Module, Callable, torch.dtype]: # Added build_kwargs, type hints
        # For CustomInferenceEncoder, model, transforms, and precision are provided at init.
        # The BasePatchEncoder._build will call this, so we return the pre-set values.
        return self._custom_model, self._custom_transforms, self._custom_precision


class MuskInferenceEncoder(BasePatchEncoder):
    
    def __init__(self, **build_kwargs):
        """
        MUSK initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, inference_aug=False, with_proj=False, out_norm=False, return_global=True):
        """
        Args:
            inference_aug (bool): Whether to use test-time multiscale augmentation. Default is False to allow for fair comparison with other models.
        """
        import timm
        
        self.enc_name = 'musk'
        self.inference_aug = inference_aug
        self.with_proj = with_proj
        self.out_norm = out_norm
        self.return_global = return_global
    
        try:
            from musk import utils, modeling
        except:
            traceback.print_exc()
            raise Exception("Please install MUSK `pip install fairscale git+https://github.com/lilab-stanford/MUSK`")

        weights_path = self._get_weights_path()

        if weights_path:
            raise NotImplementedError("MUSK doesn't support local model loading. PR welcome!")
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("musk_large_patch16_384")
                utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
            except:
                traceback.print_exc()
                raise Exception("Failed to download MUSK model, make sure that you were granted access and that you correctly registered your token")
        
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        from torchvision.transforms import InterpolationMode
        eval_transform = get_eval_transforms(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, target_img_size = 384, center_crop = True, interpolation=InterpolationMode.BICUBIC, antialias=True)
        precision = torch.float16
        
        return model, eval_transform, precision
    
    def forward(self, x):
        return self.model(
                image=x,
                with_head=self.with_proj,
                out_norm=self.out_norm,
                ms_aug=self.inference_aug,
                return_global=self.return_global  
                )[0]  # Forward pass yields (vision_cls, text_cls). We only need vision_cls.


class Conchv1InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        CONCH initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, with_proj=False, normalize=False):
        self.enc_name = 'conch_v1'
        self.with_proj = with_proj
        self.normalize = normalize

        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except:
            traceback.print_exc()
            raise Exception("Please install CONCH `pip install git+https://github.com/Mahmoodlab/CONCH.git`")
        
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                model, eval_transform = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=weights_path)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create CONCH v1 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/MahmoodLab/CONCH."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model, eval_transform = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path="hf_hub:MahmoodLab/conch")
            except:
                traceback.print_exc()
                raise Exception("Failed to download CONCH v1 model, make sure that you were granted access and that you correctly registered your token")
    
        precision = torch.float32
        
        return model, eval_transform, precision
    
    def forward(self, x):
        return self.model.encode_image(x, proj_contrast=self.with_proj, normalize=self.normalize)
    

class CTransPathInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        CTransPath initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        from torchvision.transforms import InterpolationMode
        from torch import nn

        try:
            from .model_zoo.ctranspath.ctran import ctranspath
        except:
            traceback.print_exc()
            raise Exception("Failed to import CTransPath model, make sure timm_ctp is installed. `pip install timm_ctp`")
        
        self.enc_name = 'ctranspath'
        weights_path = self._get_weights_path()

        model = ctranspath(img_size=224)
        model.head = nn.Identity()

        if not weights_path:
            self.ensure_has_internet(self.enc_name)
            try:
                from huggingface_hub import hf_hub_download   
                weights_path = hf_hub_download(
                    repo_id="MahmoodLab/hest-bench",
                    repo_type="dataset",
                    filename="CHIEF_CTransPath.pth",
                    subfolder="fm_v1/ctranspath",
                )
            except:
                traceback.print_exc()
                raise Exception("Failed to download CTransPath model, make sure that you were granted access and that you correctly registered your token")

        try:
            state_dict = torch.load(weights_path, weights_only=True)['model']
        except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create CTransPath model from local checkpoint at '{weights_path}'. "
                    "You can download the required `CHIEF_CTransPath.pth` from: https://huggingface.co/datasets/MahmoodLab/hest-bench/tree/main/fm_v1/ctranspath."
                )
        state_dict = {key: val for key, val in state_dict.items() if 'attn_mask' not in key}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert len(unexpected) == 0, f"Unexpected keys found in state dict: {unexpected}"
        assert missing == ['layers.0.blocks.1.attn_mask', 'layers.1.blocks.1.attn_mask', 'layers.2.blocks.1.attn_mask', 'layers.2.blocks.3.attn_mask', 'layers.2.blocks.5.attn_mask'], f"Unexpected missing keys: {missing}"

        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)

        precision = torch.float32
        
        return model, eval_transform, precision


class PhikonInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        Phikon initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        from transformers import ViTModel
        from torchvision.transforms import InterpolationMode

        self.enc_name = 'phikon'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                model_dir = os.path.dirname(weights_path)
                model = ViTModel.from_pretrained(model_dir, add_pooling_layer=False, local_files_only=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Phikon model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/owkin/phikon."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Phikon model, make sure that you were granted access and that you correctly registered your token")

        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)
        precision = torch.float32
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.forward_features(x)
        out = out.last_hidden_state[:, 0, :]
        return out
    
    def forward_features(self, x):
        out = self.model(pixel_values=x)
        return out
    

class HibouLInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        Hibou initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        from transformers import AutoModel
        from torchvision.transforms import InterpolationMode

        self.enc_name = 'hibou_l'
        weights_path = self._get_weights_path()

        if weights_path:
            raise NotImplementedError("Hibou-Large doesn't support local model loading. PR welcome!")
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Hibou-L model, make sure that you were granted access and that you correctly registered your token")
        
        mean, std = get_constants('hibou')
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True)
        precision = torch.float32

        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.forward_features(x)
        out = out.pooler_output
        return out
    
    def forward_features(self, x):
        out = self.model(pixel_values=x)
        return out


class KaikoInferenceEncoder(BasePatchEncoder):
    MODEL_NAME = None  # set in subclasses
    HF_HUB_ID = None # set in subclasses
    IMG_SIZE = None

    def __init__(self, **build_kwargs):
        """
        Kaiko initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        import timm
        from torchvision.transforms import InterpolationMode
        self.enc_name = f"kaiko-{self.MODEL_NAME}"
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                model = timm.create_model(
                    f"{self.HF_HUB_ID}",
                    num_classes=0,
                    checkpoint_path=weights_path,
                    img_size=self.IMG_SIZE,
                    dynamic_img_size=True
                )
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Kaiko model from local checkpoint at '{weights_path}'. "
                    "You can download the required `model.safetensors` and `config.yaml` from: https://huggingface.co/collections/1aurent/kaikoai-models-66636c99d8e1e34bc6dcf795."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model(
                    model_name=f"hf-hub:1aurent/{self.HF_HUB_ID}.kaiko_ai_towards_large_pathology_fms",
                    dynamic_img_size=True,
                    pretrained=True,
                    num_classes=0,
                    img_size=self.IMG_SIZE,
                )
            except:
                traceback.print_exc()
                raise Exception("Failed to download Kaiko model.")

        mean, std = get_constants("kaiko")
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, center_crop=True, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)
        precision = torch.float32

        return model, eval_transform, precision

    def forward(self, x):
        return self.model(x)


class KaikoS16InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vits16"
    HF_HUB_ID = "vit_small_patch16_224"
    IMG_SIZE = 224

    def __init__(self, **build_kwargs):
        """
        Kaiko Small 16 initialization.
        """
        super().__init__(**build_kwargs)
    

class KaikoS8InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vits8"
    HF_HUB_ID = "vit_small_patch8_224"
    IMG_SIZE = 224

    def __init__(self, **build_kwargs):
        """
        Kaiko Small 8 initialization.
        """
        super().__init__(**build_kwargs)
    

class KaikoB16InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitb16"
    HF_HUB_ID = "vit_base_patch16_224"
    IMG_SIZE = 224

    def __init__(self, **build_kwargs):
        """
        Kaiko Base 16 initialization.
        """
        super().__init__(**build_kwargs)
    

class KaikoB8InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitb8"
    HF_HUB_ID = "vit_base_patch8_224"
    IMG_SIZE = 224

    def __init__(self, **build_kwargs):
        """
        Kaiko Base 8 initialization.
        """
        super().__init__(**build_kwargs)
    

class KaikoL14InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitl14"
    HF_HUB_ID = "vit_large_patch14_reg4_dinov2"
    IMG_SIZE = 518

    def __init__(self, **build_kwargs):
        """
        Kaiko Large 14 initialization.
        """
        super().__init__(**build_kwargs)
    

class ResNet50InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        ResNet50-ImageNet initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self, 
        pretrained=True, 
        timm_kwargs={"features_only": True, "out_indices": [3], "num_classes": 0},
        img_size=224,
        pool=True
    ):
        import timm
        from torchvision.transforms import InterpolationMode

        self.enc_name = 'resnet50'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                model = timm.create_model("resnet50", pretrained=False, **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create ResNet50 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/timm/resnet50.tv_in1k."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("resnet50.tv_in1k", pretrained=pretrained, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download ResNet50 model.")

        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std, target_img_size=img_size, center_crop=True, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)

        precision = torch.float32
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
        
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.forward_features(x)
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out
    
    def forward_features(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        return out


class LunitS8InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        Lunit initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        import timm
        from timm.data import resolve_model_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = 'lunit-vits8'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {"img_size": 224}
                model = timm.create_model("vit_small_patch8_224", checkpoint_path=weights_path, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Lunit-Small model from local checkpoint at '{weights_path}'. "
                    "You can download the required `model.safetensors` and `config.yaml` from: https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:1aurent/vit_small_patch8_224.lunit_dino", pretrained=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Lunit S8 model, make sure that you were granted access and that you correctly registered your token.")

        data_config = resolve_model_data_config(model)
        eval_transform = create_transform(**data_config, is_training=False)
        precision = torch.float32

        return model, eval_transform, precision
    

class UNIInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        UNI initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self, 
        timm_kwargs={"dynamic_img_size": True, "num_classes": 0, "init_values": 1e-5}
    ):
        import timm
        from torchvision import transforms

        self.enc_name = 'uni_v1'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {
                    'img_size': 224,
                    'patch_size': 16,
                    'init_values': 1e-5,
                    'num_classes': 0,
                    'dynamic_img_size': True,
                }
                model = timm.create_model("vit_large_patch16_224", **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create UNI model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/MahmoodLab/UNI."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download UNI model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        precision = torch.float16
        return model, eval_transform, precision
    

class UNIv2InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        UNIv2 initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        import timm
        from torchvision import transforms

        self.enc_name = 'uni_v2'
        weights_path = self._get_weights_path()

        timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }

        if weights_path:
            try:
                model = timm.create_model(model_name='vit_giant_patch14_224', pretrained=False, **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create UNI2-h model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/MahmoodLab/UNI2-h."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download UNI v2 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        precision = torch.bfloat16
        return model, eval_transform, precision
    

class GigaPathInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        GigaPath initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self, 
    ):
        import timm
        assert timm.__version__ == '0.9.16', f"Gigapath requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = 'gigapath'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {
                    "img_size": 224,
                    "in_chans": 3,
                    "patch_size": 16,
                    "embed_dim": 1536,
                    "depth": 40,
                    "num_heads": 24,
                    "mlp_ratio": 5.33334,
                    "num_classes": 0
                }
                model = timm.create_model("vit_giant_patch14_dinov2", pretrained=False, **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create GigaPath model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/prov-gigapath/prov-gigapath."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download GigaPath model, make sure that you were granted access and that you correctly registered your token")

        mean, std = get_constants('imagenet')
        eval_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        precision = torch.float32
        return model, eval_transform, precision

    
class VirchowInferenceEncoder(BasePatchEncoder):
    import timm
    
    def __init__(self, **build_kwargs):
        """
        Virchow initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        import torchvision
        from torchvision import transforms

        self.enc_name = 'virchow'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {
                    "img_size": 224,
                    "init_values": 1e-5,
                    "num_classes": 0,
                    "mlp_ratio": 5.3375,
                    "global_pool": "",
                    "dynamic_img_size": True,
                    'mlp_layer': timm.layers.SwiGLUPacked,
                    'act_layer': torch.nn.SiLU,
                }
                model = timm.create_model("vit_huge_patch14_224", **timm_kwargs)
                model.load_state_dict(state_dict=torch.load(weights_path, map_location="cpu"), strict=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Virchow model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/paige-ai/Virchow."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Virchow model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        precision = torch.float16
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        output = self.model(x)
        class_token = output[:, 0]

        if self.return_cls:
            return class_token
        else:
            patch_tokens = output[:, 1:]
            embeddings = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            return embeddings


class Virchow2InferenceEncoder(BasePatchEncoder):
    import timm
    
    def __init__(self, **build_kwargs):
        """
        Virchow 2 initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        import torchvision
        from torchvision import transforms

        self.enc_name = 'virchow2'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {
                    "img_size": 224,
                    "init_values": 1e-5,
                    "num_classes": 0,
                    "reg_tokens": 4,
                    "mlp_ratio": 5.3375,
                    "global_pool": "",
                    "dynamic_img_size": True,
                    'mlp_layer': timm.layers.SwiGLUPacked,
                    'act_layer': torch.nn.SiLU,
                }
                model = timm.create_model("vit_huge_patch14_224", **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Virchow2 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/paige-ai/Virchow2."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Virchow-2 model, make sure that you were granted access and that you correctly registered your token")
        
        eval_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        precision = torch.float16
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        output = self.model(x)
    
        class_token = output[:, 0]
        if self.return_cls:
            return class_token
        
        patch_tokens = output[:, 5:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return embedding


class HOptimus0InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        H-Optimus0 initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self,
        timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False}
    ):
        import timm
        assert timm.__version__ == '0.9.16', f"H-Optimus requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = 'hoptimus0'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {
                    "num_classes": 0,
                    "img_size": 224,
                    "global_pool": "token",
                    'init_values': 1e-5,
                    'dynamic_img_size': False
                }
                model = timm.create_model("vit_giant_patch14_reg4_dinov2", **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create H-Optimus-0 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/bioptimus/H-optimus-0."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download HOptimus-0 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose([
            transforms.Resize(224),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
        
        precision = torch.float16
        return model, eval_transform, precision


class HOptimus1InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        H-Optimus1 initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self,
        timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False},
        **kwargs
    ):
        import timm
        assert timm.__version__ == '0.9.16', f"H-Optimus requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = 'hoptimus1'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {
                    "num_classes": 0,
                    "img_size": 224,
                    "global_pool": "token",
                    'init_values': 1e-5,
                    'dynamic_img_size': False
                }
                model = timm.create_model("vit_giant_patch14_reg4_dinov2", **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create H-Optimus-1 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/bioptimus/H-optimus-1."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download HOptimus-1 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose([
            transforms.Resize(224),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
        
        precision = torch.float16
        return model, eval_transform, precision


class Phikonv2InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        Phikonv2 initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        from transformers import AutoModel
        import torchvision.transforms as T
        from .utils.constants import IMAGENET_MEAN, IMAGENET_STD

        self.enc_name = 'phikon_v2'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                model_dir = os.path.dirname(weights_path)
                model = AutoModel.from_pretrained(model_dir)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Phikonv2 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `model.safetensors` and `config.json` from: https://huggingface.co/owkin/phikon-v2."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = AutoModel.from_pretrained("owkin/phikon-v2")
            except:
                traceback.print_exc()
                raise Exception("Failed to download Phikon v2 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = T.Compose([
            T.Resize(224),  
            T.CenterCrop(224),  
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # Normalize with specified mean and std
        ])

        precision = torch.float32
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.model(x)
        out = out.last_hidden_state[:, 0, :]
        return out


class Conchv15InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        CONCHv1.5 initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, img_size=448):
        from trident.patch_encoder_models.model_zoo.conchv1_5.conchv1_5 import create_model_from_pretrained

        self.enc_name = 'conch_v15'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                model, eval_transform = create_model_from_pretrained(checkpoint_path=weights_path, img_size=img_size)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create CONCH v1.5 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model_vision.bin` and `config.json` from: https://huggingface.co/MahmoodLab/conchv1_5."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model, eval_transform = create_model_from_pretrained(checkpoint_path="hf_hub:MahmoodLab/conchv1_5", img_size=img_size)
            except:
                traceback.print_exc()
                raise Exception("Failed to download CONCH v1.5 model, make sure that you were granted access and that you correctly registered your token")

        precision = torch.float16
        return model, eval_transform, precision


class Midnight12kInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        Midnight 12-k initialization by Kaiko.
        """
        super().__init__(**build_kwargs)

    def _build(self, return_type: Literal["cls_token", "cls+mean"] = "cls_token"):
        from transformers import AutoModel
        from .utils.constants import KAIKO_MEAN, KAIKO_STD
        from torchvision import transforms

        self.enc_name = "midnight12k"
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                model_dir = os.path.dirname(weights_path)
                model = AutoModel.from_pretrained(model_dir)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Midnight-12k model from local checkpoint at '{weights_path}'. "
                    "You can download the required `model.safetensors` and `config.json` from: https://huggingface.co/kaiko-ai/midnight."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = AutoModel.from_pretrained("kaiko-ai/midnight")
            except:
                traceback.print_exc()
                raise Exception("Failed to download Midnight-12k model")

        eval_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=KAIKO_MEAN, std=KAIKO_STD),
            ]
        )

        precision = torch.float32
        self.return_type = return_type
        return model, eval_transform, precision

    def forward(self, x):
        out = self.model(x).last_hidden_state
        cls_token = out[:, 0, :]
        if self.return_type == "cls_token":
            return cls_token
        elif self.return_type == "cls+mean":
            patch_embeddings = out[:, 1:, :]
            return torch.cat([cls_token, patch_embeddings.mean(1)], dim=-1)
        else:
            raise ValueError(
                f"expected return_type to be one of 'cls_token' or 'cls+mean', but got '{self.return_type}'"
            )
