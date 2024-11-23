from typing import Dict, Any, Type, List, Tuple, Callable, Union, Optional, Iterable
import torch
import torch.nn as nn
from functools import partial
from inspect import signature

__all__ = [
    "val2list",
    "val2tuple",
    "squeeze_list",
    "build_kwargs_from_config",
    "build_act",
    "build_norm",
    'get_same_padding'
]

def val2list(x: Union[List, Tuple, Any], repeat_time=1) -> List:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

def val2tuple(x: Union[List, Tuple, Any], min_len: int = 1, idx_repeat: int = -1) -> Tuple:
    x = val2list(x)
    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)

def squeeze_list(x: Optional[List]) -> Union[List, Any]:
    if x is not None and len(x) == 1:
        return x[0]
    else:
        return x

def build_kwargs_from_config(config: Dict, target_func: Callable) -> Dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs

# register activation function here
REGISTERED_ACT: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}

def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    if name in REGISTERED_ACT:
        act_cls = REGISTERED_ACT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None

# register normalization function here
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out

REGISTERED_NORM_DICT: Dict[str, Type[nn.Module]] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
}

def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None

def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2






# from functools import partial
# import torch
# import torch.nn as nn
# from inspect import signature

# __all__ = [
#     "val2list",
#     "val2tuple",
#     "squeeze_list",
#     "build_kwargs_from_config",
#     "build_act",
#     "build_norm",
#     'get_same_padding'
# ]

# def val2list(x: list or tuple or any, repeat_time=1) -> list: # type: ignore
#     if isinstance(x, (list, tuple)):
#         return list(x)
#     return [x for _ in range(repeat_time)]

# def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple: # type: ignore
#     x = val2list(x)
#     # repeat elements if necessary
#     if len(x) > 0:
#         x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

#     return tuple(x)

# def squeeze_list(x: list or None) -> list or any: # type: ignore
#     if x is not None and len(x) == 1:
#         return x[0]
#     else:
#         return x

# def build_kwargs_from_config(config: dict, target_func: callable) -> dict[str, any]:
#     valid_keys = list(signature(target_func).parameters)
#     kwargs = {}
#     for key in config:
#         if key in valid_keys:
#             kwargs[key] = config[key]
#     return kwargs

# # register activation function here
# REGISTERED_ACT: dict[str, type] = {
#     "relu": nn.ReLU,
#     "relu6": nn.ReLU6,
#     "hswish": nn.Hardswish,
#     "silu": nn.SiLU,
#     "gelu": partial(nn.GELU, approximate="tanh"),
# }
# def build_act(name: str, **kwargs) -> nn.Module or None: # type: ignore
#     if name in REGISTERED_ACT:
#         act_cls = REGISTERED_ACT[name]
#         args = build_kwargs_from_config(kwargs, act_cls)
#         return act_cls(**args)
#     else:
#         return None

# # register normalization function here
# class LayerNorm2d(nn.LayerNorm):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = x - torch.mean(x, dim=1, keepdim=True)
#         out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
#         if self.elementwise_affine:
#             out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
#         return out
# REGISTERED_NORM_DICT: dict[str, type] = {
#     "bn2d": nn.BatchNorm2d,
#     "ln": nn.LayerNorm,
#     "ln2d": LayerNorm2d,
# }
# def build_norm(name="bn2d", num_features=None, **kwargs) -> nn.Module or None: # type: ignore
#     if name in ["ln", "ln2d"]:
#         kwargs["normalized_shape"] = num_features
#     else:
#         kwargs["num_features"] = num_features
#     if name in REGISTERED_NORM_DICT:
#         norm_cls = REGISTERED_NORM_DICT[name]
#         args = build_kwargs_from_config(kwargs, norm_cls)
#         return norm_cls(**args)
#     else:
#         return None

# def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:# type: ignore
#     if isinstance(kernel_size, tuple):
#         return tuple([get_same_padding(ks) for ks in kernel_size])
#     else:
#         assert kernel_size % 2 > 0, "kernel size should be odd number"
#         return kernel_size // 2
