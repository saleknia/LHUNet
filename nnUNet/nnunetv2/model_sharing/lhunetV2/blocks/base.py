from typing import Optional, Sequence, Tuple, Union, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from timm.models.layers import trunc_normal_

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm


__all__ = ["BaseBlock", "get_conv_layer", "get_padding", "get_output_padding", "DownsampleWithSpaceToDepth"]


class BaseBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError(
            "padding value should not be negative, please change the kernel size and/or stride."
        )
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError(
            "out_padding value should not be negative, please change the kernel size and/or stride."
        )
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]

class SpaceToDepth3D(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        self.block_size = block_size
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        bs = self.block_size
        
        # print(f"Input shape: {x.shape}")
        
        # Pad to divisible size
        D_pad = D - (D % bs)
        H_pad = H - (H % bs)
        W_pad = W - (W % bs)
        
        if D_pad != D or H_pad != H or W_pad != W:
            x = x[:, :, :D_pad, :H_pad, :W_pad]
            # print(f"After padding: {x.shape}")
        
        D_new = D_pad // bs
        H_new = H_pad // bs
        W_new = W_pad // bs
        
        # print(f"New spatial dims: {D_new}, {H_new}, {W_new}")
        # print(f"Expected output channels: {C * (bs ** 3)}")
        
        # Reshape
        x = x.reshape(B, C, D_new, bs, H_new, bs, W_new, bs)
        # print(f"After reshape: {x.shape}")
        
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        # print(f"After permute: {x.shape}")
        
        x = x.reshape(B, C * (bs ** 3), D_new, H_new, W_new)
        # print(f"Output shape: {x.shape}")
        
        return x


class DownsampleWithSpaceToDepth(nn.Module):
    """
    3D spatial downsampling that preserves ALL information
    Assumes in_channels is divisible by 8
    """
    def __init__(self, in_channels, out_channels=None, block_size=2):
        super().__init__()
        
        # out_channels is ignored since channels remain the same
        # In 3D, block_size=2 gives 2×2×2 = 8 voxels
        compressed_channels = in_channels // (block_size ** 3)
        
        # Compress channels
        self.compress = nn.Conv3d(in_channels, compressed_channels, kernel_size=1)
        
        # Space-to-depth rearranges 2×2×2 blocks into channels
        self.space_to_depth = SpaceToDepth3D(block_size=block_size)
        
    def forward(self, x):
        # Step 1: Compress to in_channels // 8
        x = self.compress(x)
        
        # Step 2: Space-to-depth (channels × 8 = back to in_channels, spatial ÷ 2)
        x = self.space_to_depth(x)
        
        return x