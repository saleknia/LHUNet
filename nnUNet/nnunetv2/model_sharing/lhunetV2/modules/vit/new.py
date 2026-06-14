import torch
import torch.nn as nn
from ..cnn import *


# ========== SIMPLIFIED: Multi-Scale Context Aggregator (MCA) ==========
class MultiScaleContextAggregator(nn.Module):
    """
    Simplified MCA with equal channel split and no global pooling
    Uses only dilated depthwise convolutions
    """
    def __init__(self, dim, dilations=[2, 3, 4]):
        super().__init__()
        self.dilations = dilations
        num_groups = len(dilations)
        
        # Equal split for all groups
        split_sizes = [dim // num_groups] * num_groups
        split_sizes[-1] = dim - sum(split_sizes[:-1])  # Adjust last group
        self.split_sizes = split_sizes
        
        # ===== Channel Calibration Unit (CCU) from CENet =====
        self.ccu = CCU3D(dim)
        
        # ===== Depthwise dilated convs for ALL groups =====
        self.dw_convs = nn.ModuleList()
        for i, d in enumerate(dilations):
            padding = d
            self.dw_convs.append(
                nn.Conv3d(split_sizes[i], split_sizes[i], kernel_size=3,
                          padding=padding, dilation=d, groups=split_sizes[i])
            )
        
        # ===== Gate and value paths (from CENet MCA) =====
        self.gate = nn.Conv3d(dim, dim, kernel_size=1)
        self.value_proj = nn.Conv3d(dim, dim, kernel_size=1)
        
        # ===== Fusion =====
        self.fusion = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1),
            nn.SiLU(inplace=True)
        )
        
        # Activation for gating
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        shortcut = x
        
        # Step 1: Channel Calibration (CCU)
        x = self.ccu(x)
        
        # Step 2: Multi-scale feature extraction
        splits = torch.split(x, self.split_sizes, dim=1)
        outputs = []
        
        # Dilated convolutions on all groups
        for split, dw_conv in zip(splits, self.dw_convs):
            outputs.append(dw_conv(split))
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(outputs, dim=1)
        
        # Step 3: Gated fusion (from CENet MCA)
        g = self.gate(x)
        v = self.value_proj(multi_scale)
        
        # Gate and value multiply
        x = self.fusion(self.act(g) * self.act(v))
        
        # Residual connection
        return x + shortcut


# ========== CCU3D (Channel Calibration Unit) - unchanged ==========
class CCU3D(nn.Module):
    """
    Channel Calibration Unit from CENet
    Uses max, mean, and std pooling for rich channel statistics
    """
    def __init__(self, channels, hidden_scale=3):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, hidden_scale * channels, kernel_size=3, 
                             groups=channels, bias=False, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(hidden_scale * channels, channels, kernel_size=1, 
                             groups=channels, bias=False, padding=0)
        self.bn = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        
        # Global pooling across spatial dimensions
        x_max = torch.max(x.view(B, C, -1), dim=2)[0]  # (B, C)
        x_mean = torch.mean(x, dim=(2, 3, 4))          # (B, C)
        x_std = torch.std(x, dim=(2, 3, 4), unbiased=False)  # (B, C)
        
        # Stack: (B, C, 3)
        u = torch.stack([x_max, x_mean, x_std], dim=-1)
        
        # Channel calibration via 1D convolutions
        z = self.fc2(self.act(self.fc1(u))).view(B, C)
        if B > 1:
            z = self.bn(z)
        g = torch.sigmoid(z).reshape(B, C, 1, 1, 1)
        
        return x * g


# ========== MLPWithDWConv - unchanged ==========
class MLPWithDWConv(nn.Module):
    """
    FFN + Depthwise Conv for local feature refinement
    Similar to EfficientViT but optimized for 3D
    """
    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        
        # Channel projection (expand)
        self.fc1 = nn.Conv3d(dim, hidden_dim, kernel_size=1)
        
        # Depthwise conv for local refinement
        self.dwconv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, 
                                padding=1, groups=hidden_dim)
        
        # Channel projection (compress)
        self.fc2 = nn.Conv3d(hidden_dim, dim, kernel_size=1)
        
        self.act = nn.GELU()
        
    def forward(self, x):
        # Expand channels
        x = self.fc1(x)
        x = self.act(x)
        
        # Local refinement via depthwise conv
        x = self.dwconv(x)
        x = self.act(x)
        
        # Compress channels
        x = self.fc2(x)
        
        return x


# ========== UPDATED: HybAttnBlock with stage-adaptive dilations ==========
class HybAttnBlock(nn.Module):
    """
    Serial design with MCA + LKAd + MLP for semantic stages (3-5)
    Stage-adaptive dilations to ensure kernels fit within feature maps
    """
    def __init__(
        self,
        input_size: int,
        dim: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        cnn_block_code="d",      # Keep for LKAd
        vit_block_code="c",      # IGNORED - we remove ViT
        use_rb=False,
        use_r=False,
        stage_id=None,           # Stage ID for adaptive dilations
    ) -> None:
        super().__init__()
        
        # ===== Stage-adaptive dilations (safe for feature map size) =====
        if stage_id == 3:        # Processes 12×12×12 after maxpool
            dilations = [2, 3, 4, 5]
        elif stage_id == 4:      # Processes 6×6×6 after maxpool
            dilations = [1, 2]
        else:                     # Stage 5: processes 3×3×3
            dilations = [1]       
        
        print("=" * 50)
        print(f"MCALeaderBlock Stage {stage_id} (Serial: MCA + LKAd + MLP)")
        print(f"  - dim: {dim}")
        print(f"  - dilations: {dilations}")
        print(f"  - LKAd: {cnn_block_code}")
        print("=" * 50)
        
        # 1. MCA - Multi-scale Context Aggregator with stage-appropriate dilations
        self.mca = MultiScaleContextAggregator(dim, dilations=dilations)
        
        # 2. LKAd - Deformable attention for shape adaptation
        if cnn_block_code == "d":
            print("  - Using LKAd for deformable shape adaptation")
            self.lkad = LKAd3D_Block(d_model=dim)
        else:
            self.lkad = LKAd3D_Block(d_model=dim)
        
        # 3. MLP with depthwise conv for channel mixing + local refinement
        self.mlp = MLPWithDWConv(dim, expansion_ratio=4)
        
        # ===== SINGLE LayerNorm (reduced from 3) =====
        self.norm = nn.LayerNorm(dim)
        
        # ===== Final batch norm (for compatibility) =====
        self.bnorm = nn.BatchNorm3d(dim)
        
        # Optional dropout
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # For compatibility with original interface
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1), requires_grad=True)
        self.delta = nn.Parameter(torch.ones(dim, 1, 1, 1), requires_grad=True)
        self.use_r = use_r
        
        print()

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        
        # ===== SINGLE NORM at the beginning =====
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x_flat = self.norm(x_flat)
        x = x_flat.transpose(1, 2).reshape(B, C, D, H, W)
        
        # ===== Serial processing =====
        x = x + self.mca(x)      # MCA: multi-scale context
        x = x + self.lkad(x)     # LKAd: deformable shape adaptation
        x = x + self.mlp(x)      # MLP: channel mixing + local refinement
        
        # ===== Final batch norm =====
        x = self.dropout(x)
        x = self.bnorm(x)
        
        return x