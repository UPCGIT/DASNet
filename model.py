from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import collections.abc
from itertools import repeat
from einops import rearrange


# ====================== IIA空间注意力模块 ======================
class AttentionWeight(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(AttentionWeight, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size, padding=padding, groups=channel, bias=False)
        self.bn = nn.BatchNorm1d(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, w, c, h = x.size()
        x_weight = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_weight = self.conv1(x_weight).view(b, c, h)
        x_weight = self.sigmoid(self.bn(self.conv2(x_weight)))
        x_weight = x_weight.view(b, 1, c, h)
        return x * x_weight


class IIA(nn.Module):
    def __init__(self, channel):
        super(IIA, self).__init__()
        self.attention = AttentionWeight(channel)

    def forward(self, x):
        x_h = x.permute(0, 3, 1, 2).contiguous()
        x_h = self.attention(x_h).permute(0, 2, 3, 1)

        x_w = x.permute(0, 2, 1, 3).contiguous()
        x_w = self.attention(x_w).permute(0, 2, 1, 3)

        x = x_h + x_w
        return x


# ====================== SOSA注意力模块 (原GAM, 使用IIA) ======================
class SOSA(nn.Module):
    def __init__(self, in_channels, rate=8):
        super(SOSA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // rate, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // rate, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = IIA(channel=in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        x_gap = self.avg_pool(x).view(b, c)
        x_att_permute = self.channel_attention(x_gap).view(b, c, 1, 1)
        x = x * x_att_permute

        x_spatial_att = self.spatial_attention(x)
        out = x_spatial_att + residual
        return out


# ====================== 空间特征提取模块 (spatial) ======================
class spatial(nn.Module):
    def __init__(self, in_channel=64, depth=32):
        super(spatial, self).__init__()
        self.conv = nn.Conv2d(in_channel, depth, 3, 1, padding=1)

        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(depth, depth, 1, 1, groups=depth),
            nn.Conv2d(depth, depth, 1, 1),
            nn.BatchNorm2d(depth), nn.ReLU()
        )
        self.atrous_block2 = nn.Sequential(
            nn.Conv2d(depth, depth, 3, 1, padding=2, dilation=2, groups=depth),
            nn.Conv2d(depth, depth, 1, 1),
            nn.BatchNorm2d(depth), nn.ReLU()
        )
        self.atrous_block3 = nn.Sequential(
            nn.Conv2d(depth, depth, 3, 1, padding=3, dilation=3, groups=depth),
            nn.Conv2d(depth, depth, 1, 1),
            nn.BatchNorm2d(depth), nn.ReLU()
        )
        self.atrous_block4 = nn.Sequential(
            nn.Conv2d(depth, depth, 3, 1, padding=4, dilation=4, groups=depth),
            nn.Conv2d(depth, depth, 1, 1),
            nn.BatchNorm2d(depth), nn.ReLU()
        )

    def forward(self, x):
        x_conv = self.conv(x)
        out = torch.cat([
            self.atrous_block1(x_conv),
            self.atrous_block2(x_conv),
            self.atrous_block3(x_conv),
            self.atrous_block4(x_conv)
        ], dim=1)
        return out


# ====================== 光谱特征提取模块 (原spectral) ======================
class spectral(nn.Module):
    """光谱特征提取模块"""

    def __init__(self, in_channels=64, hidden_channels=32, out_channels=64):
        super(spectral, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.conv1d_1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv1d_2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv1d_3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        # 修复: 添加了乘号 *
        self.conv1d_final = nn.Conv1d(3 * hidden_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_flat = x.view(batch_size, channels, -1)

        g1 = self.relu(self.bn1(self.conv1d_1(x_flat)))
        g2 = self.relu(self.bn2(self.conv1d_2(g1)))
        g3 = self.relu(self.bn3(self.conv1d_3(g2)))

        concat_features = torch.cat([g1, g2, g3], dim=1)
        x_spe = self.conv1d_final(concat_features)
        x_spe = x_spe.view(batch_size, self.out_channels, height, width)

        return x_spe


# ====================== HSSA及Transformer相关组件 ======================
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FFN(nn.Module):
    def __init__(self, dim, bias, kernel_size):
        super(FFN, self).__init__()
        if kernel_size not in [3, 5, 7]:
            raise ValueError("Invalid kernel_size. Must be 3, 5, or 7.")

        self.kernel_size = kernel_size
        hidden_features = 180

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3,
                                   groups=hidden_features, bias=bias)
        self.relu3 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        if self.kernel_size == 3:
            x = self.relu3(self.dwconv3x3(x))
        elif self.kernel_size == 5:
            x = self.relu3(self.dwconv5x5(x))
        elif self.kernel_size == 7:
            x = self.relu3(self.dwconv7x7(x))
        x = self.project_out(x)
        return x


class HSSA(nn.Module):
    def __init__(self, dim, num_heads, group_num=4, bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.group_num = group_num
        self.dim_group = dim // group_num
        self.dim_head = self.dim_group // num_heads

        self.scale = self.dim_head ** -0.5

        self.qkv = nn.Conv3d(in_channels=group_num, out_channels=group_num * 3, kernel_size=(1, 1, 1), bias=False)
        self.qkv_dw = nn.Conv3d(in_channels=group_num * 3, out_channels=group_num * 3, kernel_size=(1, 3, 3),
                                padding=(0, 1, 1), groups=group_num * 3, bias=bias)

        self.w = nn.Parameter(torch.ones(2))
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.group_num, self.dim_group, H, W)

        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b t (head c) h1 w1 -> b head c (h1 w1 t)', head=self.num_heads)
        k = rearrange(k, 'b t (head c) h1 w1 -> b head c (h1 w1 t)', head=self.num_heads)
        v = rearrange(v, 'b t (head c) h1 w1 -> b head c (h1 w1 t)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale

        attn_soft = F.softmax(attn, dim=-1)
        attn_relu = F.relu(attn) ** 2

        w = F.softmax(self.w, dim=0)
        attn = w[0] * attn_soft + w[1] * attn_relu

        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = rearrange(out, 'b head c (h1 w1 t) -> b t (head c) h1 w1', head=self.num_heads, h1=H, w1=W)
        out = out.reshape(B, -1, H, W)
        out = self.proj(out)

        return out


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    # 修复: 添加了减号 -
    keep_prob = 1.0 - drop_prob

    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x_feat = self.proj(x)
        x_seq = x_feat.flatten(2).transpose(1, 2)
        x_seq = self.norm(x_seq)
        return x_feat, x_seq


class Block(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 norm_layer=nn.LayerNorm, LayerNorm_type='WithBias', kernel_size=3, group_num=4):
        super(Block, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_selective = HSSA(
            dim=dim,
            num_heads=num_heads,
            bias=qkv_bias,
            group_num=group_num
        )
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn_conv = FFN(dim, qkv_bias, kernel_size)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x.clone()
        x_norm = self.norm1(x)
        x_attn = self.attn_selective(x_norm)
        x = shortcut + self.drop_path(x_attn)
        x_norm = self.norm2(x)
        x_ffn = self.ffn_conv(x_norm)
        x = x + self.drop_path(x_ffn)
        return x


# ====================== 双分支网络 ======================
class DualBranchDAASPP_Transformer(nn.Module):
    def __init__(self, img_size=27, patch_size=4, in_c=6, num_classes=16,
                 embed_dim=64, depth=4, num_heads=4, mlp_ratio=4.0, qkv_bias=True,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 LayerNorm_type='WithBias', kernel_size=3,
                 group_num=4,
                 use_spatial=True, use_spectral=True):
        super(DualBranchDAASPP_Transformer, self).__init__()

        self.embed_dim = embed_dim
        self.use_spatial = use_spatial
        self.use_spectral = use_spectral
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # ==================== 1. Transformer 分支的 Patch Embedding ====================
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        self.grid_size = self.patch_embed.grid_size
        num_patches = self.patch_embed.num_patches

        self.pos_embed_branch2 = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]

        # ==================== 2. CNN 分支的独立处理 ====================
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(in_c, embed_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

        if use_spatial:
            self.spatial = spatial(in_channel=embed_dim, depth=32)
            # 修复: 4 * 32
            self.spatial_conv = nn.Conv2d(4 * 32, embed_dim, 1, 1)
        else:
            self.daaspp = nn.Identity()
            self.daaspp_conv = nn.Identity()

        if use_spectral:
            self.spectral = spectral(in_channels=embed_dim, hidden_channels=32, out_channels=embed_dim)
        else:
            self.spectral = nn.Identity()

        # 修复: 原名SOSA, 且添加乘号 2 * embed_dim
        self.SOSA = SOSA(2 * embed_dim)

        # 修复: 添加乘号 2 * embed_dim
        self.branch1_fusion_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, 1)

        self.cnn_downsample = nn.AdaptiveAvgPool2d(self.grid_size)

        self.branch1_output = nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

        # ==================== 3. Transformer 分支构建 ====================
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, drop_ratio=drop_ratio,
                  attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], norm_layer=norm_layer,
                  LayerNorm_type=LayerNorm_type, kernel_size=kernel_size,
                  group_num=group_num)
            for i in range(depth)
        ])
        self.norm_branch2 = LayerNorm(embed_dim, LayerNorm_type)

        # ==================== 4. 融合与输出 ====================
        self.global_pool_branch1 = nn.AdaptiveAvgPool2d(1)
        self.global_pool_branch2 = nn.AdaptiveAvgPool2d(1)

        self.fusion_dim = embed_dim * 2
        self.head = nn.Linear(self.fusion_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed_branch2, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        B, C, H, W = x.shape

        # ==================== 分支 1: CNN ====================
        x_cnn_base = self.cnn_stem(x)

        if self.use_spatial:
            x_spatial = self.spatial(x_cnn_base)
            x_spatial = self.spatial_conv(x_spatial)
        else:
            x_spatial = x_cnn_base

        if self.use_spectral:
            # 修复: 将 self.speem 替换为 self.spectral
            x_spectral = self.spectral(x_cnn_base)
        else:
            x_spectral = x_cnn_base

        x_cnn_fusion = torch.cat([x_spatial, x_spectral], dim=1)

        # 修复: 将 self.gam 替换为 self.SOSA
        x_cnn_enhanced = self.SOSA(x_cnn_fusion)
        x_branch1 = self.branch1_fusion_conv(x_cnn_enhanced)

        x_branch1_pooled = self.global_pool_branch1(x_branch1)
        x_branch1_flat = torch.flatten(x_branch1_pooled, 1)

        # ==================== 分支 2: Transformer ====================
        x_feat, x_seq = self.patch_embed(x)
        x_branch2 = self.pos_drop(x_seq + self.pos_embed_branch2)

        grid_h, grid_w = self.grid_size
        x_branch2 = x_branch2.permute(0, 2, 1).view(B, self.embed_dim, grid_h, grid_w)
        x_branch2 = self.blocks(x_branch2)
        x_branch2 = self.norm_branch2(x_branch2)

        x_branch2_pooled = self.global_pool_branch2(x_branch2)
        x_branch2_flat = torch.flatten(x_branch2_pooled, 1)

        # ==================== 最终融合 ====================
        x_fusion = torch.cat([x_branch1_flat, x_branch2_flat], dim=1)

        return x_fusion

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# ====================== 测试代码 ======================

if __name__ == "__main__":
    # 模拟输入数据: (batch_size=64, channels=6, height=27, width=27)
    x = torch.randn(64, 1, 6, 27, 27)
    print(f"输入数据形状: {x.shape}")

    # 创建双分支模型
    model = DualBranchDAASPP_Transformer(
        img_size=27,
        patch_size=4,
        in_c=6,
        num_classes=16,
        embed_dim=64,
        depth=1,
        num_heads=4,
        use_spatial=True,
        use_spectral=True
    )

    # 前向传播
    output = model(x)
    print(f"输出数据形状: {output.shape}")

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {total_params:,}")

    print("\n测试通过！")