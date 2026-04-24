"""U-Net score network for score-based diffusion models on MNIST."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for noise level conditioning."""

    def __init__(self, dim):
        """Initialise the embedding layer.

        Args:
            dim: Output embedding dimension.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Encode scalar noise levels into sinusoidal embeddings.

        Args:
            x: Noise levels of shape (B,).

        Returns:
            A tensor of sinusoidal embeddings of shape (B, dim).
        """
        half = self.dim // 2
        denom = max(half - 1, 1)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=x.device).float() / denom
        )
        args = x[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with noise-level conditioning via additive time embedding."""

    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1, num_groups=8):
        """Build a residual block conditioned on time embeddings.

        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
            time_dim: Dimension of the time embedding.
            dropout: Dropout probability for the second convolution path.
            num_groups: Maximum number of groups for GroupNorm.
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        """Apply the residual block.

        Args:
            x: Input feature map.
            t_emb: Time embedding tensor.

        Returns:
            Updated feature map after residual addition.
        """
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    """Downsample feature maps by a factor of two."""

    def __init__(self, ch):
        """Initialise the downsampling convolution.

        Args:
            ch: Number of input and output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        """Apply downsampling convolution."""
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample feature maps by a factor of two."""

    def __init__(self, ch):
        """Initialise the upsampling convolution.

        Args:
            ch: Number of input and output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        """Upsample features then apply convolution."""
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ScoreNet(nn.Module):
    """U-Net noise predictor for the score-based diffusion baseline and experts."""

    def __init__(
        self,
        in_channels=1,
        base_ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        time_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.n_levels = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        skip_channels = [base_ch]
        ch = base_ch
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
                skip_channels.append(ch)
            if i < len(ch_mult) - 1:
                self.down_samples.append(Downsample(ch))
                skip_channels.append(ch)

        self.mid1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i in range(len(ch_mult) - 1, -1, -1):
            out_ch = base_ch * ch_mult[i]
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout))
                ch = out_ch
            if i > 0:
                self.up_samples.append(Upsample(ch))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        )

        assert len(skip_channels) == 0, "Skip connection mismatch in U-Net"

    def forward(self, x, sigma):
        """Predict noise from noised images and noise levels.

        Args:
            x: Noised input images.
            sigma: Noise levels for each sample.

        Returns:
            Predicted noise tensor with same shape as `x`.
        """
        t_emb = self.time_embed(sigma.log() / 4)

        h = self.conv_in(x)
        skips = [h]

        blk, ds = 0, 0
        for i in range(self.n_levels):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[blk](h, t_emb)
                blk += 1
                skips.append(h)
            if i < self.n_levels - 1:
                h = self.down_samples[ds](h)
                ds += 1
                skips.append(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        blk, us = 0, 0
        for i in range(self.n_levels - 1, -1, -1):
            for _ in range(self.num_res_blocks + 1):
                h = torch.cat([h, skips.pop()], dim=1)
                h = self.up_blocks[blk](h, t_emb)
                blk += 1
            if i > 0:
                h = self.up_samples[us](h)
                us += 1

        return self.conv_out(h)


class ScoringHead(nn.Module):
    """Per-expert competence scorer used by Resilient MCL."""

    def __init__(self, in_channels=1, time_dim=64):
        """Initialise the per-expert scoring head.

        Args:
            in_channels: Number of input image channels.
            time_dim: Dimension of time embedding features.
        """
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(32),
            nn.Linear(32, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 + time_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, sigma):
        """Estimate per-sample expert competence score.

        Args:
            x: Noised input images.
            sigma: Noise levels for each sample.

        Returns:
            A score vector of shape (B,).
        """
        t_emb = self.time_embed(sigma.log() / 4)
        img_feat = self.cnn(x)
        return self.head(torch.cat([img_feat, t_emb], dim=1)).squeeze(-1)


class GatingNet(nn.Module):
    """Small CNN that predicts K expert logits from (x_t, sigma)."""

    def __init__(self, K, in_channels=1, img_size=28, time_dim=128):
        """Initialise the gating network.

        Args:
            K: Number of experts to route across.
            in_channels: Number of input image channels.
            img_size: Unused image size argument kept for compatibility.
            time_dim: Dimension of time embedding features.
        """
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(64),
            nn.Linear(64, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + time_dim, 128),
            nn.SiLU(),
            nn.Linear(128, K),
        )

    def forward(self, x, sigma):
        """Predict routing logits for each expert.

        Args:
            x: Noised input images.
            sigma: Noise levels for each sample.

        Returns:
            A tensor of shape (B, K) with expert logits.
        """
        t_emb = self.time_embed(sigma.log() / 4)
        img_feat = self.cnn(x)
        return self.head(torch.cat([img_feat, t_emb], dim=1))
