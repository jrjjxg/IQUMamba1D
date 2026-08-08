import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class RMSNormChannel1D(nn.Module):
    """
    RMSNorm over channel dimension for [B, C, L].
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight.view(1, -1, 1)


class SafeBiMambaEncoderBlock1D(nn.Module):
    """
    Safe bidirectional Mamba block for encoder stages.

    Input/output:
        x: [B, C, L]

    Important design:
        - pre-norm
        - bidirectional scan
        - delta residual
        - learnable residual scale initialized near 0
        - no channel replacement
    """
    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        residual_scale_init: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if Mamba is None:
            raise ImportError("mamba_ssm is required for SafeBiMambaEncoderBlock1D.")

        self.norm = RMSNormChannel1D(channels)

        self.fwd = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.bwd = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, x):
        """
        x: [B, C, L]
        """
        x_norm = self.norm(x)
        u = x_norm.transpose(1, 2)  # [B, L, C]

        y_f = self.fwd(u)

        u_rev = torch.flip(u, dims=[1])
        y_b = self.bwd(u_rev)
        y_b = torch.flip(y_b, dims=[1])

        # delta residual, not feature replacement
        delta = 0.5 * ((y_f - u) + (y_b - u))  # [B, L, C]
        delta = delta.transpose(1, 2)          # [B, C, L]
        delta = self.dropout(delta)

        return x + self.res_scale * delta


class EncoderWithBiMamba1D(nn.Module):
    """
    Wrap an existing ResidualConvEncoder and apply BiMamba after selected stages.
    """
    def __init__(
        self,
        encoder: nn.Module,
        use_mamba_stages,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        residual_scale_init: float = 0.0,
        block_type: str = "safe", # "safe" or "original"
    ):
        super().__init__()
        self.encoder = encoder

        assert len(use_mamba_stages) == len(encoder.output_channels), \
            "use_mamba_stages must match encoder stages."

        self.mamba_blocks = nn.ModuleList()
        for use, ch in zip(use_mamba_stages, encoder.output_channels):
            if use:
                if block_type == "original":
                    from models.IQUBiMamba1D import BiMambaLayer
                    self.mamba_blocks.append(
                        BiMambaLayer(
                            dim=ch,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                            channel_token=False,
                        )
                    )
                elif block_type == "unidirectional":
                    from models.IQUMamba1D import MambaLayer
                    m_block = MambaLayer(
                        dim=ch,
                        channel_token=False,
                    )
                    self.mamba_blocks.append(m_block)
                else:
                    self.mamba_blocks.append(
                        SafeBiMambaEncoderBlock1D(
                            channels=ch,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                            residual_scale_init=residual_scale_init,
                        )
                    )
            else:
                self.mamba_blocks.append(nn.Identity())

        # expose attributes used by decoder
        self.output_channels = encoder.output_channels
        self.strides = encoder.strides
        self.conv_op = encoder.conv_op
        self.norm_op = encoder.norm_op
        self.norm_op_kwargs = encoder.norm_op_kwargs
        self.kernel_sizes = encoder.kernel_sizes
        self.conv_pad_sizes = encoder.conv_pad_sizes
        self.nonlin = encoder.nonlin
        self.nonlin_kwargs = encoder.nonlin_kwargs

    def forward(self, x):
        skips = self.encoder(x)
        skips = [
            block(feat) for block, feat in zip(self.mamba_blocks, skips)
        ]
        return skips
