import torch
import torch.nn as nn
import torch.nn.functional as F

class ESDMaskWrapper1D(nn.Module):
    """
    Encoder-Separation-Decoder + Mask wrapper.
    
    Converts a standard U-Net/Separator that normally outputs waveforms 
    into a mask-estimating separator that acts on an encoded feature space.
    
    x:   [B, 2, L]
    out: [B, 2*num_sources, L]
    """
    def __init__(
        self,
        separator: nn.Module,
        num_sources: int = 2,
        enc_channels: int = 256,
        kernel_size: int = 16,
        stride: int = 8,
        mask_act: str = "sigmoid",
    ):
        super().__init__()
        self.separator = separator
        self.num_sources = num_sources
        self.enc_channels = enc_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.mask_act = mask_act

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=enc_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
            ),
            nn.PReLU(),
        )

        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_channels,
            out_channels=2,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def _fix_length(self, y: torch.Tensor, target_len: int) -> torch.Tensor:
        if y.size(-1) > target_len:
            return y[..., :target_len]
        if y.size(-1) < target_len:
            return F.pad(y, (0, target_len - y.size(-1)))
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 2, L]
        """
        B, _, L = x.shape

        # Encode: [B, C, T]
        z = self.encoder(x)
        B, C, T = z.shape

        # Separator outputs mask logits
        mask_logits = self.separator(z)

        # If the separator is a UNet with deep supervision, it might return a list
        if isinstance(mask_logits, list):
            mask_logits = mask_logits[-1]

        # Interpolate if sequence length doesn't strictly match
        if mask_logits.size(-1) != T:
            mask_logits = F.interpolate(
                mask_logits,
                size=T,
                mode="linear",
                align_corners=False,
            )

        # Reshape to [B, num_sources, C, T]
        mask_logits = mask_logits.view(B, self.num_sources, C, T)

        # Apply activation
        if self.mask_act == "softmax":
            masks = torch.softmax(mask_logits, dim=1)
        elif self.mask_act == "relu":
            masks = F.relu(mask_logits)
        else:
            masks = torch.sigmoid(mask_logits)

        # Mask and Decode
        outs = []
        for i in range(self.num_sources):
            # Element-wise multiplication over channels
            zi = z * masks[:, i]
            yi = self.decoder(zi)
            yi = self._fix_length(yi, L)
            outs.append(yi)

        # Concat the sources: [B, 2*num_sources, L]
        return torch.cat(outs, dim=1)
