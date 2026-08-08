import torch
import torch.nn as nn
from models.dual_domain_bandsplit import STFTFrontend, BandSplitModule, BandSplitMambaBlock, BandMergeModule

class BandSplitSeparator(nn.Module):
    """
    Pure Band-Split Separator for IQ Signals.
    Splits the frequency spectrum into bands, applies intra-band and inter-band Mamba, 
    and predicts separated signals via direct complex mapping.
    """
    def __init__(self, 
                 n_srcs: int = 2,
                 n_fft: int = 256,
                 hop_length: int = 64,
                 win_length: int = 256,
                 center: bool = True,
                 n_bands: int = 16,
                 hidden_dim: int = 128,
                 n_mamba_layers: int = 6,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        
        self.stft = STFTFrontend(n_fft, hop_length, win_length, center)
        n_freq = n_fft // 2 + 1
        
        # STFT outputs 4 channels: real_I, imag_I, real_Q, imag_Q
        in_ch = 4
        
        self.band_split = BandSplitModule(in_ch, n_freq, n_bands, hidden_dim)
        
        self.mamba_layers = nn.ModuleList([
            BandSplitMambaBlock(hidden_dim, n_bands, d_state, d_conv, expand)
            for _ in range(n_mamba_layers)
        ])
        
        # Output channels: 4 * n_srcs
        out_ch = 4 * n_srcs
        self.band_merge = BandMergeModule(out_ch, n_freq, n_bands, hidden_dim)
        
        self.register_buffer("window", torch.hann_window(win_length, periodic=True))
        
    def forward(self, x):
        # x: [B, 2, L]
        B, _, L = x.shape
        
        # 1. STFT -> [B, 4, F, T]
        mix_spec = self.stft(x)
        B, C, F, T = mix_spec.shape
        
        # 2. Band-Split
        band_feats = self.band_split(mix_spec) # [B, T, K, H]
        for layer in self.mamba_layers:
            band_feats = layer(band_feats)
            
        # 3. Band-Merge
        sep_spec = self.band_merge(band_feats, B, T, F) # [B, 4*n_srcs, F, T]
        
        # 4. iSTFT
        outputs = []
        for s in range(self.n_srcs):
            # Extract 4 channels for this source: real_I, imag_I, real_Q, imag_Q
            src_spec = sep_spec[:, s*4 : (s+1)*4, :, :]
            
            # Form complex spectrum
            I_spec = torch.complex(src_spec[:, 0], src_spec[:, 1])
            Q_spec = torch.complex(src_spec[:, 2], src_spec[:, 3])
            
            I_time = torch.istft(I_spec, self.n_fft, self.hop_length, self.win_length, self.window, center=self.center, length=L)
            Q_time = torch.istft(Q_spec, self.n_fft, self.hop_length, self.win_length, self.window, center=self.center, length=L)
            
            # Stack into [B, 2, L]
            src_time = torch.stack([I_time, Q_time], dim=1)
            outputs.append(src_time)
            
        # Concat -> [B, 2*n_srcs, L]
        return torch.cat(outputs, dim=1)
