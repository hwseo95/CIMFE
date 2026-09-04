__all__ = ['SCRSSL']

from typing import Optional
import torch
from torch import nn
from torch import Tensor

from .layers.pos_encoding import positional_encoding
from .layers.basics import Transpose, get_activation_fn
from .layers.attention import MultiheadAttention


class SCRSSL(nn.Module):
    """
    Self-supervised patch encoder, pretrained via masked reconstruction
    (+ an optional auxiliary PSI-prediction task).

    Output: (embedding, reconstruction, psi_prediction), each
      z:        [bs x num_patch x d_model]
      out:      [bs x num_patch x n_vars x patch_len]
      pred_psi: [bs x num_patch*patch_len]
    """
    def __init__(self, c_in: int, patch_len: int, num_patch: int,
                 n_layers: int = 3, d_model=128, n_heads=16, shared_embedding=True, d_ff: int = 256,
                 norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0., act: str = "relu",
                 res_attention: bool = False, pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, head_dropout=0,
                 d_hidden=16, verbose: bool = False, **kwargs):

        super().__init__()

        # Backbone
        self.backbone = PatchEncoder(c_in, num_patch=num_patch, patch_len=patch_len,
                                      n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                                      shared_embedding=shared_embedding, d_ff=d_ff,
                                      attn_dropout=attn_dropout, dropout=dropout, act=act,
                                      res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      d_hidden=d_hidden, norm=norm,
                                      pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Heads
        self.n_vars = c_in
        self.patch_len = patch_len
        self.reconst_head = PretrainHead(self.n_vars, d_model, patch_len, head_dropout)
        self.psi_head = PSIPredictionHead(d_model, patch_len, head_dropout)

    def forward(self, z, num_patch=None):
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """
        z = self.backbone(z, num_patch)  # z: [bs x num_patch x d_model]
        out = self.reconst_head(z)
        pred_psi = self.psi_head(z)
        return z, out, pred_psi

    @property
    def device(self):
        return next(self.parameters()).device


class PSIPredictionHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        x = self.linear(self.dropout(x))
        return x.flatten(start_dim=1)


class PretrainHead(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.n_vars = n_vars
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len * n_vars)

    def forward(self, x):
        """
        x: tensor [bs x num_patch x d_model]
        output: tensor [bs x num_patch x nvars x patch_len]
        """
        x = self.linear(self.dropout(x))
        x = x.reshape(x.shape[0], x.shape[1], self.n_vars, -1)
        return x


class PatchEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len,
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 d_hidden=16,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars):
                self.W_P.append(nn.Linear(self.n_vars * patch_len, d_model))
        else:
            self.W_P = nn.Linear(self.n_vars * patch_len, d_model)

        # Positional encoding
        self.W_pos = positional_encoding('sincos', False, 12 * 60, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                   store_attn=store_attn)

    def forward(self, x, num_patch=None) -> Tensor:
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        bs, max_patch, n_vars, patch_len = x.shape
        if num_patch is not None:
            mask = torch.arange(max_patch).unsqueeze(0).expand(bs, max_patch) >= num_patch.unsqueeze(1)
            mask = mask.to(x.device)
        else:
            mask = None
        x = torch.reshape(x, (bs, max_patch, n_vars * patch_len))         # x: [bs x num_patch x nvars*patch_len]
        u = self.W_P(x)                                                   # u: [bs x num_patch x d_model]
        u = self.dropout(u + self.W_pos[:max_patch].unsqueeze(0))         # u: [bs x num_patch x d_model]
        z = self.encoder(u, mask)                                         # z: [bs x num_patch x d_model]

        return z


class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                       attn_dropout=attn_dropout, dropout=dropout,
                                                       activation=activation, res_attention=res_attention,
                                                       pre_norm=pre_norm, store_attn=store_attn) for _ in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores, key_padding_mask=mask)
            return output
        else:
            for mod in self.layers:
                output = mod(output, key_padding_mask=mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True,
                 activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                 get_activation_fn(activation),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask)
        if self.store_attn:
            self.attn = attn
        src = src + self.dropout_attn(src2)  # residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src
