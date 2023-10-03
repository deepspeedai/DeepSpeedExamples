import torch
import torch.nn.functional as F
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from torch import nn
import os
import sys
import math
# sys.path.append('/vc_data/users/xwu/image-language/DeepSpeedExamples-internal-high-loss/applications/DeepSpeed-Chat-multi-modal/training/utils')
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
from torch.nn.init import trunc_normal_


class VisProjection_vit(nn.Module):
    def __init__(self, vis_config, lang_dim):
        super().__init__()
        # TODO: for now, hard-coded for ViT
        self.vis_layer = CLIPEncoderLayer(vis_config)
        self.projection = nn.Sequential( 
            nn.Linear(vis_config.hidden_size, lang_dim), # an example implementation
            nn.LayerNorm(lang_dim, eps=1e-12))
    def forward(self, vis_input):
        vis_feature = self.vis_layer(vis_input, None, None)[0] # only need the first output
        return self.projection(vis_feature)
    

# The following code is adopted from QWen-Clip
def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos

# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class VisProjection_perceiver(nn.Module):
    def __init__(self, vis_config, lang_dim):
        super().__init__()
        # TODO: for now, hard-coded for perceiver
        grid_size = 16
        self.num_queries = grid_size ** 2
        self.embed_dim = lang_dim
        self.num_heads = lang_dim // 128 

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(lang_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, lang_dim))
        trunc_normal_(self.query, std=.02)

        self.kv_proj = nn.Linear(vis_config.hidden_size, lang_dim) 

        self.attn = nn.MultiheadAttention(lang_dim, self.num_heads)
        self.ln_q = nn.LayerNorm(lang_dim, eps=1e-12)
        self.ln_kv = nn.LayerNorm(lang_dim, eps=1e-12)
        self.projection = nn.Sequential(
            nn.LayerNorm(lang_dim, eps=1e-12), 
            nn.Linear(lang_dim, lang_dim) # an example implementation
            )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):
        # import pdb; pdb.set_trace()
        pos_embed = get_abs_pos(self.pos_embed, x.size(1))

        x = x[:, 1:, :] # remove cls token
        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)


        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1),
            x + pos_embed.unsqueeze(1),
            x,
            attn_mask=attn_mask)[0]
        return self.projection(out.permute(1, 0, 2))

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)