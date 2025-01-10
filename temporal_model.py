import math
from typing import List
import torch
import random
import numpy as np
import torch.nn as nn
from torch import einsum
import tinycudann as tcnn
from einops import rearrange, repeat
from einops_exts import rearrange_many

def exists(val):
    return val is not None

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class HashPositionalEncoding(nn.Module):
    def __init__(self, d_model, n, max_len=100):
        super(HashPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, n)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term).unsqueeze(2).repeat(1, 1, n)
        pe[:, 1::2] = torch.cos(position * div_term).unsqueeze(2).repeat(1, 1, n)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, vision_attn_masks=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        if len(x.shape) == 3:
            mode = "3d"
        elif len(x.shape) == 4:
            mode = "4d"
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2) # TODO: Change the shape of vision attention mask according to this.
        if vision_attn_masks is not None:
            if mode == "3d":
                vision_attn_masks = torch.cat((vision_attn_masks, 
                                                torch.ones((latents.shape[0], latents.shape[-2]), dtype=latents.dtype, device=latents.device)),
                                                dim=-1)
            elif mode == "4d":
                vision_attn_masks = torch.cat((vision_attn_masks, 
                                                torch.ones((latents.shape[0], latents.shape[1], latents.shape[-2]), dtype=latents.dtype, device=latents.device)),
                                                dim=-1)
        if mode == "3d":
            k, v = self.to_kv(kv_input).chunk(2, dim=-1)
            q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        elif mode == "4d":
            k, v = self.to_kv(kv_input).chunk(2, dim=-1)
            q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b t h n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        # Apply vision attention mask here.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
        if vision_attn_masks is not None:
            if mode == "3d":
                attn_bias = torch.zeros((q.size(0), 1, q.size(-2), k.size(-2)), dtype=q.dtype, device=q.device)
                vision_attn_masks = repeat(vision_attn_masks, 'b n -> b 1 l n', l=q.size(-2))
            elif mode == "4d":
                attn_bias = torch.zeros((q.size(0), q.size(1), 1, q.size(-2), k.size(-2)), dtype=q.dtype, device=q.device)
                vision_attn_masks = repeat(vision_attn_masks, 'b t n -> b t 1 l n', l=q.size(-2))
            attn_bias.masked_fill_(vision_attn_masks.logical_not(), float("-inf"))
            sim += attn_bias

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        if mode == "3d":
            out = rearrange(out, "b h n d -> b n (h d)", h=h)
        elif mode == "4d":
            out = rearrange(out, "b t h n d -> b t n (h d)", h=h)
        return self.to_out(out) 

class VisionTokenizer(nn.Module):
    def __init__(self, dim_media, num_tokens_per_media):
        super().__init__()
        self.dim_media = dim_media
        self.num_tokens_per_media = num_tokens_per_media

class PerceiverResampler(VisionTokenizer):
    def __init__(
        self,
        *,
        dim,
        dim_inner=None,
        depth=6,
        dim_head=96,
        heads=16,
        num_latents=8,
        ff_mult=4,
    ):
        """
        Perceiver module which takes in image features and outputs image tokens.
        Args:
            dim (int): dimension of the incoming image features
            dim_inner (int, optional): final dimension to project the incoming image features to;
                also the final dimension of the outputted features. If None, no projection is used, and dim_inner = dim.
            depth (int, optional): number of layers. Defaults to 6.
            dim_head (int, optional): dimension of each head. Defaults to 64.
            heads (int, optional): number of heads. Defaults to 8.
            num_latents (int, optional): number of latent tokens to use in the Perceiver;
                also corresponds to number of tokens per sequence to output. Defaults to 64.
            ff_mult (int, optional): dimension multiplier for the feedforward network. Defaults to 4.
        """
        if dim_inner is not None:
            projection = nn.Linear(dim, dim_inner)
        else:
            projection = None
            dim_inner = dim
        super().__init__(dim_media=dim, num_tokens_per_media=num_latents)
        self.projection = projection
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # positional embeddings
        self.video_pos_encoder = PositionalEncoding(dim, max_len=4000)
        self.clip_pos_encoder = PositionalEncoding(dim, max_len=60)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim, dim_head=dim_head, heads=heads
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def pad_and_generate_attention_mask(self, x):
        max_len = max(t.shape[0] for t in x)
        if not isinstance(x, list):
            raise ValueError("x should be a list of tensors")
        vision_attn_masks = [torch.ones((max_len), dtype=torch.bool, device=x[0].device) for _ in range(len(x))]
        for i in range(len(x)):
            vision_attn_masks[i][:x[i].shape[0]] = 0
            x[i] = nn.functional.pad(x[i], (0, 0, 0, max_len - x[i].shape[0]))
            
        x = torch.stack(x, dim=0)
        vision_attn_masks = torch.stack(vision_attn_masks, dim=0)
        return x, vision_attn_masks

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, v, D)
            vision_attn_masks (torch.Tensor): attention masks for padded visiont tokens (i.e., x)
                shape (b, v)
        Returns:
            shape (b, n, D) where n is self.num_latents
        """
        b, v = x.shape[:2]
        # positional embeddings
        x = self.clip_pos_encoder(x)
        y, vision_attn_masks = self.pad_and_generate_attention_mask(y)
        y = self.video_pos_encoder(y)
        # blocks
        latents = self.latents
        latents = repeat(latents, "n d -> b n d", b=b)
        latents = self.clip_pos_encoder(latents)
        for attn, ff in self.layers:
            latents = attn(y, latents, vision_attn_masks) + latents
            latents = ff(latents) + latents
            latents = attn(x, latents, None) + latents
            latents = ff(latents) + latents
        
        if exists(self.projection):
            return self.projection(self.norm(latents)) 
        else:
            return self.norm(latents)
    
class PerceiverResamplerGlobal(VisionTokenizer):
    def __init__(
        self,
        *,
        dim,
        dim_inner=None,
        depth=6,
        dim_head=96,
        heads=16,
        num_latents=8,
        ff_mult=4,
    ):
        """
        Perceiver module which takes in image features and outputs image tokens.
        Args:
            dim (int): dimension of the incoming image features
            dim_inner (int, optional): final dimension to project the incoming image features to;
                also the final dimension of the outputted features. If None, no projection is used, and dim_inner = dim.
            depth (int, optional): number of layers. Defaults to 6.
            dim_head (int, optional): dimension of each head. Defaults to 64.
            heads (int, optional): number of heads. Defaults to 8.
            num_latents (int, optional): number of latent tokens to use in the Perceiver;
                also corresponds to number of tokens per sequence to output. Defaults to 64.
            ff_mult (int, optional): dimension multiplier for the feedforward network. Defaults to 4.
        """
        if dim_inner is not None:
            projection = nn.Linear(dim, dim_inner)
        else:
            projection = None
            dim_inner = dim
        super().__init__(dim_media=dim, num_tokens_per_media=num_latents)
        self.projection = projection
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # positional embeddings
        self.video_pos_encoder = PositionalEncoding(dim, max_len=4000)
        self.clip_pos_encoder = PositionalEncoding(dim, max_len=60)

        self.global_layers = [1, 3]
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim, dim_head=dim_head, heads=heads
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def pad_and_generate_attention_mask(self, x, clip_x):
        max_len = max(t.shape[0] for t in x) + clip_x.shape[1]
        if not isinstance(x, list):
            raise ValueError("x should be a list of tensors")
        
        vision_attn_masks = [torch.ones((max_len), dtype=torch.bool, device=x[0].device) for _ in range(len(x))]
        for i in range(len(x)):
            vision_attn_masks[i][:x[i].shape[0]] = 0
            vision_attn_masks[i][-clip_x.shape[1]:] = 0
            x[i] = nn.functional.pad(x[i], (0, 0, 0, max_len - x[i].shape[0] - clip_x.shape[1]))
            
        x = torch.stack(x, dim=0)
        x = torch.cat((x, clip_x), dim=-2)
        vision_attn_masks = torch.stack(vision_attn_masks, dim=0)
        return x, vision_attn_masks

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, v, D)
            vision_attn_masks (torch.Tensor): attention masks for padded visiont tokens (i.e., x)
                shape (b, v)
        Returns:
            shape (b, n, D) where n is self.num_latents
        """
        b, v = x.shape[:2]
        # positional embeddings
        x = self.clip_pos_encoder(x)
        y, vision_attn_masks = self.pad_and_generate_attention_mask(y, x)
        y = self.video_pos_encoder(y)
        # blocks
        latents = self.latents
        latents = repeat(latents, "n d -> b n d", b=b)
        latents = self.clip_pos_encoder(latents)
        for idx, (attn, ff) in enumerate(self.layers):
            if idx in self.global_layers:
                latents = attn(y, latents, vision_attn_masks) + latents
                latents = ff(latents) + latents
            else:
                latents = attn(x, latents, None) + latents
                latents = ff(latents) + latents
        
        if exists(self.projection):
            return self.projection(self.norm(latents)) 
        else:
            return self.norm(latents)
        
class PerceiverResamplerGlobalPosition(VisionTokenizer):
    def __init__(
        self,
        *,
        dim,
        dim_inner=None,
        pre_depth=2,
        global_depth=2,
        depth=6,
        dim_head=96,
        heads=16,
        num_latents=8,
        ff_mult=4,
        use_global=True,
        use_position=True,
        use_homogenization=True,
    ):
        """
        Perceiver module which takes in image features and outputs image tokens.
        Args:
            dim (int): dimension of the incoming image features
            dim_inner (int, optional): final dimension to project the incoming image features to;
                also the final dimension of the outputted features. If None, no projection is used, and dim_inner = dim.
            depth (int, optional): number of layers. Defaults to 6.
            dim_head (int, optional): dimension of each head. Defaults to 64.
            heads (int, optional): number of heads. Defaults to 8.
            num_latents (int, optional): number of latent tokens to use in the Perceiver;
                also corresponds to number of tokens per sequence to output. Defaults to 64.
            ff_mult (int, optional): dimension multiplier for the feedforward network. Defaults to 4.
        """
        if dim_inner is not None:
            projection = nn.Linear(dim, dim_inner)
        else:
            projection = None
            dim_inner = dim
        super().__init__(dim_media=dim, num_tokens_per_media=num_latents)

        self.use_global = use_global
        self.use_position = use_position

        # ablation use
        self.use_homogenization = use_homogenization
        self.homo_window = 3
        
        self.projection = projection
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.max_size = 32
        self.hash_dim = 24

        # positional embeddings
        self.clip_pos_encoder = PositionalEncoding(dim, max_len=60)

        if use_global:
            self._init_global_model(dim, global_depth, dim_head, heads, ff_mult)
            
        if use_position:
            # self.hash_pos_encoder = HashPositionalEncoding(self.max_size, self.hash_dim, max_len=60)
            self.clip_projection = nn.Linear(dim, self.hash_dim)
            self.clip_reprojection = nn.Linear(self.hash_dim, dim)
            concat_dim = self._init_hash_model(dim, pre_depth, dim_head, heads, ff_mult)
            if concat_dim != dim:
                self.concat_projection = nn.Linear(concat_dim, dim)
            else:
                self.concat_projection = None

        self.layers = nn.ModuleList([])
        if self.use_position:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                            PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                            FeedForward(dim=dim, mult=ff_mult),
                        ]
                    )
                )
        else:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                            PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                            FeedForward(dim=dim, mult=ff_mult),
                        ]
                    )
                )
        self.norm = nn.LayerNorm(dim)

    def _init_global_model(self, dim, global_depth, dim_head, heads, ff_mult):
        self.global_layers = nn.ModuleList([])
        for _ in range(global_depth):
            self.global_layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        self.video_pos_encoder = PositionalEncoding(dim, max_len=4000)
        return None

    def _init_hash_model(self, dim, pre_depth, dim_head, heads, ff_mult):
        
        concat_dim = dim
        # hash model
        hash_encoder = tcnn.Encoding(
                n_input_dims=2,
                encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447,
            },
        )
        # hash_encoder.output_dim = n_levels * n_features_per_level
        mlp_head = tcnn.Network(
                        n_input_dims=hash_encoder.n_output_dims,
                        n_output_dims=self.hash_dim,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "None",
                            "n_neurons": 64,
                            "n_hidden_layers": 2,
                        },
                    )
        
        self.hash_model = torch.nn.Sequential(hash_encoder, mlp_head)
        self.hash_projection = nn.Linear(mlp_head.n_output_dims * self.max_size, dim)
        self.hash_layers = nn.ModuleList([])
        for _ in range(pre_depth):
            self.hash_layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=self.hash_dim, dim_head=dim_head, heads=heads),
                        PerceiverAttention(dim=self.hash_dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=self.hash_dim, mult=ff_mult),
                    ]
                )
            )
        return concat_dim

    def pad_and_generate_attention_mask(self, x, clip_x):
        max_len = max(t.shape[0] for t in x) + clip_x.shape[1]
        if not isinstance(x, list):
            raise ValueError("x should be a list of tensors")
        
        vision_attn_masks = [torch.ones((max_len), dtype=torch.bool, device=x[0].device) for _ in range(len(x))]
        for i in range(len(x)):
            vision_attn_masks[i][:x[i].shape[0]] = 0
            vision_attn_masks[i][-clip_x.shape[1]:] = 0
            x[i] = nn.functional.pad(x[i], (0, 0, 0, max_len - x[i].shape[0] - clip_x.shape[1]))
            
        x = torch.stack(x, dim=0)
        x = torch.cat((x, clip_x), dim=-2)
        vision_attn_masks = torch.stack(vision_attn_masks, dim=0)
        return x, vision_attn_masks
    
    def hash_encode_and_mask(self, position):
        hash_positions = []
        hash_masks = []
        for i in range(len(position)):
            # i: [b]
            batch_hash_mask = []
            one_hash_positions = []
            for j in range(len(position[i])):
                # j: [t]
                one_hash_mask = torch.ones(self.max_size, dtype=torch.bool, device='cuda')
                # one_hash_mask = [1 for _ in range(self.max_size)]
                if len(position[i][j]) < self.max_size:
                    one_hash_mask[:len(position[i][j])] = 0
                    position_temp = position[i][j] + [[-1, -1] for _ in range(self.max_size - len(position[i][j]))]
                elif len(position[i][j]) > self.max_size:
                    position_temp = random.sample(position[i][j], self.max_size)
                else:
                    position_temp = position[i][j]
                one_hash_positions.append(position_temp)
                batch_hash_mask.append(one_hash_mask)
                
            batch_hash_mask = torch.stack(batch_hash_mask, dim=0)
            hash_masks.append(batch_hash_mask)
            hash_positions.append(one_hash_positions)

        # hash_masks = torch.tensor(hash_masks, dtype=torch.bool, device='cuda')
        hash_masks = torch.stack(hash_masks, dim=0)
        input_data = torch.tensor(hash_positions).cuda()
        b, t, n, d = input_data.shape
        input_data = rearrange(input_data, 'b t n d -> (b t n) d')
        hash_positions = self.hash_model(input_data)
        # hash_positions = rearrange(hash_positions, 'b t n d -> b t (n d)')
        hash_positions = rearrange(hash_positions, '(b t n) d -> b t n d', b=b, t=t, n=n)
        return hash_positions, hash_masks
    
    def homogenize_video_features(self, x):
        b, t, d = x.shape
        padding = (0, 0, 0, self.homo_window - t % self.homo_window)
        x = torch.nn.functional.pad(x, padding, "constant", 0)
        x = x.unfold(1, self.homo_window, self.homo_window).select(3, 0).repeat(1, 1, self.homo_window).view(b, -1, d)[:, :t, :]
        return x

    def forward(self, x, y, position):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, v, D)
            y (list of torch.Tensor): video features
                shape (b, t, D)
            position (list): 2d position features
                shape (b, t, n, 2)
        Returns:
            shape (b, n, D) where n is self.num_latents
        """
        b, _, d = x.shape
        # TODO: add filter
        # 分两个阶段

        # positional embeddings
        if self.use_position:
            hash_positions, hash_masks = self.hash_encode_and_mask(position)
            hash_positions = hash_positions.to(torch.float32)
            # hash_positions = self.hash_projection(hash_positions)
            # hash_positions = self.hash_pos_encoder(hash_positions)
            # default: hash_positions [16, 30, 32, 24]

        if self.use_global:
            y, vision_attn_masks = self.pad_and_generate_attention_mask(y, x)
            y = self.video_pos_encoder(y)

        x = self.clip_pos_encoder(x)
        # blocks
        latents = self.latents
        latents = repeat(latents, "n d -> b n d", b=b)
        latents = self.clip_pos_encoder(latents)

        if self.use_position:
            clip_features = self.clip_projection(x).unsqueeze(2)
            for idx, (pos_sa, pos_attn, pos_ff) in enumerate(self.hash_layers):
                clip_features = pos_sa(clip_features, clip_features, None) + clip_features
                clip_features = pos_attn(hash_positions, clip_features, hash_masks) + clip_features
                clip_features = pos_ff(clip_features) + clip_features
            clip_features = x + self.clip_reprojection(clip_features.squeeze())

        if self.use_global:
            if self.use_homogenization:
                y = self.homogenize_video_features(y)
            for idx, (global_sa, global_attn, global_ff) in enumerate(self.global_layers):
                latents = global_sa(latents, latents, None) + latents
                latents = global_attn(y, latents, vision_attn_masks) + latents
                latents = global_ff(latents) + latents

        if self.use_homogenization:
            x = self.homogenize_video_features(x)
        for idx, (sa, attn, ff) in enumerate(self.layers):
            # latents = sa(latents, latents, None) + latents
            if self.use_position:
                latents = attn(clip_features, latents, None) + latents
            else:
                latents = attn(x, latents, None) + latents
            latents = ff(latents) + latents
    
        if exists(self.projection):
            return self.projection(self.norm(latents)) 
        else:
            return self.norm(latents)
        
if __name__ == '__main__':
    model = PerceiverResamplerGlobal(dim=256)
    # shape (batch, time, sequence length, dimension)
    vision_features = torch.randn(4, 30, 256)
    video_features = [torch.randn(1000, 256)] * 2 + [torch.randn(500, 256)] * 2
    vision_attn_masks = None
    # vision_features = vision_features[:, None, :, :] # Expand dimensions.
    y = model(vision_features, video_features)
    # shape (batch, time, num_latents, dimension)
    print(y.shape)