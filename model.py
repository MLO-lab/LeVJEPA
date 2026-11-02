import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import NamedTuple
from einops import rearrange

class FrameWindows(NamedTuple):
    # attention layout of Encoder.assemble: `frames` blocks of `per_frame` tokens, then one readout token
    frames: int
    per_frame: int
    window: int

def windowed_attention(q, k, v, layout):
    # Same result as the build_mask mask, without materialising it: frame t attends to the contiguous key range of
    # frames [t - window + 1, t], and the readout (last token, never a key) to the same keys as the last frame. One
    # unmasked SDPA per frame on views of q/k/v runs the fast flash kernel, which with large frames beats one masked
    # call over the whole sequence (eval: 16 x 196 tokens, ~39% of the pairs allowed).
    T, n, window = layout
    out = []
    for t in range(T):
        keys = slice(max(0, t - window + 1) * n, (t + 1) * n)
        queries = slice(t * n, (t + 1) * n + (t == T - 1))
        out.append(F.scaled_dot_product_attention(q[:, :, queries], k[:, :, keys], v[:, :, keys]))
    return torch.cat(out, dim=2)

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_channel=3, out_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channel * patch_size * patch_size, out_dim)

    def patchify(self, video):
        gh = video.shape[-2] // self.patch_size
        gw = video.shape[-1] // self.patch_size

        return rearrange(video, "b t c (gh p1) (gw p2) -> b t (gh gw) (c p1 p2)", p1=self.patch_size, p2=self.patch_size), (gh, gw)

    def forward(self, video, keep_per_frame=None, iid=False):
        x, (gh, gw) = self.patchify(video)
        B, T, N, _ = x.shape

        if keep_per_frame is not None and iid:
            # the same T * keep_per_frame tokens in total, drawn uniformly from the whole clip, so the count per frame
            # varies; sorted to keep the tokens in frame order. Returns (B, S, D) and indices into the flattened T * N
            idx = torch.rand(B, T * N, device=x.device).topk(T * keep_per_frame, dim=-1).indices.sort(dim=-1).values
            x = torch.take_along_dim(x.flatten(1, 2), idx[..., None], dim=1)
            return self.proj(x), idx, (gh, gw)

        if keep_per_frame is None:
            idx = torch.arange(N, device=x.device).expand(B, T, N)
        else:
            idx = torch.rand(B, T, N, device=x.device).topk(keep_per_frame, dim=-1).indices

        x = torch.take_along_dim(x, idx[..., None], dim=2)
        return self.proj(x), idx, (gh, gw)

def _rope(x, cos, sin):
    # rotates the pairs (2i, 2i+1); the per-axis sections have even sizes, so no pair straddles two axes
    y = x.unflatten(-1, (-1, 2))
    y = torch.stack((-y[..., 1], y[..., 0]), dim=-1).flatten(-2)
    return x * cos.to(x.dtype) + y * sin.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)

        # rotary sections for the (time, row, column) axes: half of the head dim for time, a quarter for each spatial axis
        self.rope_dims = [head_dim // 4 * 2, head_dim // 4, head_dim // 4]
        assert head_dim % 8 == 0

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, rope, mask):
        q, k, v = rearrange(self.qkv(x), "batch seq (three heads dim) -> three batch heads seq dim", three=3, heads=self.num_heads)

        # QK-norm (https://arxiv.org/abs/2010.04245, https://arxiv.org/abs/2302.05442)
        q = _rope(self.q_norm(q), *rope)
        k = _rope(self.k_norm(k), *rope)

        if isinstance(mask, FrameWindows):
            out = windowed_attention(q, k, v, mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(rearrange(out, "batch head seq dim -> batch seq (head dim)"))

    def rope_tables(self, pos):
        # cos/sin (B, 1, S, head_dim) in float32 for the (time, row, column) axes; the same for every layer, so the
        # encoder builds them once per forward
        cos, sin = [], []
        for axis, d in enumerate(self.rope_dims):
            omega = 1.0 / 10000 ** (torch.arange(d // 2, device=pos.device, dtype=torch.float32) / (d // 2))
            freq = pos[..., axis].to(torch.float32)[:, None, :, None] * omega
            cos.append(freq.cos().repeat_interleave(2, dim=-1))
            sin.append(freq.sin().repeat_interleave(2, dim=-1))
        return torch.cat(cos, dim=-1), torch.cat(sin, dim=-1)

class SwiGLU(nn.Module):
    # https://arxiv.org/pdf/2002.05202
    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or int(8 / 3 * dim)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, input_dim=384, num_heads=8):
        super().__init__()
        self.norm_attn = nn.RMSNorm(input_dim)
        self.norm_mlp = nn.RMSNorm(input_dim)

        self.attn = Attention(input_dim, num_heads=num_heads, head_dim=input_dim // num_heads)
        self.mlp = SwiGLU(dim=input_dim)

    def forward(self, x, rope, mask):
        x = x + self.attn(self.norm_attn(x), rope=rope, mask=mask)
        return x + self.mlp(self.norm_mlp(x))

class Encoder(nn.Module):
    def __init__(self, patch_size=16, residual_dim=768, num_heads=8, depth=12, attention_window=16):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size, out_dim=residual_dim)
        self.readout = nn.Parameter(torch.randn(1, residual_dim) * 0.02)

        self.attention_window = attention_window
        self.windowed_min_tokens = 64   # tokens per frame from which windowed_attention replaces the dense mask

        self.blocks = nn.ModuleList(Block(residual_dim, num_heads) for _ in range(depth))
        self.norm = nn.RMSNorm(residual_dim)

    def assemble(self, x, time, idx, grid, T):
        # x (B, S, D) in frame order; time (1 or B, S) frame and idx (B, S) spatial index of every token
        B, S, D = x.shape
        gh, gw = grid
        device = x.device

        # (time, row, column) of every patch token
        patch_pos = torch.stack([time.expand(B, S), idx // gw, idx % gw], dim=-1)

        # a single readout token at the last frame, placed in a virtual row beneath the patch tokens
        readout_pos = torch.tensor([[T - 1, gh, 0]], device=device)

        tokens = torch.cat([x, self.readout.expand(B, 1, D)], dim=1)
        pos = torch.cat([patch_pos, readout_pos.expand(B, -1, -1)], dim=1)
        frame_id = torch.cat([time, torch.full_like(time[:, :1], T - 1)], dim=1)
        is_readout = torch.cat([torch.zeros(S, dtype=torch.bool, device=device),
                                torch.ones(1, dtype=torch.bool, device=device)])

        return tokens, pos, frame_id, is_readout

    def build_mask(self, frame_id, is_readout, window):
        # frame_id (1 or B, S): shared by the batch with per-frame dropping, per sample with iid dropping
        ti, tj = frame_id[:, :, None], frame_id[:, None, :]
        mask = (tj <= ti) & (tj > ti - window) & ~is_readout
        return mask[:, None]

    def forward(self, video, keep_per_frame=None, window=None, iid=False):
        # keep_per_frame: tokens kept per frame (None keeps all); iid=True instead keeps T * keep_per_frame tokens
        # drawn uniformly over the whole clip, and patches / idx come back flat as (B, S, D) / (B, S)
        x, idx, grid = self.patch_embed(video, keep_per_frame, iid)
        T = video.shape[1]
        flat = x.dim() == 3
        if flat:
            N = grid[0] * grid[1]
            time, spatial = idx // N, idx % N
        else:
            k = x.shape[2]
            time = torch.arange(T, device=x.device).repeat_interleave(k)[None]
            x, spatial = rearrange(x, "b t k d -> b (t k) d"), rearrange(idx, "b t k -> b (t k)")
        S = x.shape[1]
        tokens, pos, frame_id, is_readout = self.assemble(x, time, spatial, grid, T)

        window = window or self.attention_window
        if not flat and k >= self.windowed_min_tokens:
            mask = FrameWindows(T, k, window)
        else:  # few tokens per frame (training views) or uneven frames (iid): one masked call over the sequence
            mask = self.build_mask(frame_id, is_readout, window)

        rope = self.blocks[0].attn.rope_tables(pos)
        for block in self.blocks:
            tokens = block(tokens, rope, mask)
        tokens = self.norm(tokens)

        patches = tokens[:, :S] if flat else rearrange(tokens[:, :S], "b (t k) d -> b t k d", t=T)
        return tokens[:, -1], patches, idx

class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, output_dim=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        shape = x.shape
        return self.proj(x.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)
