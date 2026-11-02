import os

import torch
import torch.nn as nn
from einops import rearrange

from legacy import vision_transformer

# LeVJEPA v1 checkpoints: the paper's ViT (legacy/vision_transformer.py), a [cls] token over the whole clip and
# block-causal attention, behind the interface of model.Encoder so that eval/eval.py can probe it. Both formats keep the
# encoder under `encoder.*`: Lightning checkpoints of the v1 main.py (with the EMA under `state_dict_ema`) and the
# Hugging Face release (galilai-group/LeVJEPA-1.5-Large, which holds the EMA weights).

class LegacyEncoder(nn.Module):
    def __init__(self, name="vit_large", img_size=224, patch_size=16, num_frames=16, tubelet_size=1, use_rope=True,
                 attn_mode="block_causal", **_):
        super().__init__()
        self.vit = getattr(vision_transformer, name)(img_size=img_size, patch_size=patch_size, num_frames=num_frames,
                                                     tubelet_size=tubelet_size, use_rope=use_rope, attn_mode=attn_mode)
        self.tubelet_size = tubelet_size
        # block-causal attention has no window: every frame attends to all earlier ones
        self.attention_window = num_frames

    def forward(self, clip, **_):
        """clip (B, T, C, H, W) -> (cls (B, D), patches (B, T / tubelet_size, N, D), None), like model.Encoder."""
        x = self.vit(rearrange(clip, "b t c h w -> b c t h w"))
        return x[:, 0], rearrange(x[:, 1:], "b (t n) d -> b t n d", t=clip.shape[1] // self.tubelet_size), None

def legacy_state_dict(checkpoint, ema=True):
    """Encoder weights of a v1 checkpoint: a Lightning .ckpt (the EMA copy if `ema` and the checkpoint has one), a
    .safetensors file, or a Hugging Face repo id."""
    if os.path.isfile(checkpoint) and not checkpoint.endswith(".safetensors"):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["state_dict_ema"] if ema and "state_dict_ema" in ckpt else ckpt["state_dict"]
    else:
        from safetensors.torch import load_file
        if not os.path.isfile(checkpoint):
            from huggingface_hub import hf_hub_download
            checkpoint = hf_hub_download(checkpoint, "model.safetensors")
        state = load_file(checkpoint)
    return {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}

def load_legacy_encoder(checkpoint, ema=True, **model):
    """A v1 encoder with its weights loaded, e.g. load_legacy_encoder("galilai-group/LeVJEPA-1.5-Large")."""
    encoder = LegacyEncoder(**model)
    state = legacy_state_dict(checkpoint, ema)
    if encoder.vit.pos_embed is None:  # with RoPE there is no positional embedding to load
        state.pop("pos_embed", None)
    encoder.vit.load_state_dict(state)
    return encoder.eval()
