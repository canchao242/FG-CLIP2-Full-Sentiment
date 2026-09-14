"""Residual cross-attention head; legacy models and checkpoints stay unchanged."""

import torch
from torch import nn

import train_fgclip2_emotion_enhanced_v8_3_ablation_suite_summary_plus as core


class NormalizedCrossAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.patch_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())

    def forward(self, tokens, patches, weights, patch_mask, return_attention=False):
        if not patch_mask.bool().any(dim=-1).all():
            raise ValueError("Every image must contain at least one valid patch.")
        query = self.query_norm(tokens)
        key_value = self.patch_norm(patches)
        attended, attention = self.attention(
            query, key_value, key_value,
            key_padding_mask=~patch_mask.bool(),
            need_weights=return_attention, average_attn_weights=False,
        )
        pooled = (self.output(attended) * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, attention


class ResidualFullClassifier(core.EmotionGatedFusionClassifier):
    """The no-cross feature path plus a zero-initialized, learnable increment.

    Shared modules are constructed before the cross branch, so reseeding gives
    exactly the same shared initialization in Full-v2 and its no-cross control.
    At alpha=0 the eval-mode logits equal the control's logits. This is an
    initialization property, NOT a guarantee of validation/test performance.
    """

    def __init__(self, *, use_cross=True, cross_dropout=0.1, **kwargs):
        kwargs.pop("ablation_cfg", None)
        super().__init__(ablation_cfg=core.get_ablation_config("no_cross"), **kwargs)
        self.use_cross = use_cross
        if use_cross:
            self.cross_attention = NormalizedCrossAttention(
                self.proj_dim, kwargs.get("num_heads", 4), cross_dropout,
            )
            self.cross_alpha = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("cross_alpha", None)

    def forward(
        self, img_global, txt_global, flat_patches, patch_mask,
        token_embeds, attention_mask, language_ids=None, return_aux=True,
        return_attention=False, return_intermediates=False,
    ):
        img_g = self.img_global_proj(img_global)
        txt_g = self.txt_global_proj(txt_global)
        patches, region, patch_weights, patch_mask = self.region_attention(
            flat_patches, txt_global, patch_mask=patch_mask,
        )
        tokens, local, token_weights, _ = self.token_emotion(token_embeds, attention_mask)
        # Empty content must not acquire a spurious local feature from Linear biases.
        local = local * attention_mask.bool().any(dim=-1, keepdim=True).to(local.dtype)
        gate_logits = self.gate_net(torch.cat([img_g, txt_g, region, local], dim=-1))
        gate = torch.sigmoid(gate_logits - gate_logits.mean(dim=-1, keepdim=True))
        fused = gate * img_g + (1 - gate) * txt_g + 0.5 * (region + local)
        region_slot = (1 - gate) * region
        cross = torch.zeros_like(region)
        attention = None
        if self.use_cross:
            # The legacy encoders already multiply each token by (1 + weight).
            # Undo that modulation here; use content weights once for output pooling.
            raw_tokens = tokens / (1 + token_weights.unsqueeze(-1))
            raw_patches = patches / (1 + patch_weights.unsqueeze(-1))
            cross, attention = self.cross_attention(
                raw_tokens, raw_patches, token_weights, patch_mask, return_attention,
            )
            region_slot = region_slot + self.cross_alpha * cross
        features = torch.cat([
            img_g, txt_g, fused, region_slot,
            (img_g - txt_g).abs(), img_g * txt_g,
            (region - local).abs(), region * local,
        ], dim=-1)
        logits = self.classifier(features)
        if not return_aux:
            return logits
        aux = {
            "patch_weights": patch_weights, "token_weights": token_weights,
            "gate": gate, "gate_mean": gate.mean().detach(),
            "language_ids": language_ids, "region_vec": region,
            "text_local": local, "cross_vec": cross, "cross_attn": attention,
            "cross_alpha": self.cross_alpha,
            "ablation_cfg": {**self.ablation_cfg, "use_cross_attention": self.use_cross},
        }
        if return_intermediates:
            aux["projected_text"] = txt_g
            aux["unweighted_tokens"] = tokens / (1 + token_weights.unsqueeze(-1))
            aux["unweighted_patches"] = patches / (1 + patch_weights.unsqueeze(-1))
            aux["valid_patch_mask"] = patch_mask
        return logits, aux


def make_content_mask(encoded, prefix_lengths):
    """Exclude prompts/special/padding tokens using offsets, not guessed counts.

    A token spanning the prompt/content boundary is retained when its end offset
    reaches into the content. Fully truncated/empty content has an all-zero mask.
    """
    masks = []
    for offsets, special, valid, prefix_len in zip(
        encoded["offset_mapping"], encoded["special_tokens_mask"],
        encoded["attention_mask"], prefix_lengths,
    ):
        masks.append([
            int(bool(v) and not bool(s) and end > prefix_len and end > start)
            for (start, end), s, v in zip(offsets, special, valid)
        ])
    return masks
