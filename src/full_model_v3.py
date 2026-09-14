"""Sentence-guided cross attention with zero-initialized residual logits.

Uses the EXISTING static-token cache plus its contextual sentence vector.
This is sentence conditioning, not contextual token feature extraction.
"""

import torch
from torch import nn

from full_model_v2 import NormalizedCrossAttention, ResidualFullClassifier


class LogitCorrection(nn.Module):
    def __init__(self, dim, heads, classes, use_attention=True, dropout=0.):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.sentence_norm = nn.LayerNorm(dim)
            self.interaction = NormalizedCrossAttention(dim, heads, dropout)
        else:
            # An explicit extra-head control; no inactive attention parameters.
            # Not parameter-matched to Full: it tests whether simple extra
            # capacity/pooling suffices, not an isolated equal-parameter effect.
            self.pool_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.classifier = nn.Sequential(
            nn.LayerNorm(4 * dim), nn.Linear(4 * dim, max(8, dim // 2)),
            nn.GELU(), nn.Linear(max(8, dim // 2), classes),
        )
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, tokens, patches, token_weights, patch_mask, sentence, return_attention=False):
        valid = patch_mask.bool()
        if not valid.any(dim=-1).all():
            raise ValueError("Every image must contain at least one valid patch.")
        attention = None
        if self.use_attention:
            query = tokens + 0.5 * self.sentence_norm(sentence).unsqueeze(1)
            cross, attention = self.interaction(query, patches, token_weights, valid, return_attention)
        else:
            pooled = (patches * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
            cross = self.pool_projection(pooled)
        features = torch.cat([sentence, cross, (sentence - cross).abs(), sentence * cross], dim=-1)
        has_content = (token_weights.sum(-1, keepdim=True) > 0).to(sentence.dtype)
        delta = self.classifier(features) * has_content
        return delta, cross * has_content, attention


class LogitResidualFullClassifier(ResidualFullClassifier):
    """No-cross path is unchanged; an additive branch corrects its logits.

    Constructing the extra branch preserves the shared CPU RNG state. With
    dropout=0 in this branch, the shared path also consumes identical random
    draws at runtime. Validation selection can still prefer epoch 0: zero
    initialization is not a claim or guarantee that Full will generalize better.
    """

    def __init__(self, *, use_attention=True, cross_dropout=0., **kwargs):
        super().__init__(use_cross=False, **kwargs)
        self.correction_uses_attention = use_attention
        # All modules are constructed on CPU before the runner moves the model.
        with torch.random.fork_rng(devices=[]):
            self.cross_attention = LogitCorrection(
                self.proj_dim, kwargs.get("num_heads", 4), kwargs.get("num_classes", 3),
                use_attention, cross_dropout,
            )

    def forward(self, img_global, txt_global, flat_patches, patch_mask, token_embeds,
                attention_mask, language_ids=None, return_aux=True, return_attention=False):
        logits, aux = super().forward(
            img_global, txt_global, flat_patches, patch_mask, token_embeds,
            attention_mask, language_ids, return_aux=True, return_intermediates=True,
        )
        delta, cross, attention = self.cross_attention(
            aux.pop("unweighted_tokens"), aux.pop("unweighted_patches"), aux["token_weights"],
            aux.pop("valid_patch_mask"), aux.pop("projected_text"), return_attention,
        )
        logits = logits + delta
        if not return_aux:
            return logits
        aux.update(cross_vec=cross, cross_attn=attention, correction_logits=delta)
        aux["ablation_cfg"] = {**aux["ablation_cfg"], "use_cross_attention": self.correction_uses_attention}
        return logits, aux
