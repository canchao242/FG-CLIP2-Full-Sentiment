"""Full-v4: contextual-token correction, preserving the static no-cross base."""

import torch
from torch import nn

from full_model_v2 import NormalizedCrossAttention, ResidualFullClassifier


class ContextualCorrection(nn.Module):
    def __init__(self, token_dim, dim, heads, classes, use_attention=True, dropout=0.):
        super().__init__()
        self.use_attention = use_attention
        # Construct all common modules first to match their initialization.
        self.context_projection = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, dim), nn.GELU())
        self.context_score = nn.Linear(dim, 1)
        nn.init.zeros_(self.context_score.weight)
        nn.init.zeros_(self.context_score.bias)
        self.classifier = nn.Sequential(nn.LayerNorm(4 * dim), nn.Linear(4 * dim, max(8, dim // 2)),
                                        nn.GELU(), nn.Linear(max(8, dim // 2), classes))
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)
        if use_attention:
            self.sentence_norm = nn.LayerNorm(dim)
            self.interaction = NormalizedCrossAttention(dim, heads, dropout)
        else:
            self.pool_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())

    def forward(self, context, patches, token_mask, patch_mask, sentence, return_attention=False):
        valid = patch_mask.bool()
        if not valid.any(-1).all():
            raise ValueError("Every image must contain at least one valid patch.")
        h = self.context_projection(context)
        # Compute the masked softmax in FP32, including all-empty texts safely.
        scores = self.context_score(h).squeeze(-1).float()
        weights = torch.softmax(scores.masked_fill(~token_mask.bool(), -1e4), dim=-1)
        weights = weights * token_mask.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        weights = weights.to(h.dtype)
        local = (h * weights.unsqueeze(-1)).sum(1)
        attention = None
        if self.use_attention:
            query = h + .5 * self.sentence_norm(sentence).unsqueeze(1)
            cross, attention = self.interaction(query, patches, weights, valid, return_attention)
        else:
            pooled = (patches * valid.unsqueeze(-1)).sum(1) / valid.sum(-1, keepdim=True)
            cross = self.pool_projection(pooled)
        features = torch.cat([local, cross, (local - cross).abs(), local * cross], dim=-1)
        has_content = token_mask.bool().any(-1, keepdim=True).to(h.dtype)
        delta = self.classifier(features) * has_content
        return delta, cross * has_content, weights, attention


class ContextualResidualFullClassifier(ResidualFullClassifier):
    """Input token channels concatenate static and contextual frozen features.

    Existing shared weights see ONLY the original static half. Contextual
    features are used only by the new zero-initialized logit correction branch.
    No performance guarantee follows from equality at initialization.
    """

    def __init__(self, *, use_attention=True, cross_dropout=0., **kwargs):
        super().__init__(use_cross=False, **kwargs)
        self.base_token_dim = kwargs["token_dim"]
        self.correction_uses_attention = use_attention
        with torch.random.fork_rng(devices=[]):
            self.cross_attention = ContextualCorrection(
                self.base_token_dim, self.proj_dim, kwargs.get("num_heads", 4),
                kwargs.get("num_classes", 3), use_attention, cross_dropout)

    def forward(self, img_global, txt_global, flat_patches, patch_mask, token_embeds,
                attention_mask, language_ids=None, return_aux=True, return_attention=False):
        if token_embeds.shape[-1] != 2 * self.base_token_dim:
            raise ValueError("Full-v4 requires separately verified static and contextual token channels")
        static, context = token_embeds.split(self.base_token_dim, dim=-1)
        logits, aux = super().forward(img_global, txt_global, flat_patches, patch_mask,
            static, attention_mask, language_ids, return_aux=True, return_intermediates=True)
        aux.pop("unweighted_tokens")
        delta, cross, weights, attention = self.cross_attention(context,
            aux.pop("unweighted_patches"), attention_mask, aux.pop("valid_patch_mask"),
            aux.pop("projected_text"), return_attention)
        logits = logits + delta
        if not return_aux:
            return logits
        aux.update(correction_logits=delta, cross_vec=cross, cross_attn=attention,
                   contextual_token_weights=weights)
        aux["ablation_cfg"] = {**aux["ablation_cfg"], "use_cross_attention": self.correction_uses_attention}
        return logits, aux
