"""Staged contextual residual adaptation; the original No-cross is untouched."""

import torch

from full_model_v2 import ResidualFullClassifier
from full_model_v4 import ContextualResidualFullClassifier


class StagedResidualFullClassifier(ContextualResidualFullClassifier):
    """Freeze both base weights AND base dropout during correction warm-up.

    The branch has the same architecture as v4. The changes are the training
    schedule, optional direct branch supervision, and FP32 residual addition.
    No guarantee of improved validation or test performance is implied.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_frozen = False

    def set_base_frozen(self, frozen):
        self.base_frozen = bool(frozen)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("cross_attention.") or not frozen)
            if frozen and not name.startswith("cross_attention."):
                parameter.grad = None
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "base_frozen", False):
            for name, module in self.named_children():
                if name != "cross_attention":
                    module.eval()
        return self

    @staticmethod
    def add_correction(base, correction):
        # Avoid rounding a small learned correction away when the base is FP16.
        return base.float() + correction.float()

    def forward(self, img_global, txt_global, flat_patches, patch_mask,
                token_embeds, attention_mask, language_ids=None,
                return_aux=True, return_attention=False):
        if token_embeds.shape[-1] != 2 * self.base_token_dim:
            raise ValueError("Full-v5 requires separately verified static and contextual tokens")
        static, context = token_embeds.split(self.base_token_dim, dim=-1)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.base_frozen):
            base_logits, aux = ResidualFullClassifier.forward(
                self, img_global, txt_global, flat_patches, patch_mask, static,
                attention_mask, language_ids, return_aux=True, return_intermediates=True)
        aux.pop("unweighted_tokens")
        delta, cross, weights, attention = self.cross_attention(
            context, aux.pop("unweighted_patches"), attention_mask,
            aux.pop("valid_patch_mask"), aux.pop("projected_text"), return_attention)
        logits = self.add_correction(base_logits, delta)
        if not return_aux:
            return logits
        aux.update(base_logits=base_logits, correction_logits=delta, cross_vec=cross,
                   cross_attn=attention, contextual_token_weights=weights,
                   base_frozen=self.base_frozen)
        aux["ablation_cfg"] = {**aux["ablation_cfg"], "use_cross_attention": self.correction_uses_attention}
        return logits, aux
