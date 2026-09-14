"""Explicit validation-calibrated v5 inference; original weights stay intact."""

import json
import math
from pathlib import Path

import torch

from full_model_v5 import StagedResidualFullClassifier
import train_full_model_v5 as runner


class ScaledResidualFullClassifier(StagedResidualFullClassifier):
    def __init__(self, *, residual_scale, **kwargs):
        if not math.isfinite(residual_scale) or not 0 <= residual_scale <= 1:
            raise ValueError("An explicit finite residual scale in [0,1] is required")
        super().__init__(**kwargs)
        # Not a learned parameter. Persist this setting in the separate manifest,
        # never silently assume it is contained in an original v5 checkpoint.
        self.residual_scale = float(residual_scale)

    def forward(self, img_global, txt_global, flat_patches, patch_mask,
                token_embeds, attention_mask, language_ids=None,
                return_aux=True, return_attention=False):
        _, aux = super().forward(img_global,txt_global,flat_patches,patch_mask,
                                 token_embeds,attention_mask,language_ids,
                                 return_aux=True,return_attention=return_attention)
        delta = aux["correction_logits"].float() * self.residual_scale
        logits = self.add_correction(aux["base_logits"],delta)
        if not return_aux:
            return logits
        aux.update(unscaled_correction_logits=aux["correction_logits"],
                   correction_logits=delta,residual_scale=self.residual_scale)
        return logits,aux


def load_scaled_checkpoint(manifest_path, device="cpu"):
    """Load one explicitly versioned scale manifest plus its immutable weights."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["format"] != "fgclip2_v5_fixed_scale_v1" or manifest["variant"] not in ("full_v5","pooled_v5"):
        raise ValueError("Unsupported scale manifest")
    source = Path(manifest["checkpoint_path"])
    if runner.cached.file_sha256(source) != manifest["checkpoint_sha256"]:
        raise ValueError("Source checkpoint hash mismatch")
    saved = torch.load(source,map_location="cpu",weights_only=False)
    config = saved["config"]
    if config["variant"] != manifest["variant"] or config["seed"] != manifest["seed"] or saved["fingerprint"] != runner.shared.digest(config):
        raise ValueError("Manifest does not match checkpoint identity")
    protocol = config["protocol"]
    if manifest["cache_metadata_sha256"] != protocol["cache_metadata_sha256"]:
        raise ValueError("Different source feature-cache identity")
    for name,value in protocol["core_settings"].items():
        if getattr(runner.core,name) != value:
            raise ValueError(f"Runtime setting differs from training: {name}")
    for name,expected in protocol["source_sha256"].items():
        if runner.cached.file_sha256(runner.ROOT/name) != expected:
            raise ValueError(f"Source implementation changed: {name}")
    model = ScaledResidualFullClassifier(residual_scale=manifest["residual_scale"],
        use_attention=manifest["variant"]=="full_v5",cross_dropout=0.,**manifest["feature_dims"],
        proj_dim=runner.core.PROJ_DIM,hidden_dim=runner.core.HIDDEN_DIM,
        num_classes=runner.core.NUM_CLASSES,num_heads=runner.core.NUM_HEADS,dropout=runner.core.DROPOUT)
    model.load_state_dict(saved["clf_state_dict"],strict=True)
    return model.to(device).eval(),manifest
