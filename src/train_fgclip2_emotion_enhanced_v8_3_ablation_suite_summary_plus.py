import argparse
from pathlib import Path
import gc
import json
import math
import os
import random
import time
from collections import Counter
from contextlib import nullcontext

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# =========================
# 路径配置
# =========================
MODEL_ID = "qihoo360/fg-clip2-base"
_RELEASE_ROOT = Path(__file__).resolve().parent

ZH_TRAIN_CSV = str(_RELEASE_ROOT / "data_submission_v1/zh/train.csv")
ZH_VAL_CSV = str(_RELEASE_ROOT / "data_submission_v1/zh/val.csv")
ZH_TEST_CSV  = None

EN_TRAIN_CSV = str(_RELEASE_ROOT / "data_submission_v1/en/train.csv")
EN_VAL_CSV = str(_RELEASE_ROOT / "data_submission_v1/en/val.csv")
EN_TEST_CSV  = None

# v8_4.1 balanced：保留性能优化，单独保存以保护 v8_3/v8_4 历史结果。
SAVE_DIR = str(_RELEASE_ROOT / "checkpoints/legacy_ablation")


# =========================
# 训练配置
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BATCH_SIZE = 16
# Evaluation is inference-only and batch-invariant (no BatchNorm; sample-centred
# gate), so a larger batch reduces repeated-validation wall time without changing
# the training batch or checkpoint-selection rule.
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 20
HEAD_LR = 5e-5
BACKBONE_LR = 5e-7
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

MAX_NUM_PATCHES = 256
MAX_TEXT_LEN = 64
WALK_TYPE = "short"
# embedding 更接近旧 v8_3 的最佳均衡结果；contextual 保留为显式消融选项。
TOKEN_FEATURE_SOURCE = "embedding"

TOKENIZER = None

TARGET_EN_FRAC = 0.368
STEPS_PER_EPOCH = 1200

NUM_CLASSES = 3
LABEL_SMOOTHING = 0.05
PATIENCE = 4

ENABLE_STAGE2_FINETUNE = False
STAGE2_EPOCHS = 2
UNFREEZE_LAST_N_PARAM_TENSORS = 8
SAVE_WEIGHTED_BEST = True

# 情感增强模块配置
PROJ_DIM = 256
HIDDEN_DIM = 512
DROPOUT = 0.4
NUM_HEADS = 4
AUX_LOSS_WEIGHT_MAX = 0.02
AUX_WARMUP_EPOCHS = 4
EN_CLASS1_BOOST = 1.12
ZH_CLASS2_BOOST = 1.07
LOSS_CLASS_WEIGHT_POWER = 0.45
FOCAL_GAMMA = 1.20
GATE_CENTER_LOSS_WEIGHT = 0.25
GATE_LANG_BALANCE_WEIGHT = 0.50
GATE_DROPOUT = 0.10
PATCH_EMO_TEMPERATURE = 0.7
TOKEN_EMO_TEMPERATURE = 0.7
GATE_CENTER_MARGIN = 0.03
EXPORT_BEST_VAL_OUTPUTS = True

BASE_SAVE_DIR = SAVE_DIR

# =========================
# 消融实验配置
# =========================
# 说明：
# full: 完整模型
# no_region: 去掉图像区域情感注意力，region_vec 置零，patch token 只做普通投影
# no_token: 去掉文本 token 情感加权，text_local 置零，token 权重改为 masked uniform
# no_cross: 去掉文本->图像 patch 的跨模态注意力，cross_vec 置零
# no_gate: 去掉动态门控，固定 gate=0.5
# no_matching: 去掉差异特征和逐元素乘积特征，保留前四个主特征
# global_only: 只保留 img_g 与 txt_g 的基础融合，用来当强基线
ABLATION_EXPERIMENTS = {
    "full": {
        "use_region_attention": True,
        "use_token_emotion": True,
        "use_cross_attention": True,
        "use_gate_fusion": True,
        "use_matching_features": True,
    },
    "no_region": {
        "use_region_attention": False,
        "use_token_emotion": True,
        "use_cross_attention": True,
        "use_gate_fusion": True,
        "use_matching_features": True,
    },
    "no_token": {
        "use_region_attention": True,
        "use_token_emotion": False,
        "use_cross_attention": True,
        "use_gate_fusion": True,
        "use_matching_features": True,
    },
    "no_cross": {
        "use_region_attention": True,
        "use_token_emotion": True,
        "use_cross_attention": False,
        "use_gate_fusion": True,
        "use_matching_features": True,
    },
    "no_gate": {
        "use_region_attention": True,
        "use_token_emotion": True,
        "use_cross_attention": True,
        "use_gate_fusion": False,
        "use_matching_features": True,
    },
    "no_matching": {
        "use_region_attention": True,
        "use_token_emotion": True,
        "use_cross_attention": True,
        "use_gate_fusion": True,
        "use_matching_features": False,
    },
    "global_only": {
        "use_region_attention": False,
        "use_token_emotion": False,
        "use_cross_attention": False,
        "use_gate_fusion": False,
        "use_matching_features": False,
    },
}


def get_ablation_config(name: str):
    if name not in ABLATION_EXPERIMENTS:
        raise ValueError(f"未知消融实验: {name}. 可选: {list(ABLATION_EXPERIMENTS.keys())}")
    return dict(ABLATION_EXPERIMENTS[name])

# v8_2: 优先尝试使用视觉编码器的语义 patch hidden states；失败时自动退回原 pixel patch。
USE_VISUAL_HIDDEN_PATCHES = True
VISUAL_PATCH_FALLBACK = True


# =========================
# 工具函数
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def count_trainable_params(module):
    total = 0
    trainable = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def flatten_patch_tensor(pixel_values: torch.Tensor) -> torch.Tensor:
    """
    兼容 image_processor 的不同输出格式。
    目标统一成 [B, N, D_patch]。
    """
    if pixel_values.dim() == 5:
        # [B, N, C, H, W]
        return pixel_values.flatten(2)
    if pixel_values.dim() == 4:
        # 可能是 [B, N, H, W] 或 [B, N, P, D]
        return pixel_values.flatten(2)
    if pixel_values.dim() == 3:
        # [B, N, D]
        return pixel_values
    return pixel_values.view(pixel_values.size(0), pixel_values.size(1), -1)


def build_patch_mask(flat_patches: torch.Tensor) -> torch.Tensor:
    # [B, N]
    return flat_patches.abs().sum(dim=-1) > 0


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    scores = scores.masked_fill(~mask, -1e4)
    probs = F.softmax(scores, dim=dim)
    probs = probs * mask.to(probs.dtype)
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-6)
    return probs / denom


def print_dataset_stats(ds, name):
    labels = ds.df["label"].astype(int).tolist()
    cnt = Counter(labels)
    print(f"\n[{name}] size={len(ds)}")
    for k in sorted(cnt.keys()):
        print(f"  label {k}: {cnt[k]}")


def get_global_class_weights(zh_train, en_train, num_classes=3):
    counts = torch.zeros(num_classes, dtype=torch.float)

    for y in zh_train.df["label"].astype(int).tolist():
        counts[y] += 1
    for y in en_train.df["label"].astype(int).tolist():
        counts[y] += 1

    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.mean()
    return weights, counts


def build_joint_sample_weights(
    zh_train,
    en_train,
    target_en_frac=TARGET_EN_FRAC,
    num_classes=3,
    en_class1_boost=EN_CLASS1_BOOST,
    zh_class2_boost=ZH_CLASS2_BOOST,
):
    n_zh = len(zh_train)
    n_en = len(en_train)

    p = target_en_frac
    w_zh_lang = 1.0
    w_en_lang = (p / max(1e-6, (1 - p))) * (n_zh / max(1, n_en))

    cls_weights, cls_counts = get_global_class_weights(zh_train, en_train, num_classes=num_classes)
    cls_weights_sqrt = torch.pow(cls_weights, LOSS_CLASS_WEIGHT_POWER)

    weights = []
    zh_labels = zh_train.df["label"].astype(int).tolist()
    en_labels = en_train.df["label"].astype(int).tolist()

    for y in zh_labels:
        extra = zh_class2_boost if y == 2 else 1.0
        weights.append(w_zh_lang * cls_weights_sqrt[y].item() * extra)

    for y in en_labels:
        extra = en_class1_boost if y == 1 else 1.0
        weights.append(w_en_lang * cls_weights_sqrt[y].item() * extra)

    weights = torch.tensor(weights, dtype=torch.double)

    print("\n[Sampler]")
    print(f"target_en_frac={p:.3f}")
    print(f"lang weights: zh={w_zh_lang:.4f}, en={w_en_lang:.4f}")
    print(f"class counts: {cls_counts.tolist()}")
    print(f"raw class weights: {[round(x, 4) for x in cls_weights.tolist()]}")
    print(f"pow({LOSS_CLASS_WEIGHT_POWER}) class weights: {[round(x, 4) for x in cls_weights_sqrt.tolist()]}")
    print(f"en_class1_boost={en_class1_boost:.3f}")
    print(f"zh_class2_boost={zh_class2_boost:.3f}")

    return weights


class SmoothedFocalLoss(nn.Module):
    def __init__(self, class_weights=None, label_smoothing=0.0, gamma=1.5):
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.label_smoothing = float(label_smoothing)
        self.gamma = float(gamma)

    def forward(self, logits, target):
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        with torch.no_grad():
            smooth = self.label_smoothing
            true_dist = torch.full_like(log_probs, smooth / max(num_classes - 1, 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smooth)

        focal_factor = (1.0 - probs).pow(self.gamma)
        loss = -(true_dist * focal_factor * log_probs)

        if self.class_weights is not None:
            weight = self.class_weights.view(1, -1)
            loss = loss * weight

        loss = loss.sum(dim=-1)
        return loss.mean()


# =========================
# Dataset
# =========================
class ImageTextDataset(Dataset):
    def __init__(self, csv_path: str, image_processor, tokenizer, lang_tag: str):
        self.df = pd.read_csv(csv_path)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.lang_tag = lang_tag
        self.lang_id = 0 if lang_tag == "zh" else 1

        required_cols = {"image_path", "text", "label"}
        miss = required_cols - set(self.df.columns)
        if len(miss) > 0:
            raise ValueError(f"{csv_path} 缺少必要列: {miss}")

        # 避免在每次采样时使用 pandas.iloc；WeightedRandomSampler 会频繁重复采样。
        self.image_paths = self.df["image_path"].astype(str).tolist()
        self.texts = self.df["text"].fillna("").astype(str).tolist()
        self.labels = self.df["label"].astype(int).tolist()

        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.pad_id = self.tokenizer.eos_token_id
            else:
                self.pad_id = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        text = self.texts[idx]
        label = self.labels[idx]

        if self.lang_tag == "zh":
            text = "Language: Chinese. Text: " + text
        else:
            text = "Language: English. Text: " + text

        try:
            with Image.open(img_path) as img:
                image_input = self.image_processor(
                    images=img.convert("RGB"),
                    max_num_patches=MAX_NUM_PATCHES,
                    return_tensors="pt",
                )
        except Exception as e:
            raise RuntimeError(f"读取图片失败: {img_path} | error: {e}") from e

        pixel_values = image_input["pixel_values"].squeeze(0)
        pixel_attention_mask = image_input.get("pixel_attention_mask")
        if pixel_attention_mask is None:
            pixel_attention_mask = build_patch_mask(flatten_patch_tensor(pixel_values.unsqueeze(0))).squeeze(0)
        else:
            pixel_attention_mask = pixel_attention_mask.squeeze(0).bool()
        spatial_shapes = image_input["spatial_shapes"].squeeze(0)

        tok = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].squeeze(0)

        if "attention_mask" in tok:
            attention_mask = tok["attention_mask"].squeeze(0)
        else:
            attention_mask = (input_ids != self.pad_id).long()

        return (
            pixel_values,
            pixel_attention_mask,
            spatial_shapes,
            input_ids,
            attention_mask,
            torch.tensor(label, dtype=torch.long),
            self.lang_id,
        )


def collate_fn(batch):
    pixel_values, pixel_attention_mask, spatial_shapes, input_ids, attention_mask, labels, language_ids = zip(*batch)
    return (
        torch.stack(pixel_values, dim=0),
        torch.stack(pixel_attention_mask, dim=0),
        torch.stack(spatial_shapes, dim=0),
        torch.stack(input_ids, dim=0),
        torch.stack(attention_mask, dim=0),
        torch.stack(labels, dim=0),
        torch.tensor(language_ids, dtype=torch.long),
    )


# =========================
# backbone 特征提取
# =========================
def _get_nested_attr(obj, attr_path):
    cur = obj
    for part in attr_path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def get_text_embedding_layer(model, tokenizer=None):
    """
    Fgclip2Model 没有实现 get_input_embeddings()，所以这里做一个更稳的多级回退：
    1) 先尝试标准 get_input_embeddings()
    2) 再尝试常见文本 embedding 路径
    3) 最后在所有 nn.Embedding 里按名字/词表大小启发式选择

    解析结果按 path 缓存在函数属性上（同架构的不同 model 实例复用同一相对路径），
    避免每个 batch 都重复走一遍 try/except + 属性遍历。
    """
    cached_path = getattr(get_text_embedding_layer, "_cached_path", None)
    if cached_path is not None:
        if cached_path == "__get_input_embeddings__":
            emb = model.get_input_embeddings()
        else:
            emb = _get_nested_attr(model, cached_path)
        if isinstance(emb, nn.Embedding):
            return emb

    # 1) 标准 Hugging Face 接口
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            setattr(get_text_embedding_layer, "_cached_path", "__get_input_embeddings__")
            return emb
    except (NotImplementedError, AttributeError):
        pass

    # 2) 常见属性路径
    candidate_paths = [
        "text_model.embeddings.token_embedding",
        "text_model.embeddings.word_embeddings",
        "text_model.model.embed_tokens",
        "text_model.embed_tokens",
        "language_model.model.embed_tokens",
        "language_model.embed_tokens",
        "model.text_model.embeddings.token_embedding",
        "model.text_model.embeddings.word_embeddings",
        "model.text_model.model.embed_tokens",
        "model.text_model.embed_tokens",
        "model.language_model.model.embed_tokens",
        "model.language_model.embed_tokens",
    ]
    for p in candidate_paths:
        emb = _get_nested_attr(model, p)
        if isinstance(emb, nn.Embedding):
            setattr(get_text_embedding_layer, "_cached_path", p)
            return emb

    # 3) 启发式搜索所有 Embedding 层
    vocab_size = None
    if tokenizer is not None:
        try:
            vocab_size = len(tokenizer)
        except Exception:
            vocab_size = None
    if vocab_size is None:
        try:
            vocab_size = int(model.config.vocab_size)
        except Exception:
            vocab_size = None

    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            score = 0
            lname = name.lower()
            if "token" in lname or "word" in lname or "embed_tokens" in lname:
                score += 5
            if "position" in lname or "pos" in lname:
                score -= 10
            if vocab_size is not None:
                if module.num_embeddings == vocab_size:
                    score += 8
                elif abs(module.num_embeddings - vocab_size) <= 8:
                    score += 3
            candidates.append((score, name, module))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[2].embedding_dim), reverse=True)
        best_score, best_name, best_module = candidates[0]
        print(f"[Embedding fallback] use text embedding: {best_name} | "
              f"num_embeddings={best_module.num_embeddings}, dim={best_module.embedding_dim}, score={best_score}")
        setattr(get_text_embedding_layer, "_cached_path", best_name)
        return best_module

    raise RuntimeError(
        "未找到可用的文本 embedding 层。当前 Fgclip2Model 未实现 get_input_embeddings()，"
        "且在模型子模块中也没有识别到合适的 nn.Embedding。"
    )


def _find_visual_encoder(model):
    """
    v8_2: 尝试找到 FG-CLIP2 内部视觉编码器。
    不同 trust_remote_code 版本的模块命名可能不同，所以这里采用多级候选路径。
    如果找不到，会返回 None，后续自动 fallback 到原始 pixel patch。
    """
    cached_path = getattr(_find_visual_encoder, "_cached_path", None)
    cached_missing = getattr(_find_visual_encoder, "_cached_missing", False)

    if cached_missing:
        return None

    if cached_path is not None:
        module = _get_nested_attr(model, cached_path)
        if module is not None:
            return module

    candidate_paths = [
        "vision_model",
        "visual",
        "image_model",
        "model.vision_model",
        "model.visual",
        "model.image_model",
        "vision_tower",
        "model.vision_tower",
        "visual_encoder",
        "model.visual_encoder",
    ]

    for path in candidate_paths:
        module = _get_nested_attr(model, path)
        if module is not None:
            setattr(_find_visual_encoder, "_cached_path", path)
            if not getattr(_find_visual_encoder, "_printed", False):
                print(f"[Visual encoder] use: {path}")
                setattr(_find_visual_encoder, "_printed", True)
            return module

    setattr(_find_visual_encoder, "_cached_missing", True)
    if not getattr(_find_visual_encoder, "_printed_missing", False):
        print("[Warning] visual encoder module not found; fallback to pixel patches.")
        setattr(_find_visual_encoder, "_printed_missing", True)
    return None


def _find_text_encoder(model):
    cached_path = getattr(_find_text_encoder, "_cached_path", None)
    if cached_path is not None:
        module = _get_nested_attr(model, cached_path)
        if module is not None:
            return module

    candidate_paths = [
        "text_model",
        "model.text_model",
        "language_model",
        "model.language_model",
    ]
    for path in candidate_paths:
        module = _get_nested_attr(model, path)
        if module is not None:
            setattr(_find_text_encoder, "_cached_path", path)
            if not getattr(_find_text_encoder, "_printed", False):
                print(f"[Text encoder] use: {path}")
                setattr(_find_text_encoder, "_printed", True)
            return module
    return None


def _pick_hidden_patch_tensor(outputs):
    """
    从视觉编码器输出中提取 patch-level hidden states。
    目标形状：[B, N, D]。
    """
    hidden = None

    # last_hidden_state 已经是最终层输出；优先使用它，避免请求并保留所有层 hidden_states。
    if hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
    elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        hidden = outputs.hidden_states[-1]
    elif isinstance(outputs, dict):
        if outputs.get("last_hidden_state", None) is not None:
            hidden = outputs["last_hidden_state"]
        elif outputs.get("hidden_states", None) is not None:
            hidden = outputs["hidden_states"][-1]
    elif isinstance(outputs, (tuple, list)):
        for x in outputs:
            if torch.is_tensor(x) and x.dim() == 3:
                hidden = x
                break
            if isinstance(x, (tuple, list)):
                for y in x:
                    if torch.is_tensor(y) and y.dim() == 3:
                        hidden = y
                        break
                if hidden is not None:
                    break

    if hidden is None or (not torch.is_tensor(hidden)) or hidden.dim() != 3:
        return None

    return hidden


def _pick_pooled_tensor(outputs):
    pooled = getattr(outputs, "pooler_output", None)
    if torch.is_tensor(pooled):
        return pooled
    if isinstance(outputs, dict):
        pooled = outputs.get("pooler_output")
        if torch.is_tensor(pooled):
            return pooled
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if torch.is_tensor(item) and item.dim() == 2:
                return item
    return None


def _call_visual_encoder_once(visual_encoder, pixel_values, pixel_attention_mask, spatial_shapes):
    variants = [
        dict(pixel_values=pixel_values, attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes),
        dict(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes),
        dict(pixel_values=pixel_values, spatial_shapes=spatial_shapes),
        dict(pixel_values=pixel_values),
    ]
    cached = getattr(_call_visual_encoder_once, "_cached_variant", None)
    order = ([cached] if cached is not None else []) + [i for i in range(len(variants)) if i != cached]
    last_error = None
    for index in order:
        try:
            outputs = visual_encoder(**variants[index])
            setattr(_call_visual_encoder_once, "_cached_variant", index)
            return outputs
        except TypeError as e:
            last_error = e
    raise RuntimeError(f"all visual encoder call variants failed; last_error={last_error}")


def _call_text_encoder_once(text_encoder, input_ids, attention_mask):
    variants = [
        dict(input_ids=input_ids, attention_mask=attention_mask, walk_type=WALK_TYPE),
        dict(input_ids=input_ids, attention_mask=attention_mask),
        dict(input_ids=input_ids),
    ]
    cached = getattr(_call_text_encoder_once, "_cached_variant", None)
    order = ([cached] if cached is not None else []) + [i for i in range(len(variants)) if i != cached]
    last_error = None
    for index in order:
        try:
            outputs = text_encoder(**variants[index])
            setattr(_call_text_encoder_once, "_cached_variant", index)
            return outputs
        except TypeError as e:
            last_error = e
    raise RuntimeError(f"all text encoder call variants failed; last_error={last_error}")


def _fallback_image_global(model, pixel_values, pixel_attention_mask, spatial_shapes):
    variants = [
        dict(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        ),
        dict(pixel_values=pixel_values, spatial_shapes=spatial_shapes),
    ]
    last_error = None
    for kwargs in variants:
        try:
            return model.get_image_features(**kwargs)
        except TypeError as e:
            last_error = e
    raise RuntimeError(f"get_image_features failed; last_error={last_error}")


def _apply_text_pooling_head(model, pooled_output):
    if WALK_TYPE == "box" and hasattr(model, "boxtext_head"):
        return model.boxtext_head(pooled_output)
    if WALK_TYPE == "long" and hasattr(model, "longtext_head"):
        return model.longtext_head(pooled_output)
    return pooled_output


def extract_backbone_features(
    model,
    pixel_values,
    pixel_attention_mask,
    spatial_shapes,
    input_ids,
    attention_mask,
    require_grad=False,
):
    """视觉和文本主干各执行一次，同时返回全局与 token/patch 特征。"""
    flat_pixel_patches = flatten_patch_tensor(pixel_values)
    if pixel_attention_mask is None:
        patch_mask = build_patch_mask(flat_pixel_patches)
    else:
        patch_mask = pixel_attention_mask.bool()

    ctx = nullcontext() if require_grad else torch.no_grad()
    with ctx:
        visual_patches = None
        visual_encoder = _find_visual_encoder(model)
        try:
            if visual_encoder is None:
                raise RuntimeError("visual encoder module not found")
            vision_outputs = _call_visual_encoder_once(
                visual_encoder,
                pixel_values=pixel_values,
                pixel_attention_mask=patch_mask,
                spatial_shapes=spatial_shapes,
            )
            img_f = _pick_pooled_tensor(vision_outputs)
            if img_f is None:
                raise RuntimeError("visual encoder returned no pooled output")
            if USE_VISUAL_HIDDEN_PATCHES:
                visual_patches = _pick_hidden_patch_tensor(vision_outputs)
        except Exception as e:
            if not VISUAL_PATCH_FALLBACK:
                raise RuntimeError(f"视觉主干单次前向失败: {e}") from e
            if not getattr(extract_backbone_features, "_printed_visual_error", False):
                print(f"[Warning] visual one-pass extraction failed; use image-global + pixel patches. error={e}")
                setattr(extract_backbone_features, "_printed_visual_error", True)
            img_f = _fallback_image_global(model, pixel_values, patch_mask, spatial_shapes)

        if visual_patches is not None:
            if visual_patches.size(1) == patch_mask.size(1) + 1:
                # 部分视觉模型在最前面附加 CLS token。
                visual_patches = visual_patches[:, 1:, :]
            elif visual_patches.size(1) != patch_mask.size(1):
                message = (
                    f"visual patch/mask length mismatch: patches={visual_patches.size(1)}, "
                    f"mask={patch_mask.size(1)}"
                )
                if not VISUAL_PATCH_FALLBACK:
                    raise RuntimeError(message)
                if not getattr(extract_backbone_features, "_printed_patch_mismatch", False):
                    print(f"[Warning] {message}; fallback to pixel patches.")
                    setattr(extract_backbone_features, "_printed_patch_mismatch", True)
                visual_patches = None

        if visual_patches is None:
            visual_patches = flat_pixel_patches

        text_encoder = _find_text_encoder(model)
        try:
            if text_encoder is None:
                raise RuntimeError("text encoder module not found")
            text_outputs = _call_text_encoder_once(text_encoder, input_ids, attention_mask)
            txt_f = _pick_pooled_tensor(text_outputs)
            contextual_token_embeds = _pick_hidden_patch_tensor(text_outputs)
            if txt_f is None or contextual_token_embeds is None:
                raise RuntimeError("text encoder returned incomplete pooled/token outputs")
            txt_f = _apply_text_pooling_head(model, txt_f)
            if TOKEN_FEATURE_SOURCE == "contextual":
                token_embeds = contextual_token_embeds
            elif TOKEN_FEATURE_SOURCE == "embedding":
                emb_layer = get_text_embedding_layer(model, tokenizer=TOKENIZER)
                token_embeds = emb_layer(input_ids)
            else:
                raise ValueError(f"unknown TOKEN_FEATURE_SOURCE: {TOKEN_FEATURE_SOURCE}")
        except Exception as e:
            if not getattr(extract_backbone_features, "_printed_text_error", False):
                print(f"[Warning] contextual text extraction failed; fallback to legacy path. error={e}")
                setattr(extract_backbone_features, "_printed_text_error", True)
            txt_f = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                walk_type=WALK_TYPE,
            )
            emb_layer = get_text_embedding_layer(model, tokenizer=TOKENIZER)
            token_embeds = emb_layer(input_ids)

    if not getattr(extract_backbone_features, "_printed_shapes", False):
        print(
            f"[Backbone one-pass] image={tuple(img_f.shape)}, text={tuple(txt_f.shape)}, "
            f"patches={tuple(visual_patches.shape)}, tokens={tuple(token_embeds.shape)}, "
            f"token_source={TOKEN_FEATURE_SOURCE}, "
            f"valid_patches_mean={patch_mask.float().sum(dim=1).mean().item():.1f}"
        )
        setattr(extract_backbone_features, "_printed_shapes", True)

    return img_f, txt_f, visual_patches, token_embeds, patch_mask


# =========================
# 情感增强模块
# =========================
class RegionEmotionAttention(nn.Module):
    """
    显著情感区域建模：
    - v8_2 优先使用视觉编码器 hidden patch；失败时 fallback 到 image_processor patch
    - 用 text global feature 作为 query，对 patch 做区域情感注意力
    """
    def __init__(self, patch_dim: int, txt_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(txt_dim),
            nn.Linear(txt_dim, hidden_dim),
        )
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.val_proj = nn.Linear(hidden_dim, hidden_dim)
        self.patch_emo = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.scale = math.sqrt(hidden_dim)

    def forward(self, flat_patches, txt_global, patch_mask=None):
        # flat_patches: [B, N, Dp]
        patch_mask = build_patch_mask(flat_patches) if patch_mask is None else patch_mask.bool()
        patch_h = self.patch_proj(flat_patches)

        q = self.query_proj(txt_global).unsqueeze(1)              # [B,1,H]
        k = self.key_proj(patch_h)                                # [B,N,H]
        v = self.val_proj(patch_h)                                # [B,N,H]

        align_scores = (q * k).sum(dim=-1) / self.scale           # [B,N]
        emo_scores = self.patch_emo(patch_h).squeeze(-1) / PATCH_EMO_TEMPERATURE
        attn_scores = align_scores + emo_scores
        attn_weights = masked_softmax(attn_scores, patch_mask, dim=-1)

        region_vec = torch.sum(attn_weights.unsqueeze(-1) * v, dim=1)
        region_vec = self.out_proj(region_vec)

        # 让显著区域信息回流到 patch token
        patch_h_emo = patch_h * (1.0 + attn_weights.unsqueeze(-1))
        return patch_h_emo, region_vec, attn_weights, patch_mask


class TokenEmotionEncoder(nn.Module):
    """
    文本情感增强：
    - 复用 FG-CLIP2 文本编码器的上下文化 token hidden states
    - 学习 token 级情感权重，再做加权池化得到 text_local
    """
    def __init__(self, token_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.emo_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, token_embeds, attention_mask):
        # token_embeds: [B,T,E]
        token_h = self.token_proj(token_embeds)
        token_mask = attention_mask.bool()
        emo_scores = self.emo_head(token_h).squeeze(-1) / TOKEN_EMO_TEMPERATURE
        token_weights = masked_softmax(emo_scores, token_mask, dim=-1)
        token_h_emo = token_h * (1.0 + token_weights.unsqueeze(-1))
        text_local = torch.sum(token_weights.unsqueeze(-1) * token_h_emo, dim=1)
        text_local = self.out_proj(text_local)
        return token_h_emo, text_local, token_weights, token_mask


class EmotionWeightedCrossAttention(nn.Module):
    """
    情感加权跨模态注意力：
    query = 文本 token
    key/value = 显著区域 patch
    最终用 token-level emotion weight 对 cross output 做池化
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        token_h_emo,
        token_weights,
        token_mask,
        patch_h_emo,
        patch_mask,
        return_attention=False,
    ):
        key_padding_mask = ~patch_mask
        cross_tokens, attn_map = self.cross_attn(
            query=token_h_emo,
            key=patch_h_emo,
            value=patch_h_emo,
            key_padding_mask=key_padding_mask,
            # need_weights=False 可启用 PyTorch 的优化 SDPA 路径；训练损失并未使用注意力图。
            need_weights=return_attention,
            average_attn_weights=False,
        )
        cross_tokens = self.out_proj(cross_tokens)
        cross_vec = torch.sum(token_weights.unsqueeze(-1) * cross_tokens, dim=1)
        return cross_tokens, cross_vec, attn_map


class EmotionGatedFusionClassifier(nn.Module):
    def __init__(
        self,
        img_dim: int,
        txt_dim: int,
        patch_dim: int,
        token_dim: int,
        proj_dim: int = 256,
        hidden_dim: int = 512,
        num_classes: int = 3,
        dropout: float = 0.4,
        num_heads: int = 4,
        ablation_cfg=None,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.ablation_cfg = dict(ablation_cfg or ABLATION_EXPERIMENTS["full"])

        self.img_global_proj = nn.Sequential(
            nn.LayerNorm(img_dim),
            nn.Linear(img_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.txt_global_proj = nn.Sequential(
            nn.LayerNorm(txt_dim),
            nn.Linear(txt_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.region_attention = RegionEmotionAttention(
            patch_dim=patch_dim,
            txt_dim=txt_dim,
            hidden_dim=proj_dim,
            dropout=dropout,
        )
        self.token_emotion = TokenEmotionEncoder(
            token_dim=token_dim,
            hidden_dim=proj_dim,
            dropout=dropout,
        )
        # The final no-cross checkpoint is structurally pruned: when cross-attention
        # is disabled, no inactive MultiheadAttention parameters are instantiated or
        # serialized. Full/no-region/no-token ablations still instantiate the module.
        self.cross_attention = None
        if self.ablation_cfg.get("use_cross_attention", True):
            self.cross_attention = EmotionWeightedCrossAttention(
                hidden_dim=proj_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        # No-cross uses the four inputs stated in the final paper equation
        # [image_global, text_global, region, text_local]. Full keeps the fifth
        # cross vector. This also removes the otherwise permanently zero input
        # block from the final checkpoint's LayerNorm and first gate projection.
        self.gate_input_features = 5 if self.ablation_cfg.get("use_cross_attention", True) else 4
        self.gate_net = nn.Sequential(
            nn.LayerNorm(proj_dim * self.gate_input_features),
            nn.Linear(proj_dim * self.gate_input_features, proj_dim * 2),
            nn.GELU(),
            nn.Dropout(GATE_DROPOUT),
            nn.Linear(proj_dim * 2, proj_dim),
        )

        # 为了让所有消融实验可以共用同一个分类器输入维度，禁用某些模块时用 0 向量占位。
        # 8 对应 forward() 里两处 torch.cat 分支拼接的特征数：
        # img_g, txt_g, z_fusion, z_cross, global_diff, global_prod, local_diff, local_prod。
        # 若增删拼接的特征，必须同步改这里，否则会在 self.classifier 里报 shape mismatch；
        # 下面 forward() 里两处 torch.cat 后都有 assert 会先一步提示。
        self.num_fusion_features = 8
        self.fusion_dim = proj_dim * self.num_fusion_features
        fusion_dim = self.fusion_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _uniform_token_weights(self, attention_mask):
        token_mask = attention_mask.bool()
        scores = torch.zeros_like(attention_mask, dtype=torch.float)
        return masked_softmax(scores, token_mask, dim=-1), token_mask

    def _uniform_patch_weights(self, flat_patches, patch_mask=None):
        patch_mask = build_patch_mask(flat_patches) if patch_mask is None else patch_mask.bool()
        scores = torch.zeros(flat_patches.size(0), flat_patches.size(1), device=flat_patches.device, dtype=flat_patches.dtype)
        return masked_softmax(scores, patch_mask, dim=-1), patch_mask

    def forward(
        self,
        img_global,
        txt_global,
        flat_patches,
        patch_mask,
        token_embeds,
        attention_mask,
        language_ids=None,
        return_aux=True,
    ):
        cfg = self.ablation_cfg
        use_region_attention = cfg.get("use_region_attention", True)
        use_token_emotion = cfg.get("use_token_emotion", True)
        use_cross_attention = cfg.get("use_cross_attention", True)
        use_gate_fusion = cfg.get("use_gate_fusion", True)
        use_matching_features = cfg.get("use_matching_features", True)

        img_g = self.img_global_proj(img_global)
        txt_g = self.txt_global_proj(txt_global)
        zero_vec = torch.zeros_like(img_g)

        is_global_only = not any(
            [
                use_region_attention,
                use_token_emotion,
                use_cross_attention,
                use_gate_fusion,
                use_matching_features,
            ]
        )
        if is_global_only:
            gate = torch.full_like(img_g, 0.5)
            z_fusion = gate * img_g + (1.0 - gate) * txt_g
            feat = torch.cat(
                [img_g, txt_g, z_fusion, zero_vec, zero_vec, zero_vec, zero_vec, zero_vec],
                dim=-1,
            )
            assert feat.size(-1) == self.fusion_dim, (
                f"fusion feature count changed ({feat.size(-1) // self.proj_dim} != "
                f"{self.num_fusion_features}); update self.num_fusion_features."
            )
            logits = self.classifier(feat)
            if not return_aux:
                return logits
            patch_weights, patch_mask = self._uniform_patch_weights(flat_patches, patch_mask)
            token_weights, _ = self._uniform_token_weights(attention_mask)
            return logits, {
                "patch_weights": patch_weights,
                "token_weights": token_weights,
                "cross_attn": None,
                "gate_mean": gate.mean().detach(),
                "gate": gate,
                "language_ids": language_ids,
                "region_vec": zero_vec,
                "text_local": zero_vec,
                "cross_vec": zero_vec,
                "ablation_cfg": cfg,
            }

        # 1) 图像 patch / region 分支
        if use_region_attention:
            patch_h_emo, region_vec, patch_weights, patch_mask = self.region_attention(
                flat_patches, txt_global, patch_mask=patch_mask
            )
        else:
            # no_region: 不做文本引导区域池化，也不产生 region_vec。
            # 但为了 no_region + cross_attention 能继续测试“普通 patch token 的 cross attention”，
            # 这里保留 patch token 的普通投影，不加入区域情感权重回流。
            patch_mask = build_patch_mask(flat_patches) if patch_mask is None else patch_mask.bool()
            patch_h_emo = self.region_attention.patch_proj(flat_patches)
            patch_weights, _ = self._uniform_patch_weights(flat_patches, patch_mask)
            region_vec = zero_vec

        # 2) 文本 token emotion 分支
        if use_token_emotion:
            token_h_emo, text_local, token_weights, token_mask = self.token_emotion(token_embeds, attention_mask)
        else:
            # no_token: 不学习 token 情感权重，token 权重退化为 masked uniform，text_local 置零。
            token_h = self.token_emotion.token_proj(token_embeds)
            token_weights, token_mask = self._uniform_token_weights(attention_mask)
            token_h_emo = token_h
            text_local = zero_vec

        # 3) 跨模态注意力分支
        if use_cross_attention:
            if self.cross_attention is None:
                raise RuntimeError("cross-attention is enabled in the config but absent from the model")
            _cross_tokens, cross_vec, attn_map = self.cross_attention(
                token_h_emo=token_h_emo,
                token_weights=token_weights,
                token_mask=token_mask,
                patch_h_emo=patch_h_emo,
                patch_mask=patch_mask,
                return_attention=False,
            )
        else:
            cross_vec = zero_vec
            attn_map = None

        # 4) 动态门控分支
        if use_gate_fusion:
            gate_parts = [img_g, txt_g, region_vec, text_local]
            if use_cross_attention:
                gate_parts.append(cross_vec)
            gate_in = torch.cat(gate_parts, dim=-1)
            assert gate_in.size(-1) == self.proj_dim * self.gate_input_features
            gate_logits = self.gate_net(gate_in)
            # 只使用当前样本自身的统计量，既保持 batch 不变性，也防止语言整体偏向某一模态。
            gate_logits = gate_logits - gate_logits.mean(dim=-1, keepdim=True)
            gate = torch.sigmoid(gate_logits)
        else:
            # no_gate: 固定 0.5，等价于图文全局平均融合。
            gate = torch.full_like(img_g, 0.5)

        # 5) 融合向量
        z_fusion = gate * img_g + (1.0 - gate) * txt_g
        if use_region_attention or use_token_emotion:
            z_fusion = z_fusion + 0.5 * (region_vec + text_local)

        if use_cross_attention or use_region_attention:
            z_cross = gate * cross_vec + (1.0 - gate) * region_vec
        else:
            z_cross = zero_vec

        if use_matching_features:
            global_diff = torch.abs(img_g - txt_g)
            global_prod = img_g * txt_g
            local_diff = torch.abs(region_vec - text_local)
            local_prod = region_vec * text_local
        else:
            global_diff = zero_vec
            global_prod = zero_vec
            local_diff = zero_vec
            local_prod = zero_vec

        feat = torch.cat(
            [
                img_g,
                txt_g,
                z_fusion,
                z_cross,
                global_diff,
                global_prod,
                local_diff,
                local_prod,
            ],
            dim=-1,
        )
        assert feat.size(-1) == self.fusion_dim, (
            f"fusion feature count changed ({feat.size(-1) // self.proj_dim} != "
            f"{self.num_fusion_features}); update self.num_fusion_features."
        )

        logits = self.classifier(feat)

        if not return_aux:
            return logits

        aux = {
            "patch_weights": patch_weights,
            "token_weights": token_weights,
            "cross_attn": attn_map,
            "gate_mean": gate.mean().detach(),
            "gate": gate,
            "language_ids": language_ids,
            "region_vec": region_vec,
            "text_local": text_local,
            "cross_vec": cross_vec,
            "ablation_cfg": cfg,
        }
        return logits, aux


# =========================
# 评估与打分
# =========================
def compute_aux_loss(aux_dict):
    """
    v6 改进：
    1) 不再使用带明显负偏移的熵项；
    2) 改成正值 regularization，便于观察和退火；
    3) 显式约束 gate 不要偏向某一语言。
    """
    cfg = aux_dict.get("ablation_cfg", ABLATION_EXPERIMENTS["full"])
    gate = aux_dict["gate"]
    language_ids = aux_dict.get("language_ids", None)
    aux_loss = gate.new_zeros(())

    if cfg.get("use_region_attention", True):
        patch_w = aux_dict["patch_weights"].clamp_min(1e-6)
        aux_loss = aux_loss + 0.35 * patch_w.pow(2).sum(dim=-1).mean()
    if cfg.get("use_token_emotion", True):
        token_w = aux_dict["token_weights"].clamp_min(1e-6)
        aux_loss = aux_loss + 0.35 * token_w.pow(2).sum(dim=-1).mean()

    if cfg.get("use_gate_fusion", True):
        gate_mean_per_sample = gate.mean(dim=-1)
        gate_center_dev = (gate_mean_per_sample - 0.5).abs() - GATE_CENTER_MARGIN
        gate_center = F.relu(gate_center_dev).pow(2).mean()
        aux_loss = aux_loss + GATE_CENTER_LOSS_WEIGHT * gate_center

        if language_ids is not None:
            zh_mask = (language_ids == 0)
            en_mask = (language_ids == 1)
            if zh_mask.any() and en_mask.any():
                zh_mean = gate_mean_per_sample[zh_mask].mean()
                en_mean = gate_mean_per_sample[en_mask].mean()
                aux_loss = aux_loss + GATE_LANG_BALANCE_WEIGHT * (zh_mean - en_mean).pow(2)
    return aux_loss


@torch.inference_mode()
def evaluate_one(model, clf, loader, name: str, criterion, collect_outputs=False):
    model.eval()
    clf.eval()

    loss_sum = torch.zeros((), device=DEVICE, dtype=torch.float32)
    total = 0
    label_batches = []
    pred_batches = []
    logit_batches = []
    language_batches = []
    gate_batches = []
    zh_count = 0
    en_count = 0

    for pixel_values, pixel_attention_mask, spatial_shapes, input_ids, attention_mask, labels, language_ids in loader:
        zh_count += int((language_ids == 0).sum().item())
        en_count += int((language_ids == 1).sum().item())
        pixel_values = pixel_values.to(DEVICE, non_blocking=True)
        pixel_attention_mask = pixel_attention_mask.to(DEVICE, non_blocking=True)
        spatial_shapes = spatial_shapes.to(DEVICE, non_blocking=True)
        input_ids = input_ids.to(DEVICE, non_blocking=True)
        attention_mask = attention_mask.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        language_ids = language_ids.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"), dtype=torch.float16):
            logits, aux = forward_batch(
                model=model,
                clf=clf,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
                input_ids=input_ids,
                attention_mask=attention_mask,
                language_ids=language_ids,
                require_grad_backbone=False,
            )
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1)
        bs = labels.size(0)
        loss_sum += loss.detach().float() * bs
        total += bs
        label_batches.append(labels)
        pred_batches.append(preds)
        gate_batches.append(aux["gate"].mean(dim=-1))
        if collect_outputs:
            logit_batches.append(logits.float())
            language_batches.append(language_ids)

    labels_all = torch.cat(label_batches).cpu().numpy()
    preds_all = torch.cat(pred_batches).cpu().numpy()
    gates_all = torch.cat(gate_batches).float().cpu().numpy()
    avg_loss = float((loss_sum / max(total, 1)).item())
    acc = accuracy_score(labels_all, preds_all)
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
    per_class_f1 = f1_score(
        labels_all,
        preds_all,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    gate_mean = float(gates_all.mean()) if gates_all.size else 0.0

    print(f"\n[{name}]")
    print(f"loss={avg_loss:.4f} acc={acc:.4f} macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f} n={total}")
    print(f"lang count | zh={zh_count}, en={en_count}")
    print(f"gate_mean={gate_mean:.4f}")
    print(f"per-class f1: {[round(x, 4) for x in per_class_f1.tolist()]}")
    print("confusion_matrix:")
    print(confusion_matrix(labels_all, preds_all, labels=list(range(NUM_CLASSES))))
    print("classification_report:")
    print(
        classification_report(
            labels_all,
            preds_all,
            digits=4,
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
    )

    metrics = {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1.tolist(),
        "n": total,
        "gate_mean": gate_mean,
    }
    outputs = None
    if collect_outputs:
        logits_all = torch.cat(logit_batches).cpu()
        outputs = {
            "logits": logits_all.numpy(),
            "probs": torch.softmax(logits_all, dim=-1).numpy(),
            "labels": labels_all,
            "preds": preds_all,
            "lang_ids": torch.cat(language_batches).cpu().numpy(),
            "gate_mean": gates_all,
        }
    return metrics, outputs


def compute_scores(zh_metrics, en_metrics):
    balanced_macro = (zh_metrics["macro_f1"] + en_metrics["macro_f1"]) / 2.0
    balanced_acc = (zh_metrics["acc"] + en_metrics["acc"]) / 2.0
    weighted_macro = (
        zh_metrics["macro_f1"] * zh_metrics["n"] +
        en_metrics["macro_f1"] * en_metrics["n"]
    ) / max(1, zh_metrics["n"] + en_metrics["n"])
    return balanced_macro, balanced_acc, weighted_macro


# =========================
# 冻结 / 解冻
# =========================
def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_last_n_param_tensors(model, n=8):
    frozen_params = [p for p in model.parameters() if p.requires_grad is False]
    if n <= 0 or len(frozen_params) == 0:
        return 0

    n = min(n, len(frozen_params))
    for p in frozen_params[-n:]:
        p.requires_grad = True
    return n


# =========================
# checkpoint
# =========================
def save_checkpoint(
    save_path,
    model,
    clf,
    optimizer,
    scheduler,
    epoch,
    best_score,
    config_dict,
    extra_metrics=None,
    include_training_state=True,
):
    ckpt = {
        "epoch": epoch,
        "best_score": best_score,
        "clf_state_dict": clf.state_dict(),
        "config": config_dict,
        "extra_metrics": extra_metrics,
    }
    if include_training_state:
        ckpt["optimizer_state_dict"] = optimizer.state_dict() if optimizer is not None else None
        ckpt["scheduler_state_dict"] = scheduler.state_dict() if scheduler is not None else None

    any_backbone_trainable = any(p.requires_grad for p in model.parameters())
    if any_backbone_trainable:
        ckpt["model_state_dict"] = model.state_dict()

    torch.save(ckpt, save_path)


def save_best_checkpoint(
    tag,
    score,
    save_dir,
    model,
    clf,
    optimizer,
    scheduler,
    epoch,
    config_dict,
    extra_metrics,
    val_outputs=None,
):
    """
    保存某个 tag（balanced/weighted/zh/en）当前最优的 checkpoint，
    并按需保存对应的验证输出。val_outputs 形如 {"zh_val": outputs, "en_val": outputs}。
    """
    save_checkpoint(
        save_path=os.path.join(save_dir, f"best_head_{tag}.ckpt"),
        model=model,
        clf=clf,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        best_score=score,
        config_dict=config_dict,
        extra_metrics=extra_metrics,
        include_training_state=False,
    )
    if val_outputs is not None:
        for split_name, outputs in val_outputs.items():
            save_validation_outputs(
                outputs,
                split_name,
                os.path.join(save_dir, f"best_{tag}_{split_name}_outputs.npz"),
            )
    print(f"[BEST] {tag} model updated at epoch {epoch}, score={score:.4f}")


# =========================
# 训练
# =========================
def forward_batch(
    model,
    clf,
    pixel_values,
    pixel_attention_mask,
    spatial_shapes,
    input_ids,
    attention_mask,
    language_ids,
    require_grad_backbone=False,
):
    img_f, txt_f, flat_patches, token_embeds, patch_mask = extract_backbone_features(
        model=model,
        pixel_values=pixel_values,
        pixel_attention_mask=pixel_attention_mask,
        spatial_shapes=spatial_shapes,
        input_ids=input_ids,
        attention_mask=attention_mask,
        require_grad=require_grad_backbone,
    )

    logits, aux = clf(
        img_global=img_f,
        txt_global=txt_f,
        flat_patches=flat_patches,
        patch_mask=patch_mask,
        token_embeds=token_embeds,
        attention_mask=attention_mask,
        language_ids=language_ids,
        return_aux=True,
    )
    return logits, aux


def train_one_epoch(
    model,
    clf,
    loader,
    optimizer,
    scaler,
    criterion,
    epoch_idx,
    require_grad_backbone=False,
    aux_warmup_epoch_idx=None,
):
    # 部分解冻时仍保持冻结主干的 dropout 为 eval，避免冻结特征随 batch 随机波动。
    model.eval()
    clf.train()

    # aux_warmup_epoch_idx 默认等于 epoch_idx（Stage1 行为不变）。
    # Stage2 微调会重新从 epoch 1 计数，若仍用 epoch_idx 算 warmup，会导致
    # aux_loss 权重在微调阶段重新从 0 爬升，而不是延续 Stage1 已完成的 warmup。
    if aux_warmup_epoch_idx is None:
        aux_warmup_epoch_idx = epoch_idx

    loss_sum = torch.zeros((), device=DEVICE, dtype=torch.float32)
    total = 0
    label_batches = []
    pred_batches = []
    seen_zh = 0
    seen_en = 0

    for step, (
        pixel_values,
        pixel_attention_mask,
        spatial_shapes,
        input_ids,
        attention_mask,
        labels,
        language_ids,
    ) in enumerate(loader, start=1):
        seen_zh += int((language_ids == 0).sum().item())
        seen_en += int((language_ids == 1).sum().item())
        pixel_values = pixel_values.to(DEVICE, non_blocking=True)
        pixel_attention_mask = pixel_attention_mask.to(DEVICE, non_blocking=True)
        spatial_shapes = spatial_shapes.to(DEVICE, non_blocking=True)
        input_ids = input_ids.to(DEVICE, non_blocking=True)
        attention_mask = attention_mask.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        language_ids = language_ids.to(DEVICE, non_blocking=True)
        aux_weight = AUX_LOSS_WEIGHT_MAX * min(1.0, aux_warmup_epoch_idx / max(1, AUX_WARMUP_EPOCHS))

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"), dtype=torch.float16):
            logits, aux = forward_batch(
                model=model,
                clf=clf,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
                input_ids=input_ids,
                attention_mask=attention_mask,
                language_ids=language_ids,
                require_grad_backbone=require_grad_backbone,
            )
            ce_loss = criterion(logits, labels)
            aux_loss = compute_aux_loss(aux)
            loss = ce_loss + aux_weight * aux_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=-1)
        bs = labels.size(0)
        loss_sum += loss.detach().float() * bs
        total += bs
        label_batches.append(labels)
        pred_batches.append(preds)

        if step % 100 == 0 or step == len(loader):
            frac_en = seen_en / max(1, seen_en + seen_zh)
            print(
                f"epoch={epoch_idx} step={step}/{len(loader)} "
                f"loss={loss.item():.4f} ce={ce_loss.item():.4f} aux={aux_loss.item():.4f} "
                f"aux_w={aux_weight:.4f} seen_en_frac={frac_en:.3f} gate_mean={float(aux['gate_mean'].item()):.4f}"
            )

    labels_all = torch.cat(label_batches).cpu().numpy()
    preds_all = torch.cat(pred_batches).cpu().numpy()
    avg_loss = float((loss_sum / max(total, 1)).item())
    acc = accuracy_score(labels_all, preds_all)
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)

    print(f"\n[TRAIN] epoch={epoch_idx} loss={avg_loss:.4f} acc={acc:.4f} macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f}")

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n": total,
    }


def save_validation_outputs(outputs, split_name: str, out_path: str):
    if outputs is None:
        raise ValueError("validation outputs were not collected")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        split=split_name,
        **outputs,
    )


# =========================
# main
# =========================
def run_experiment(experiment_name: str, args):
    global SAVE_DIR, NUM_EPOCHS, STEPS_PER_EPOCH, TOKEN_FEATURE_SOURCE, TOKENIZER, SEED
    global ZH_TRAIN_CSV, ZH_VAL_CSV, ZH_TEST_CSV, EN_TRAIN_CSV, EN_VAL_CSV, EN_TEST_CSV

    ablation_cfg = get_ablation_config(experiment_name)
    TOKEN_FEATURE_SOURCE = args.token_feature_source
    save_root = args.save_root if args.save_root is not None else BASE_SAVE_DIR
    SAVE_DIR = os.path.join(save_root, experiment_name)
    if args.epochs is not None:
        NUM_EPOCHS = int(args.epochs)
    if args.steps_per_epoch is not None:
        STEPS_PER_EPOCH = int(args.steps_per_epoch)
    if getattr(args, "seed", None) is not None:
        SEED = int(args.seed)
    if args.zh_train_csv is not None:
        ZH_TRAIN_CSV = args.zh_train_csv
    if args.zh_val_csv is not None:
        ZH_VAL_CSV = args.zh_val_csv
    if getattr(args, "zh_test_csv", None) is not None:
        ZH_TEST_CSV = args.zh_test_csv
    if args.en_train_csv is not None:
        EN_TRAIN_CSV = args.en_train_csv
    if args.en_val_csv is not None:
        EN_VAL_CSV = args.en_val_csv
    if getattr(args, "en_test_csv", None) is not None:
        EN_TEST_CSV = args.en_test_csv

    set_seed(SEED)
    ensure_dir(SAVE_DIR)

    print("\n" + "=" * 90)
    print(f"ABLATION EXPERIMENT: {experiment_name}")
    print(f"Ablation config: {json.dumps(ablation_cfg, ensure_ascii=False)}")
    print(f"Token feature source: {TOKEN_FEATURE_SOURCE}")
    print(f"SAVE_DIR: {SAVE_DIR}")
    print("=" * 90)
    print("DEVICE:", DEVICE)
    print("Loading processor/tokenizer/model.")

    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision="430fbc8a912c86fd4de601381b6245a0edab22f0", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision="430fbc8a912c86fd4de601381b6245a0edab22f0", trust_remote_code=True)
    TOKENIZER = tokenizer

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision="430fbc8a912c86fd4de601381b6245a0edab22f0", trust_remote_code=True).to(DEVICE)

    # Stage 1 先冻结 backbone，只训增强头
    freeze_backbone(model)
    model.eval()

    zh_train = ImageTextDataset(ZH_TRAIN_CSV, image_processor, tokenizer, lang_tag="zh")
    en_train = ImageTextDataset(EN_TRAIN_CSV, image_processor, tokenizer, lang_tag="en")
    zh_val = ImageTextDataset(ZH_VAL_CSV, image_processor, tokenizer, lang_tag="zh")
    en_val = ImageTextDataset(EN_VAL_CSV, image_processor, tokenizer, lang_tag="en")
    zh_test = ImageTextDataset(ZH_TEST_CSV, image_processor, tokenizer, lang_tag="zh") if ZH_TEST_CSV else None
    en_test = ImageTextDataset(EN_TEST_CSV, image_processor, tokenizer, lang_tag="en") if EN_TEST_CSV else None

    print_dataset_stats(zh_train, "ZH TRAIN")
    print_dataset_stats(en_train, "EN TRAIN")
    print_dataset_stats(zh_val, "ZH VAL")
    print_dataset_stats(en_val, "EN VAL")
    if zh_test is not None and en_test is not None:
        print_dataset_stats(zh_test, "ZH TEST")
        print_dataset_stats(en_test, "EN TEST")

    train_ds = ConcatDataset([zh_train, en_train])

    weights = build_joint_sample_weights(
        zh_train,
        en_train,
        target_en_frac=TARGET_EN_FRAC,
        num_classes=NUM_CLASSES,
        zh_class2_boost=ZH_CLASS2_BOOST,
    )

    num_samples = STEPS_PER_EPOCH * BATCH_SIZE
    sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)
    worker_kwargs = {
        "num_workers": NUM_WORKERS,
        "persistent_workers": (NUM_WORKERS > 0),
    }
    if NUM_WORKERS > 0:
        worker_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,
        **worker_kwargs,
    )

    zh_val_loader = DataLoader(
        zh_val,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collate_fn,
        **worker_kwargs,
    )

    en_val_loader = DataLoader(
        en_val,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collate_fn,
        **worker_kwargs,
    )

    zh_test_loader = None
    en_test_loader = None
    if zh_test is not None and en_test is not None:
        zh_test_loader = DataLoader(
            zh_test,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            pin_memory=(DEVICE == "cuda"),
            collate_fn=collate_fn,
            **worker_kwargs,
        )
        en_test_loader = DataLoader(
            en_test,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            pin_memory=(DEVICE == "cuda"),
            collate_fn=collate_fn,
            **worker_kwargs,
        )

    print("\nInferring feature dims.")
    pv, pm, ss, ids, am, _labels, _language_ids = next(iter(train_loader))
    pv = pv.to(DEVICE)
    pm = pm.to(DEVICE)
    ss = ss.to(DEVICE)
    ids = ids.to(DEVICE)
    am = am.to(DEVICE)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"), dtype=torch.float16):
        img_f, txt_f, flat_patches, token_embeds, _patch_mask = extract_backbone_features(
            model=model,
            pixel_values=pv,
            pixel_attention_mask=pm,
            spatial_shapes=ss,
            input_ids=ids,
            attention_mask=am,
            require_grad=False,
        )
        img_dim = img_f.shape[-1]
        txt_dim = txt_f.shape[-1]
        patch_dim = flat_patches.shape[-1]
        token_dim = token_embeds.shape[-1]

    print(f"img_dim={img_dim}, txt_dim={txt_dim}, patch_dim={patch_dim}, token_dim={token_dim}")

    clf = EmotionGatedFusionClassifier(
        img_dim=img_dim,
        txt_dim=txt_dim,
        patch_dim=patch_dim,
        token_dim=token_dim,
        proj_dim=PROJ_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        num_heads=NUM_HEADS,
        ablation_cfg=ablation_cfg,
    ).to(DEVICE)
    del pv, pm, ss, ids, am, img_f, txt_f, flat_patches, token_embeds, _patch_mask

    model_total, model_trainable = count_trainable_params(model)
    clf_total, clf_trainable = count_trainable_params(clf)
    print(f"Backbone params: total={model_total:,}, trainable={model_trainable:,}")
    print(f"\nClassifier params: total={clf_total:,}, trainable={clf_trainable:,}")

    class_weights, class_counts = get_global_class_weights(zh_train, en_train, num_classes=NUM_CLASSES)
    loss_class_weights = torch.pow(class_weights, LOSS_CLASS_WEIGHT_POWER).to(DEVICE)

    print("\nLoss setup:")
    print(f"class_counts={class_counts.tolist()}")
    print(f"raw_class_weights={[round(x, 4) for x in class_weights.tolist()]}")
    print(f"loss_class_weight_power={LOSS_CLASS_WEIGHT_POWER}")
    print(f"loss_class_weights={[round(x, 4) for x in loss_class_weights.cpu().tolist()]}")
    print(f"label_smoothing={LABEL_SMOOTHING}")
    print(f"focal_gamma={FOCAL_GAMMA}")
    print(f"aux_loss_weight_max={AUX_LOSS_WEIGHT_MAX}")
    print(f"aux_warmup_epochs={AUX_WARMUP_EPOCHS}")
    print(f"en_class1_boost={EN_CLASS1_BOOST}")
    print(f"zh_class2_boost={ZH_CLASS2_BOOST}")

    criterion = SmoothedFocalLoss(
        class_weights=loss_class_weights,
        label_smoothing=LABEL_SMOOTHING,
        gamma=FOCAL_GAMMA,
    )

    optimizer = torch.optim.AdamW(clf.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    config_dict = {
        "MODEL_ID": MODEL_ID,
        "MAX_NUM_PATCHES": MAX_NUM_PATCHES,
        "MAX_TEXT_LEN": MAX_TEXT_LEN,
        "WALK_TYPE": WALK_TYPE,
        "BATCH_SIZE": BATCH_SIZE,
        "EVAL_BATCH_SIZE": EVAL_BATCH_SIZE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "HEAD_LR": HEAD_LR,
        "BACKBONE_LR": BACKBONE_LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "TARGET_EN_FRAC": TARGET_EN_FRAC,
        "STEPS_PER_EPOCH": STEPS_PER_EPOCH,
        "NUM_CLASSES": NUM_CLASSES,
        "LABEL_SMOOTHING": LABEL_SMOOTHING,
        "PATIENCE": PATIENCE,
        "ENABLE_STAGE2_FINETUNE": ENABLE_STAGE2_FINETUNE,
        "STAGE2_EPOCHS": STAGE2_EPOCHS,
        "UNFREEZE_LAST_N_PARAM_TENSORS": UNFREEZE_LAST_N_PARAM_TENSORS,
        "PROJ_DIM": PROJ_DIM,
        "HIDDEN_DIM": HIDDEN_DIM,
        "DROPOUT": DROPOUT,
        "NUM_HEADS": NUM_HEADS,
        "AUX_LOSS_WEIGHT_MAX": AUX_LOSS_WEIGHT_MAX,
        "AUX_WARMUP_EPOCHS": AUX_WARMUP_EPOCHS,
        "EN_CLASS1_BOOST": EN_CLASS1_BOOST,
        "ZH_CLASS2_BOOST": ZH_CLASS2_BOOST,
        "LOSS_CLASS_WEIGHT_POWER": LOSS_CLASS_WEIGHT_POWER,
        "FOCAL_GAMMA": FOCAL_GAMMA,
        "GATE_CENTER_LOSS_WEIGHT": GATE_CENTER_LOSS_WEIGHT,
        "GATE_CENTER_MARGIN": GATE_CENTER_MARGIN,
        "GATE_LANG_BALANCE_WEIGHT": GATE_LANG_BALANCE_WEIGHT,
        "ONE_PASS_BACKBONE": True,
        "TOKEN_FEATURE_SOURCE": TOKEN_FEATURE_SOURCE,
        "GATE_MODE": "sample_centered",
        "USE_EXACT_PIXEL_ATTENTION_MASK": True,
        "SAVE_BEST_WITHOUT_OPTIMIZER_STATE": True,
        "SEED": SEED,
        "img_dim": img_dim,
        "txt_dim": txt_dim,
        "patch_dim": patch_dim,
        "token_dim": token_dim,
        "experiment_name": experiment_name,
        "ablation_cfg": ablation_cfg,
        "architecture_variant": (
            "structurally_pruned_no_cross"
            if not ablation_cfg.get("use_cross_attention", True)
            else "cross_attention_instantiated"
        ),
        "ZH_TRAIN_CSV": ZH_TRAIN_CSV,
        "ZH_VAL_CSV": ZH_VAL_CSV,
        "ZH_TEST_CSV": ZH_TEST_CSV,
        "EN_TRAIN_CSV": EN_TRAIN_CSV,
        "EN_VAL_CSV": EN_VAL_CSV,
        "EN_TEST_CSV": EN_TEST_CSV,
    }

    print("\n========== Stage 1: Train emotion-enhanced classifier head ==========")
    stage1_start_time = time.perf_counter()
    best_balanced_score = -1.0
    best_weighted_score = -1.0
    best_zh_score = -1.0
    best_en_score = -1.0
    best_epoch = -1
    bad_epochs = 0

    # 记录 best balanced epoch 对应的更完整指标
    best_balanced_acc = -1.0
    best_balanced_zh_acc = -1.0
    best_balanced_en_acc = -1.0
    best_balanced_zh_macro_f1 = -1.0
    best_balanced_en_macro_f1 = -1.0
    best_balanced_zh_weighted_f1 = -1.0
    best_balanced_en_weighted_f1 = -1.0

    # 记录各语言自身最优 macro-F1 及该 epoch 对应的 acc
    best_zh_acc = -1.0
    best_en_acc = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} | lr={current_lr:.8f} =====")

        _train_metrics = train_one_epoch(
            model=model,
            clf=clf,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            epoch_idx=epoch,
            require_grad_backbone=False,
        )

        zh_metrics, zh_val_outputs = evaluate_one(
            model,
            clf,
            zh_val_loader,
            "ZH VAL",
            criterion,
            collect_outputs=EXPORT_BEST_VAL_OUTPUTS,
        )
        en_metrics, en_val_outputs = evaluate_one(
            model,
            clf,
            en_val_loader,
            "EN VAL",
            criterion,
            collect_outputs=EXPORT_BEST_VAL_OUTPUTS,
        )
        balanced_macro, balanced_acc, weighted_macro = compute_scores(zh_metrics, en_metrics)

        print(
            f"\n[Epoch {epoch}] "
            f"balanced_acc={balanced_acc:.4f} "
            f"balanced_macro_f1={balanced_macro:.4f} "
            f"weighted_macro_f1={weighted_macro:.4f}"
        )

        scheduler.step(balanced_macro)

        extra_metrics = {
            "zh_metrics": zh_metrics,
            "en_metrics": en_metrics,
            "balanced_acc": balanced_acc,
            "balanced_macro_f1": balanced_macro,
            "weighted_macro_f1": weighted_macro,
        }

        save_checkpoint(
            save_path=os.path.join(SAVE_DIR, "last_head.ckpt"),
            model=model,
            clf=clf,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_score=best_balanced_score,
            config_dict=config_dict,
            extra_metrics=extra_metrics,
        )

        if balanced_macro > best_balanced_score:
            best_balanced_score = balanced_macro
            best_epoch = epoch
            bad_epochs = 0

            # 保存 best balanced epoch 对应的中英文详细指标
            best_balanced_acc = balanced_acc
            best_balanced_zh_acc = zh_metrics["acc"]
            best_balanced_en_acc = en_metrics["acc"]
            best_balanced_zh_macro_f1 = zh_metrics["macro_f1"]
            best_balanced_en_macro_f1 = en_metrics["macro_f1"]
            best_balanced_zh_weighted_f1 = zh_metrics["weighted_f1"]
            best_balanced_en_weighted_f1 = en_metrics["weighted_f1"]

            save_best_checkpoint(
                tag="balanced",
                score=best_balanced_score,
                save_dir=SAVE_DIR,
                model=model,
                clf=clf,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config_dict=config_dict,
                extra_metrics=extra_metrics,
                val_outputs=(
                    {"zh_val": zh_val_outputs, "en_val": en_val_outputs}
                    if EXPORT_BEST_VAL_OUTPUTS else None
                ),
            )
        else:
            bad_epochs += 1
            print(f"No balanced improvement. bad_epochs={bad_epochs}/{PATIENCE}")

        if SAVE_WEIGHTED_BEST and weighted_macro > best_weighted_score:
            best_weighted_score = weighted_macro
            save_best_checkpoint(
                tag="weighted",
                score=best_weighted_score,
                save_dir=SAVE_DIR,
                model=model,
                clf=clf,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config_dict=config_dict,
                extra_metrics=extra_metrics,
                val_outputs=(
                    {"zh_val": zh_val_outputs, "en_val": en_val_outputs}
                    if EXPORT_BEST_VAL_OUTPUTS else None
                ),
            )

        if zh_metrics["macro_f1"] > best_zh_score:
            best_zh_score = zh_metrics["macro_f1"]
            best_zh_acc = zh_metrics["acc"]
            save_best_checkpoint(
                tag="zh",
                score=best_zh_score,
                save_dir=SAVE_DIR,
                model=model,
                clf=clf,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config_dict=config_dict,
                extra_metrics=extra_metrics,
            )

        if en_metrics["macro_f1"] > best_en_score:
            best_en_score = en_metrics["macro_f1"]
            best_en_acc = en_metrics["acc"]
            save_best_checkpoint(
                tag="en",
                score=best_en_score,
                save_dir=SAVE_DIR,
                model=model,
                clf=clf,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config_dict=config_dict,
                extra_metrics=extra_metrics,
            )

        if bad_epochs >= PATIENCE:
            print("[STOP] Early stopping triggered.")
            break

    stage1_training_seconds = time.perf_counter() - stage1_start_time
    print(f"\nStage 1 best epoch={best_epoch}, best balanced_macro_f1={best_balanced_score:.4f}")
    print(f"Stage 1 training time={stage1_training_seconds:.2f}s")

    if ENABLE_STAGE2_FINETUNE:
        print("\n========== Stage 2: Finetune backbone + emotion head ==========")
        best_ckpt_path = os.path.join(SAVE_DIR, "best_head_balanced.ckpt")
        ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        clf.load_state_dict(ckpt["clf_state_dict"])

        unfrozen_n = unfreeze_last_n_param_tensors(model, n=UNFREEZE_LAST_N_PARAM_TENSORS)
        print(f"Unfroze last {unfrozen_n} parameter tensors of backbone.")

        model_total, model_trainable = count_trainable_params(model)
        clf_total, clf_trainable = count_trainable_params(clf)
        print(f"Backbone params: total={model_total:,}, trainable={model_trainable:,}")
        print(f"Classifier params: total={clf_total:,}, trainable={clf_trainable:,}")

        ft_optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in model.parameters() if p.requires_grad], "lr": BACKBONE_LR},
                {"params": clf.parameters(), "lr": HEAD_LR * 0.5},
            ],
            weight_decay=WEIGHT_DECAY,
        )
        ft_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            ft_optimizer,
            mode="max",
            factor=0.5,
            patience=1,
        )
        ft_scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

        ft_best_score = best_balanced_score
        ft_bad_epochs = 0

        for epoch in range(1, STAGE2_EPOCHS + 1):
            lrs = [pg["lr"] for pg in ft_optimizer.param_groups]
            print(f"\n===== FT Epoch {epoch}/{STAGE2_EPOCHS} | lrs={lrs} =====")

            _ft_train_metrics = train_one_epoch(
                model=model,
                clf=clf,
                loader=train_loader,
                optimizer=ft_optimizer,
                scaler=ft_scaler,
                criterion=criterion,
                epoch_idx=epoch,
                require_grad_backbone=True,
                # Stage1 已经完成过 aux_loss warmup；这里固定给一个 >= AUX_WARMUP_EPOCHS
                # 的值，让微调阶段从第一步就使用满额 aux 权重，而不是重新爬升。
                aux_warmup_epoch_idx=AUX_WARMUP_EPOCHS,
            )

            zh_metrics, _ = evaluate_one(model, clf, zh_val_loader, "ZH VAL [FT]", criterion)
            en_metrics, _ = evaluate_one(model, clf, en_val_loader, "EN VAL [FT]", criterion)
            balanced_macro, balanced_acc, weighted_macro = compute_scores(zh_metrics, en_metrics)

            print(
                f"\n[FT Epoch {epoch}] "
                f"balanced_acc={balanced_acc:.4f} "
                f"balanced_macro_f1={balanced_macro:.4f} "
                f"weighted_macro_f1={weighted_macro:.4f}"
            )

            ft_scheduler.step(balanced_macro)

            extra_metrics = {
                "zh_metrics": zh_metrics,
                "en_metrics": en_metrics,
                "balanced_acc": balanced_acc,
                "balanced_macro_f1": balanced_macro,
                "weighted_macro_f1": weighted_macro,
            }

            save_checkpoint(
                save_path=os.path.join(SAVE_DIR, "last_finetune.ckpt"),
                model=model,
                clf=clf,
                optimizer=ft_optimizer,
                scheduler=ft_scheduler,
                epoch=epoch,
                best_score=ft_best_score,
                config_dict=config_dict,
                extra_metrics=extra_metrics,
            )

            if balanced_macro > ft_best_score:
                ft_best_score = balanced_macro
                ft_bad_epochs = 0
                save_checkpoint(
                    save_path=os.path.join(SAVE_DIR, "best_finetune_balanced.ckpt"),
                    model=model,
                    clf=clf,
                    optimizer=ft_optimizer,
                    scheduler=ft_scheduler,
                    epoch=epoch,
                    best_score=ft_best_score,
                    config_dict=config_dict,
                    extra_metrics=extra_metrics,
                    include_training_state=False,
                )
                print(f"[BEST] FT model updated, balanced_macro_f1={ft_best_score:.4f}")
            else:
                ft_bad_epochs += 1
                print(f"FT no improvement. bad_epochs={ft_bad_epochs}/{max(2, PATIENCE // 2)}")

            if ft_bad_epochs >= max(2, PATIENCE // 2):
                print("[STOP] FT early stopping triggered.")
                break

        print(f"\nStage 2 best balanced_macro_f1={ft_best_score:.4f}")

    # The untouched test set is evaluated exactly once after model selection.
    # Validation remains the only source for early stopping and checkpoint choice.
    test_metrics = None
    if zh_test_loader is not None and en_test_loader is not None:
        selected_checkpoint = os.path.join(SAVE_DIR, "best_head_balanced.ckpt")
        finetune_checkpoint = os.path.join(SAVE_DIR, "best_finetune_balanced.ckpt")
        if ENABLE_STAGE2_FINETUNE and os.path.exists(finetune_checkpoint):
            selected_checkpoint = finetune_checkpoint
        selected = torch.load(selected_checkpoint, map_location=DEVICE, weights_only=False)
        clf.load_state_dict(selected["clf_state_dict"])

        print("\n========== Untouched test evaluation ==========")
        zh_test_metrics, zh_test_outputs = evaluate_one(
            model, clf, zh_test_loader, "ZH TEST", criterion, collect_outputs=True
        )
        en_test_metrics, en_test_outputs = evaluate_one(
            model, clf, en_test_loader, "EN TEST", criterion, collect_outputs=True
        )
        test_balanced_macro, test_balanced_acc, test_weighted_macro = compute_scores(
            zh_test_metrics, en_test_metrics
        )
        test_metrics = {
            "selected_checkpoint": selected_checkpoint,
            "zh": zh_test_metrics,
            "en": en_test_metrics,
            "balanced_macro_f1": test_balanced_macro,
            "balanced_acc": test_balanced_acc,
            "weighted_macro_f1": test_weighted_macro,
        }
        save_validation_outputs(zh_test_outputs, "zh_test", os.path.join(SAVE_DIR, "test_outputs_zh.npz"))
        save_validation_outputs(en_test_outputs, "en_test", os.path.join(SAVE_DIR, "test_outputs_en.npz"))
        with open(os.path.join(SAVE_DIR, "test_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(test_metrics, handle, ensure_ascii=False, indent=2)
        print(
            f"[TEST SUMMARY] balanced_acc={test_balanced_acc:.4f} "
            f"balanced_macro_f1={test_balanced_macro:.4f}"
        )

    print("\n[DONE] Training finished.")
    print(f"Checkpoints saved to: {SAVE_DIR}")

    result = {
        "experiment": experiment_name,
        "best_epoch": best_epoch,

        # best balanced epoch 的总指标
        "best_balanced_macro_f1": best_balanced_score,
        "best_balanced_acc": best_balanced_acc,

        # best balanced epoch 对应的中英文指标
        "best_balanced_zh_acc": best_balanced_zh_acc,
        "best_balanced_en_acc": best_balanced_en_acc,
        "best_balanced_zh_macro_f1": best_balanced_zh_macro_f1,
        "best_balanced_en_macro_f1": best_balanced_en_macro_f1,
        "best_balanced_zh_weighted_f1": best_balanced_zh_weighted_f1,
        "best_balanced_en_weighted_f1": best_balanced_en_weighted_f1,

        # best weighted score
        "best_weighted_macro_f1": best_weighted_score,

        # 各语言自身最优 macro-F1 及该 epoch 对应的 acc
        "best_zh_macro_f1": best_zh_score,
        "best_zh_acc": best_zh_acc,
        "best_en_macro_f1": best_en_score,
        "best_en_acc": best_en_acc,

        # complexity and measured Stage-1 runtime
        "backbone_total_params": model_total,
        "backbone_trainable_params": model_trainable,
        "classifier_total_params": clf_total,
        "classifier_trainable_params": clf_trainable,
        "end_to_end_total_params": model_total + clf_total,
        "end_to_end_trainable_params": model_trainable + clf_trainable,
        "training_time_seconds": stage1_training_seconds,

        # untouched test-set metrics (None when test CSVs were not supplied)
        "test_balanced_macro_f1": (
            test_metrics["balanced_macro_f1"] if test_metrics is not None else None
        ),
        "test_balanced_acc": test_metrics["balanced_acc"] if test_metrics is not None else None,
        "test_weighted_macro_f1": (
            test_metrics["weighted_macro_f1"] if test_metrics is not None else None
        ),
        "test_zh_acc": test_metrics["zh"]["acc"] if test_metrics is not None else None,
        "test_en_acc": test_metrics["en"]["acc"] if test_metrics is not None else None,
        "test_zh_macro_f1": (
            test_metrics["zh"]["macro_f1"] if test_metrics is not None else None
        ),
        "test_en_macro_f1": (
            test_metrics["en"]["macro_f1"] if test_metrics is not None else None
        ),

        "save_dir": SAVE_DIR,
    }

    if ENABLE_STAGE2_FINETUNE:
        del ft_optimizer, ft_scheduler, ft_scaler, ckpt
    del model, clf, optimizer, scheduler, scaler
    del train_loader, zh_val_loader, en_val_loader
    del zh_train, en_train, zh_val, en_val, train_ds, sampler, weights
    cleanup_memory()
    print("[DONE] Memory cleaned.")
    return result


def update_ablation_summary(summary_path, result_record):
    """
    追加/更新 ablation_summary.csv。
    单独跑某个实验时，不会覆盖之前实验的结果。
    如果同名 experiment 已存在，则用新结果覆盖旧行。
    """
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    fieldnames = [
        "experiment",
        "best_epoch",

        "best_balanced_macro_f1",
        "best_balanced_acc",

        "best_balanced_zh_acc",
        "best_balanced_en_acc",
        "best_balanced_zh_macro_f1",
        "best_balanced_en_macro_f1",
        "best_balanced_zh_weighted_f1",
        "best_balanced_en_weighted_f1",

        "best_weighted_macro_f1",

        "best_zh_macro_f1",
        "best_zh_acc",
        "best_en_macro_f1",
        "best_en_acc",

        "test_balanced_macro_f1",
        "test_balanced_acc",
        "test_weighted_macro_f1",
        "test_zh_acc",
        "test_en_acc",
        "test_zh_macro_f1",
        "test_en_macro_f1",

        "save_dir",
    ]

    # 只保留需要写入 summary 的字段；缺失字段填 NaN，避免旧结果/旧代码格式报错。
    clean_record = {col: result_record.get(col, np.nan) for col in fieldnames}
    new_df = pd.DataFrame([clean_record], columns=fieldnames)

    if os.path.exists(summary_path):
        old_df = pd.read_csv(summary_path)

        for col in fieldnames:
            if col not in old_df.columns:
                old_df[col] = np.nan

        old_df = old_df[fieldnames]

        if "experiment" in old_df.columns:
            old_df = old_df[old_df["experiment"] != clean_record["experiment"]]

        final_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    preferred_order = [
        "full",
        "global_only",
        "no_region",
        "no_token",
        "no_cross",
        "no_gate",
        "no_matching",
    ]

    final_df["_order"] = final_df["experiment"].apply(
        lambda x: preferred_order.index(x) if x in preferred_order else 999
    )
    final_df = final_df.sort_values("_order").drop(columns=["_order"])

    final_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\n[DONE] Ablation summary updated: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="FG-CLIP2 emotion-enhanced v8_4.1 balanced ablation runner")
    parser.add_argument(
        "--experiment",
        type=str,
        default="full",
        choices=list(ABLATION_EXPERIMENTS.keys()),
        help="选择一个消融实验运行。",
    )
    parser.add_argument(
        "--token_feature_source",
        type=str,
        default=TOKEN_FEATURE_SOURCE,
        choices=["embedding", "contextual"],
        help="局部文本分支特征；embedding 为均衡版默认值，contextual 用于对照实验。",
    )
    parser.add_argument(
        "--run_all_ablations",
        action="store_true",
        help="按顺序运行全部消融实验：full/no_region/no_token/no_cross/no_gate/no_matching/global_only。",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help="所有实验的根保存目录；不填则使用脚本里的 BASE_SAVE_DIR。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="临时覆盖 NUM_EPOCHS，快速试跑时可设为 1 或 2。",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="临时覆盖 STEPS_PER_EPOCH，快速试跑时可设为 100。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="覆盖随机种子；重复实验应显式指定。",
    )
    parser.add_argument(
        "--list_experiments",
        action="store_true",
        help="只打印可用实验名，不运行训练。",
    )
    parser.add_argument(
        "--zh_train_csv",
        type=str,
        default=None,
        help="覆盖中文训练集 CSV 路径；不填则使用脚本里的 ZH_TRAIN_CSV。",
    )
    parser.add_argument(
        "--zh_val_csv",
        type=str,
        default=None,
        help="覆盖中文验证集 CSV 路径；不填则使用脚本里的 ZH_VAL_CSV。",
    )
    parser.add_argument(
        "--zh_test_csv",
        type=str,
        default=None,
        help="中文独立测试集 CSV；仅在验证选出最佳 checkpoint 后评估一次。",
    )
    parser.add_argument(
        "--en_train_csv",
        type=str,
        default=None,
        help="覆盖英文训练集 CSV 路径；不填则使用脚本里的 EN_TRAIN_CSV。",
    )
    parser.add_argument(
        "--en_val_csv",
        type=str,
        default=None,
        help="覆盖英文验证集 CSV 路径；不填则使用脚本里的 EN_VAL_CSV。",
    )
    parser.add_argument(
        "--en_test_csv",
        type=str,
        default=None,
        help="英文独立测试集 CSV；仅在验证选出最佳 checkpoint 后评估一次。",
    )
    args = parser.parse_args()

    if args.list_experiments:
        print("Available ablation experiments:")
        for name, cfg in ABLATION_EXPERIMENTS.items():
            print(f"- {name}: {json.dumps(cfg, ensure_ascii=False)}")
        return

    if args.run_all_ablations:
        experiment_names = [
            "full",
            "no_region",
            "no_token",
            "no_cross",
            "no_gate",
            "no_matching",
            "global_only",
        ]
    else:
        experiment_names = [args.experiment]

    save_root = args.save_root if args.save_root is not None else BASE_SAVE_DIR
    os.makedirs(save_root, exist_ok=True)

    summary_path = os.path.join(save_root, "ablation_summary.csv")
    summary_rows = []

    for exp_name in experiment_names:
        result = run_experiment(exp_name, args)
        update_ablation_summary(summary_path, result)
        summary_rows.append(result)

    print("\n========== Current run summary ==========")
    for row in summary_rows:
        print(
            f"{row['experiment']}: "
            f"best_epoch={row['best_epoch']}, "
            f"balanced_f1={row['best_balanced_macro_f1']:.4f}, "
            f"balanced_acc={row['best_balanced_acc']:.4f}, "
            f"zh_f1@balanced={row['best_balanced_zh_macro_f1']:.4f}, "
            f"zh_acc@balanced={row['best_balanced_zh_acc']:.4f}, "
            f"en_f1@balanced={row['best_balanced_en_macro_f1']:.4f}, "
            f"en_acc@balanced={row['best_balanced_en_acc']:.4f}"
        )

    if os.path.exists(summary_path):
        print("\n========== All saved ablation summary ==========")
        saved_df = pd.read_csv(summary_path)
        print(saved_df.to_string(index=False))


if __name__ == "__main__":
    main()
