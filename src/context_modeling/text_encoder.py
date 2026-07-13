"""
文本编码器 — 通过 HuggingFace Transformers 编码生物医学文本。

封装预训练的生物医学语言模型，用于编码 ECG 记录相关的临床文本描述。
支持 BioBERT / PubMedBERT / ClinicalBERT 及任何 HuggingFace AutoModel。

Wraps pretrained biomedical language models for encoding clinical text.
Supports BioBERT, PubMedBERT, ClinicalBERT, or any HuggingFace AutoModel.

使用示例 / Usage:
    encoder = TextEncoder("pubmedbert")  # 或 "biobert" / "clinicalbert"
    features = encoder(["患者表现为胸痛..."])  # → (B, 768)
"""

import logging
from typing import List, Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """生物医学文本编码器 / Biomedical text encoder.

    参数 / Args:
        model_name: 模型标识 / "pubmedbert" | "biobert" | "clinicalbert" | HF model id.
        freeze:     是否冻结预训练权重 / Freeze pretrained weights.
        max_length: 最大 token 长度 / Max token length.
        pooling:    池化策略 / "cls" | "mean" | "max".
    """

    PRETRAINED_MODELS = {
        "biobert":      "dmis-lab/biobert-v1.1",
        "pubmedbert":   "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    }

    def __init__(self, model_name="pubmedbert", freeze=True, max_length=512, pooling="cls", device=None):
        super().__init__()
        if model_name in self.PRETRAINED_MODELS:
            model_name = self.PRETRAINED_MODELS[model_name]

        self.model_name, self.max_length, self.pooling = model_name, max_length, pooling
        self._output_dim = 768  # BERT-base 默认维度

        # 尝试加载模型和分词器 / Try loading model + tokenizer
        try:
            from transformers import AutoModel, AutoTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.encoder = AutoModel.from_pretrained(model_name)
                self._output_dim = self.encoder.config.hidden_size
                if freeze:
                    for p in self.encoder.parameters(): p.requires_grad = False
            except Exception as e:
                logger.warning(f"加载 {model_name} 失败: {e}。使用占位编码器。")
                self.tokenizer = self.encoder = None
        except ImportError:
            logger.warning("transformers 未安装。使用占位编码器。pip install transformers")
            self.tokenizer = self.encoder = None

        if device: self.to(device)

    @property
    def output_dim(self) -> int: return self._output_dim

    def forward(self, texts: Union[str, List[str]], return_tokens=False):
        """编码临床文本 / Encode clinical text.

        参数 / Args:
            texts:        单个字符串或列表 / Single string or list.
            return_tokens: 是否返回 token-level embeddings.

        返回 / Returns:
            (B, output_dim) pooled features, 或 (pooled, tokens) 元组。
        """
        if isinstance(texts, str): texts = [texts]
        if self.encoder is None:
            return self._placeholder_encode(len(texts), return_tokens)

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        with torch.set_grad_enabled(not all(p.requires_grad is False for p in self.encoder.parameters())):
            outputs = self.encoder(**inputs)

        # 池化 / Pooling
        if self.pooling == "cls":    pooled = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == "mean": pooled = (outputs.last_hidden_state * inputs["attention_mask"].unsqueeze(-1).float()).sum(1) / inputs["attention_mask"].unsqueeze(-1).float().sum(1)
        elif self.pooling == "max":  pooled = (outputs.last_hidden_state * inputs["attention_mask"].unsqueeze(-1).float()).max(1).values
        else: raise ValueError(f"未知池化: {self.pooling}")

        return (pooled, outputs.last_hidden_state) if return_tokens else pooled

    def _placeholder_encode(self, batch_size, return_tokens):
        """占位编码（模型不可用时）/ Placeholder when model unavailable."""
        pooled = torch.zeros(batch_size, self._output_dim)
        return (pooled, torch.zeros(batch_size, self.max_length, self._output_dim)) if return_tokens else pooled
