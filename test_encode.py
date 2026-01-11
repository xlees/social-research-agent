
import os

# 1. 关 tokenizer 并行（mac 必须）
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

# 2. 限制 torch 线程（防止 segfault）
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

from sentence_transformers import SentenceTransformer

# 3. 模型 ID
EMBED_MODEL_ID = "DMetaSoul/sbert-chinese-general-v2-distill"

# 4. 加载模型（走 safetensors，不触发 torch.load）
emb_model = SentenceTransformer(
    EMBED_MODEL_ID,
    device="cpu"   # macOS 11 基本别碰 mps
)

# 5. 使用
vec = emb_model.encode("复开率")

print(len(vec))  # 768（或模型对应维度）

print(vec)

