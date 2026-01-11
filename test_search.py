#coding: utf-8

import sys,os
import faiss
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

query = "复开率"
emb_model = SentenceTransformer("all-MiniLM-L6-v2")
query_vector = emb_model.encode([query], show_progress_bar=True, convert_to_numpy=True)
print("Query vector shape:", query_vector.shape, type(query_vector))


vdb_path = "rag.index"
rag_index = faiss.read_index(vdb_path)
print("Index loaded from", vdb_path, " with ", rag_index.ntotal, " vectors.")


# 查询与query最相似的5个向量
# faiss.normalize_L2(query_vector)
distances, indices = rag_index.search(query_vector, 5)
print("Indices:", indices)
print("Distances:", distances)
