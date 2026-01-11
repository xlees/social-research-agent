#coding: utf-8


import sys,os
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import faiss


from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

model_path = "./emb_models/thenlper/gte-large"
# DMetaSoul/sbert-chinese-general-v2
# thenlper/gte-large
# EMBED_MODEL_ID = "DMetaSoul/sbert-chinese-general-v2-distill"
# EMBED_MODEL_ID = "henlper/gte-large"
# emb_model = SentenceTransformer("all-MiniLM-L6-v2")

emb_model = SentenceTransformer(model_path, device="cpu" )

# emb_model = AutoModel.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading chunks from file...")

chunks = {}
chunk_file = "doc_chunks.txt"
if not os.path.exists(chunk_file):
    pass
else:
    print(f"{chunk_file} already exists, reading...")

    # 读取chunks文件
    with open(chunk_file, "r", encoding="utf-8") as f:
        all_contents = f.read()
        # print("all_contents length:\n", all_contents)
        print()

        for _ in all_contents.split("\x01"):
            print(f"line: {_[:50]} {len(_)}")

            if len(_) > 0:
                ind,chunk = _.split(":", 1)
                chunks[int(ind)] = chunk
    print(f"Loaded {len(chunks)} chunks from {chunk_file}.")


# 搜索
query = "乌拉圭最近几年与阿根廷的关系如何发展"
query_vector = emb_model.encode([query], show_progress_bar=True, convert_to_numpy=True)
print("Query vector shape:", query_vector.shape, type(query_vector))

if os.path.exists("rag.index"):
    rag_index = faiss.read_index("rag.index")
    print("Loaded existing FAISS index, total chunks:", rag_index.ntotal)
else:
    print("FAISS index file 'rag.index' not found. Exiting.")
    sys.exit(1)

faiss.normalize_L2(query_vector)
distances, indices = rag_index.search(query_vector, 5)
print("Indices:", indices)
print("Distances:", distances)

print("\n===== Top 5 relevant chunks: =====\n")
for idx in indices[0]:
    print(f"--- chunk {idx} ---")
    print(chunks[idx][:500])
    print()