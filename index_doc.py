#coding: utf-8


import sys,os
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import faiss


from sentence_transformers import SentenceTransformer

# DMetaSoul/sbert-chinese-general-v2
# thenlper/gte-large
# EMBED_MODEL_ID = "thenlper/gte-large"
model_path = "./emb_models/thenlper/gte-large"
# EMBED_MODEL_ID = "DMetaSoul/sbert-chinese-general-v2-distill"
# EMBED_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
# emb_model = SentenceTransformer("all-MiniLM-L6-v2")
emb_model = SentenceTransformer(model_path,device="cpu" )


doc_name = "model_q.docx"

converter = DocumentConverter()

print("Converting document:", doc_name)
doc = converter.convert(doc_name).document
print("Document converted successfully.")


# 
chunker = HybridChunker()
chunk_iter = chunker.chunk(dl_doc=doc)
    


# 基于BERT架构，采用MiniLM知识蒸馏技术，在保持高性能的同时大幅减小了模型体积
# {
#   "hidden_size": 384,
#   "num_hidden_layers": 6,
#   "num_attention_heads": 12,
#   "intermediate_size": 1536,
#   "max_position_embeddings": 512,
#   "vocab_size": 30522
# }

# EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"   中文语料效果不好

MAX_TOKENS = 100  # set to a small number for illustrative purposes

# tokenizer = HuggingFaceTokenizer(
#     tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
#     max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer` for HF case
# )
# chunker = HybridChunker(
#     tokenizer=tokenizer,
#     merge_peers=True,  # optional, defaults to True
# )
# chunk_iter = chunker.chunk(dl_doc=doc)

chunks = []
len_limit = 500
for i, chunk in enumerate(chunk_iter):
    print(f"====== {i} ======")
    print(f"original text:\n{chunk.text[:len_limit]}")

    enriched_text = chunker.contextualize(chunk=chunk)
    print(f"contextualize(chunk):\n{enriched_text[:len_limit]}")

    print()
    
    chunks.append(enriched_text)

# 存储chunks到本地文件
chunk_file = "doc_chunks.txt"
with open(chunk_file, "w", encoding="utf-8") as f:
    for idx,chunk in enumerate(chunks):
        f.write(f"{idx}:{chunk}\x01")
    
# 生成向量
chunk_embeddings = emb_model.encode(chunks, convert_to_numpy=True)
faiss.normalize_L2(chunk_embeddings)
print("chunk_embeddings.shape: ", chunk_embeddings.shape)

if os.path.exists("rag.index"):
    rag_index = faiss.read_index("rag.index")
    print("Loaded existing FAISS index, total chunks:", rag_index.ntotal)

else:
    rag_index = faiss.IndexFlatL2(chunk_embeddings.shape[1])

rag_index.add(chunk_embeddings.astype("float32"))
print("FAISS index built, total chunks:", rag_index.ntotal)

# 保存FAISS索引
faiss.write_index(rag_index, "rag.index")



# search
query = "巴塔哥尼亚大学"
query_vector = emb_model.encode([query], show_progress_bar=True, convert_to_numpy=True)
print("Query vector shape:", query_vector.shape, type(query_vector))

faiss.normalize_L2(query_vector)
distances, indices = rag_index.search(query_vector, 5)
print("Indices:", indices)
print("Distances:", distances)

print("\n===== Top 5 relevant chunks: =====\n")
for idx in indices[0]:
    print(f"--- chunk {idx} ---")
    print(chunks[idx][:500])
    print()