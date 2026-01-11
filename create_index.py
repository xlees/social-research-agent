#coding: utf-8

import sys,os

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer

from pymilvus import MilvusClient

os.environ["TOKENIZERS_PARALLELISM"] = "false"


dim = 1024

vecClient = MilvusClient("./rag_index.db")
all_collections = vecClient.list_collections()

if "latin" not in all_collections:
    vecClient.create_collection(
        collection_name="latin",
        dimension=dim  # The vectors we will use in this demo has 384 dimensions
    )

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

chunker = HybridChunker()
chunk_iter = chunker.chunk(dl_doc=doc)

all_chunks = []
len_limit = 500
for i, chunk in enumerate(chunk_iter):
    print(f"====== {i} ======")
    print(f"original text:\n{chunk.text[:len_limit]}")

    enriched_text = chunker.contextualize(chunk=chunk)
    print(f"contextualize(chunk):\n{enriched_text[:len_limit]}")

    print()
    
    all_chunks.append(enriched_text)

# print(f"Total chunks to insert: {len(all_chunks)}")
print("Generating embeddings for chunks...")
chunk_embeddings = emb_model.encode(all_chunks, convert_to_numpy=True)
print("Embeddings generated.")

vecData = [{
    "id": i,
    "vector": chunk_embeddings[i],
    "text": all_chunks[i]
} for i in range(len(all_chunks))]

print("Inserting chunks into Milvus...")
res = vecClient.insert(
    collection_name="latin",
    data=vecData
)
print(f"Inserted {len(all_chunks)} chunks into Milvus collection 'demo'.")
print(res)


# index
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="vector", # Name of the vector field to be indexed
    index_type="HNSW", # Type of the index to create
    index_name="vector_index", # Name of the index to create
    metric_type="L2", # Metric type used to measure similarity
    params={
        "M": 64, # Maximum number of neighbors each node can connect to in the graph
        "efConstruction": 100 # Number of candidate neighbors considered for connection during index construction
    } # Index building params
)