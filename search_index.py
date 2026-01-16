#coding: utf-8

import sys,os

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer

from pymilvus import MilvusClient

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_path = "./emb_models/thenlper/gte-large"
emb_model = SentenceTransformer(model_path, device="cpu" )

dim = 1024

vecClient = MilvusClient("./rag.db")
all_collections = vecClient.list_collections()
print("\n===== Existing collections:", all_collections, " =====\n")

# 搜索
# query = "阿根廷国家公园的建设与民族国家建构有什么关系"
query = "拉丁美洲都建了哪些国家公园"
query_vector = emb_model.encode([query], show_progress_bar=True, convert_to_numpy=True)
# print("Query vector shape:", query_vector.shape, type(query_vector),"\n")

res = vecClient.search(
    collection_name="whz_latin",
    # anns_field="vector",
    data=query_vector,
    limit=5,
    output_fields=["offset",  "text", "fname"],
    search_params={"metric_type": "COSINE"}
)
# print(type(res), len(res))

ind = 0
print(f"\n用户问题: {query}")
for hits in res:
    # print("TopK results:")

    for hit in hits:
        print(f"\n{ind+1}.{hit.id}\t'{hit.entity.fname}'\t{hit.entity.offset}\t{hit.distance}")
        print(f"> {hit.entity.text[:500]}")

        ind += 1

    print()
