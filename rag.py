#coding: utf-8

import sys,os

from sentence_transformers import SentenceTransformer

from pymilvus import MilvusClient

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def rag_context(query: str, limit=5) -> list:
    dim = 1024

    model_path = "./emb_models/thenlper/gte-large"
    emb_model = SentenceTransformer(model_path, device="cpu" )

    query_vector = emb_model.encode([query], show_progress_bar=True, convert_to_numpy=True)

    vec_client = MilvusClient("./rag.db")
    res = vec_client.search(
        collection_name="whz_latin",
        # anns_field="vector",
        data=query_vector,
        limit=limit,
        output_fields=["offset",  "text", "fname"],
        search_params={"metric_type": "COSINE"}
    )

    result = []
    for hits in res:
        for hit in hits:
            result.append({
                'content': hit.entity.text,
                'score': hit.distance,
                'doc': hit.entity.fname
            })

    return result

