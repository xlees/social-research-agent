import os
import torch
from volcenginesdkarkruntime import Ark
from typing import Optional, List

def encode(
    client, inputs: List[str], is_query: bool = False, mrl_dim: Optional[int] = None
):
    if is_query:
        # use instruction for optimal performance, feel free to tune this instruction for different tasks
        # to reproduce MTEB results, refer to https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/seed_models.py for detailed instructions per task)
        inputs = [
            f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {i}".format(
                i
            )
            for i in inputs
        ]
    resp = client.embeddings.create(
        model="doubao-embedding-large-text-250515",
        input=inputs,
        encoding_format="float",
    )
    embedding = torch.tensor([d.embedding for d in resp.data], dtype=torch.bfloat16)
    if mrl_dim is not None:
        assert mrl_dim in [2048, 1024, 512, 256]
        embedding = embedding[:, :mrl_dim]
    # normalize to compute cosine sim
    embedding = torch.nn.functional.normalize(embedding, dim=1, p=2).float().numpy()
    return embedding


# gets API Key from environment variable ARK_API_KEY
client = Ark(
    api_key=os.getenv("ARK_API_KEY"),
)

print("----- embeddings -----")
inputs = ["花椰菜又称菜花、花菜，是一种常见的蔬菜。"]
embedding = encode(client, inputs, is_query=False, mrl_dim=1024)
print(embedding)
