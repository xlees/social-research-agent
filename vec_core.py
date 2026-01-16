#coding: utf-8

import sys,os
import hashlib

import torch

if not hasattr(torch, "xpu"):
    class _FakeXPU:
        @staticmethod
        def is_available():
            return False

    torch.xpu = _FakeXPU()

from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from docling.datamodel.pipeline_options import PdfPipelineOptions
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True

from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


VEC_DB_PATH = "./rag.db"
COLLECTION_NAME = "whz_latin"



def file_md5(fpath: str, chunk_size: int = 8192) -> str:
    hasher = hashlib.md5()

    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
            
    return hasher.hexdigest()


def chunk_file(fpath: str) -> list:
    """将指定文件按照语义分成块

    Args:
        fpath (str): 待分块的文件路径

    Returns:
        list: 分块后的文本列表
    """
    MAX_TOKENS = 512
    
    doc_converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.MD,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
               pipeline_cls=StandardPdfPipeline, 
               backend=PyPdfiumDocumentBackend,
               pipeline_options=pipeline_options
            ),
            # InputFormat.DOCX: WordFormatOption(
            #    pipeline_cls=SimplePipeline, 
            #    backend=MsWordDocumentBackend
            # ),           
        },
    )

    doc = doc_converter.convert(fpath).document
    print(f"\nDocument '{fpath}' format converted successfully by docling.\n")

    chunk_list = []
    tokenizer = AutoTokenizer.from_pretrained("./emb_models/thenlper/gte-large")
    chunker = HybridChunker(
        tokenizer=tokenizer,    # can also just pass model name instead of tokenizer instance
        max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer`
        merge_peers=True,       # optional, defaults to True
        overlap_tokens=64,
    )
    for idx, chunk in enumerate(chunker.chunk(dl_doc=doc)):
        chunk_text = chunker.contextualize(chunk=chunk)
        chunk_list.append(chunk_text)

    print(f"Document '{fpath}' chunked into {len(chunk_list)} chunks.\n")

    return chunk_list

def chunks_into_vecdb(fpath: str, emb_model: SentenceTransformer, drop: bool=False) -> None:
    """对指定文件进行分块并插入到向量数据库中

    Args:
        fpath (str): 待分块和索引的文件路径
        emb_model (SentenceTransformer): 用于生成嵌入向量的模型
    """
    chunks: list = chunk_file(fpath)

    print(f"\nGenerating embeddings for chunks of file '{fpath}'...")
    chunk_embeddings = emb_model.encode(chunks, convert_to_numpy=True)

    # 创建向量数据
    vec_data = []
    md5_str = file_md5(fpath)
    for i, chunk in enumerate(chunks):
        vec_data.append({
            "vector": chunk_embeddings[i],
            "text": chunk,
            "fname": os.path.basename(fpath),
            "md5": md5_str,
            "offset": i
        })

    # 插入向量到Milvus
    vec_client = MilvusClient(VEC_DB_PATH)
    all_collections = vec_client.list_collections()
    # print("Existing collections in Milvus:", all_collections)

    if drop and COLLECTION_NAME in all_collections:
        vec_client.drop_collection(
            collection_name=COLLECTION_NAME
        )

    dim = chunk_embeddings.shape[1]
    if COLLECTION_NAME not in all_collections:
        vec_client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=dim,
            metric_type='COSINE',
            auto_id=True
        )


    print(f"\nInserting chunks of file '{fpath}' into Milvus...")
    res = vec_client.insert(
        collection_name=COLLECTION_NAME,
        data=vec_data
    )
    print(f"all chunks of doc '{fpath}' has been inserted into Milvus collection '{COLLECTION_NAME}'.")

    vec_client.close()

    return


def insert_all_docs(doc_dir: str, drop: bool=False) -> None:
    emb_model = SentenceTransformer(model_name_or_path="./emb_models/thenlper/gte-large", device="cpu" )

    if not os.path.exists(doc_dir):
        print(f"directory '{doc_dir}' does not exist!")
        return

    # 遍历目录下的所有文件，进行分块和插入
    for root, dirs, files in os.walk(doc_dir):
        print(f"Processing directory: {root}", dirs, files)

        for fname in files:
            print(f"\n===== Processing file: {fname} =====")

            fpath = os.path.join(root, fname)
            chunks_into_vecdb(fpath, emb_model, drop=drop)

def create_collection_index():
    print("\nCreating index on Milvus collection...")

    from pymilvus import connections, Collection
    
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector", # Name of the vector field to be indexed
        index_type="HNSW", # Type of the index to create
        index_name="vector_index", # Name of the index to create
        metric_type="L2", # Metric type used to measure similarity
        params= {
            "M": 64, # Maximum number of neighbors each node can connect to in the graph
            "efConstruction": 100 # Number of candidate neighbors considered for connection during index construction
        }
    )

    connections.connect(
        alias="default",
        uri=VEC_DB_PATH
    )
     
    collection = Collection(name=COLLECTION_NAME,using='default')
   
    collection.create_index(
        field_name="vector",
        index_params=index_params
    )
    
    print("Index parameters prepared.")


if __name__ == "__main__":

    # emb_model = SentenceTransformer("./emb_models/thenlper/gte-large",device="cpu" )
    # chunks_into_vecdb("model_q.docx", emb_model)

    if len(sys.argv) > 1:
        doc_dir = sys.argv[1]
    else:
        doc_dir = "docs"

    # insert_all_docs(doc_dir = doc_dir, drop=False)

    create_collection_index()