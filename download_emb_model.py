from transformers import AutoModel

# model = AutoModel.from_pretrained(
#     "thenlper/gte-large",
#     cache_dir="./huggingface_cache",  # 指定缓存目录
#     local_files_only=False  # 如果本地没有则下载
# )

# # 保存到指定位置
# model.save_pretrained("./emb_models/thenlper/gte-large")


from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="thenlper/gte-large",
    local_dir="./emb_models/thenlper/gte-large",
    local_dir_use_symlinks=False
)