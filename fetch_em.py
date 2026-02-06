from modelscope import snapshot_download

model_dir = snapshot_download(
    model_id='BAAI/bge-large-zh-v1.5',
    cache_dir='请在这里设置bge-large-zh-v1.5模型要保存的路径'
)
