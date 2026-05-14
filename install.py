import os
os.environ['MODELSCOPE_CACHE'] = './modelscope'
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('openai-community/gpt2-xl')