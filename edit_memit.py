import os
# 强制设置缓存路径
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/root/autodl-tmp/hf_cache"

import json
import time
import torch

# ================= 终极魔法：在所有导入之前劫持 datasets =================
import datasets
_original_load_dataset = datasets.load_dataset

def patched_load_dataset(path, *args, **kwargs):
    # 无论谁调用，只要是找 wikitext，统一转向 Salesforce 镜像
    if path == "wikitext":
        print("\n[拦截成功] 发现失效链接请求，已强制重定向至 Salesforce/wikitext 镜像...\n")
        return _original_load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", *args, **kwargs)
    return _original_load_dataset(path, *args, **kwargs)

# 关键：同时替换 datasets 模块和其内部引用
datasets.load_dataset = patched_load_dataset
import datasets.load
datasets.load.load_dataset = patched_load_dataset
# ======================================================================

from easyeditor import BaseEditor, MEMITHyperParams

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/gpt2-xl.yaml')
    local_model_path = "/root/autodl-tmp/my-editing-code/modelscope/models/openai-community/gpt2-xl"
    
    # 强制覆盖参数
    hparams.model_name = local_model_path
    hparams.mom2_dataset = "wikitext" # 必须叫这个名字，否则 EasyEditor 会报 KeyError
    hparams.mom2_n_samples = 1000
    
    editor = BaseEditor.from_hparams(hparams)
    
    data = load_data("batch_data.json") 
    
    prompts = [d["prompt"] for d in data]
    ground_truth = [d["ground_truth"] for d in data]
    target_new = [d["target_new"] for d in data]
    subject = [d["subject"] for d in data]

    print(f"\n--- 开始 MEMIT 批量知识注入 (共 {len(prompts)} 条) ---")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False
    )
    
    end_time = time.time()
    print(f"\n[任务完成] 批量编辑耗时: {end_time - start_time:.2f} 秒")
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"[资源消耗] 批量编辑峰值显存占用: {peak_mem:.2f} MB")

if __name__ == "__main__":
    main()