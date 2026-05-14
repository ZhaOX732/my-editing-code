import json
import os
from easyeditor import BaseEditor, ROMEHyperParams

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # 1. 加载 ROME 配置文件
    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2-xl.yaml')
    
    # 指向本地模型路径
    local_model_path = "/root/autodl-tmp/my-editing-code/modelscope/models/openai-community/gpt2-xl"
    hparams.model_name = local_model_path
    
    editor = BaseEditor.from_hparams(hparams)
    
    # 2. 加载待编辑数据
    data = load_data("data.json")
    
    prompts = [d["prompt"] for d in data]
    ground_truth = [d["ground_truth"] for d in data]
    target_new = [d["target_new"] for d in data]
    subject = [d["subject"] for d in data]

    print(f"\n--- [Task 2] 开始 ROME 知识注入 ---")
    
    # 关键点：keep_original_weight=False，确保权重被修改并保留
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False 
    )
    
    # 3. 保存编辑后的模型
    save_path = "./edited_gpt2_rome"
    print(f"\n正在保存编辑后的模型至: {save_path}...")
    edited_model.save_pretrained(save_path)
    # 同时保存 tokenizer 以便后续加载
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(local_model_path)
    tokenizer.save_pretrained(save_path)
    
    print("模型保存成功！现在你可以运行 evaluate.py 进行评估了。")

if __name__ == "__main__":
    main()