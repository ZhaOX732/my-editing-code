import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_baseline():
    model_dir = "/root/autodl-tmp/my-editing-code/modelscope/models/Qwen/Qwen2.5-0.5B-Instruct"

    # 3. 加载 Tokenizer 和 Model
    # 使用本地路径加载，并设置 trust_remote_code=True 确保 Qwen 模型架构正确加载
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )

    # 4. 读取实验数据
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n--- Task 1: Baseline Evaluation ---")
    for i, item in enumerate(data):
        prompt = item["prompt"]
        # 确保输入数据在模型所在的设备上
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成回答
        outputs = model.generate(**inputs, max_new_tokens=20)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Fact {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"Ground Truth: {item['ground_truth']}")
        print(f"Target New (Expected after edit): {item['target_new']}")
        print(f"Model Output: {response}\n")

if __name__ == "__main__":
    run_baseline()