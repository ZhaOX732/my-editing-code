import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # 替换为你指定的本地绝对路径
    model_path = "/root/autodl-tmp/my-editing-code/modelscope/models/openai-community/gpt2-xl"
    print(f"加载本地模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    
    data = load_data("data.json")
    
    print("\n--- 开始基线测试 (Baseline Evaluation) ---")
    for idx, item in enumerate(data):
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # 生成10个token作为回答观察
        outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"[{idx+1}/10] Prompt: {prompt}")
        print(f"  预期目标 (编辑后): {item['target_new']}")
        print(f"  原始 Ground Truth: {item['ground_truth']}")
        print(f"  模型当前输出: {response[len(prompt):].strip()}")
        print("-" * 40)

if __name__ == "__main__":
    main()