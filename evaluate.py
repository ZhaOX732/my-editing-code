import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_output(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./edited_gpt2_rome"
    
    print(f"--- [Task 4] 正在加载编辑后的模型进行评估 ---")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    model.eval()

    data = load_data("data.json")
    
    print("\n" + "="*60)
    print(f"{'主语':<15} | {'ES (可靠性)':<10} | {'PS (泛化性)':<10} | {'NS (局部性)':<10}")
    print("-" * 60)

    for item in data:
        subject = item["subject"]
        target_new = item["target_new"].strip()
        ground_truth = item["ground_truth"].strip()
        
        # 1. Reliability (ES) - 测试原始 Prompt
        res_es = generate_output(model, tokenizer, item["prompt"], device)
        es_score = "通过" if target_new.lower() in res_es.lower() else "失败"
        
        # 2. Generalization (PS) - 测试重写后的 Prompt
        res_ps = generate_output(model, tokenizer, item["rephrase_prompt"], device)
        ps_score = "通过" if target_new.lower() in res_ps.lower() else "失败"
        
        # 3. Locality (NS) - 测试无关事实（应保持原始答案）
        res_ns = generate_output(model, tokenizer, item["locality_prompt"], device)
        ns_score = "通过" if ground_truth.lower() in res_ns.lower() else "失败"

        print(f"{subject:<15} | {es_score:<10} | {ps_score:<10} | {ns_score:<10}")

    print("="*60)
    print("评估完成。如果 PS 和 NS 均为 '通过'，说明 ROME 编辑具有良好的局部性和泛化性。")

if __name__ == "__main__":
    main()