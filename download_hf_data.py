import os
import json
# 强制设置 HuggingFace 使用国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

def main():
    print("正在加载已下载的 CounterFact 数据集...")
    # 从本地缓存加载，这一步会非常快
    dataset = load_dataset("NeelNanda/counterfact-tracing", split="train")
    
    batch_data = []
    # 提取前 500 条数据
    for i in range(500):
        item = dataset[i]
        
        subject = item["subject"]
        prompt = item["prompt"]
        
        # 使用当前HF数据集的正确字段名
        target_new = item["target_false"] 
        ground_truth = item["target_true"]
        
        # 组装成作业要求的格式
        formatted_item = {
            "prompt": prompt,
            "subject": subject,
            "target_new": target_new,
            "ground_truth": ground_truth,
            "rephrase_prompt": f"Can you tell me about {subject}?",
            "locality_prompt": f"What do you know about {subject}?",
            "locality_ground_truth": ground_truth
        }
        batch_data.append(formatted_item)

    with open("batch_data.json", "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)

    print(f"成功提取了 {len(batch_data)} 条数据！已保存为 batch_data.json")

if __name__ == "__main__":
    main()