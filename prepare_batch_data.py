import json

def main():
    # 假设你的原始数据文件叫 counterfact.json
    with open("counterfact.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    batch_data = []
    # 提取前 500 条用于批量编辑
    for item in raw_data[:500]:
        subject = item["requested_rewrite"]["subject"]
        # 将提示词中的 {} 替换为真实的主语
        prompt = item["requested_rewrite"]["prompt"].replace("{}", subject)
        target_new = item["requested_rewrite"]["target_new"]["str"]
        ground_truth = item["requested_rewrite"]["target_true"]["str"]
        
        # 组装成 edit_memit.py 需要的扁平格式
        formatted_item = {
            "prompt": prompt,
            "subject": subject,
            "target_new": target_new,
            "ground_truth": ground_truth
        }
        batch_data.append(formatted_item)

    # 保存为 batch_data.json
    with open("batch_data.json", "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)

    print("数据格式化完成！已生成 batch_data.json")

if __name__ == "__main__":
    main()