from datasets import load_dataset
import json

ds = load_dataset("openai/gsm8k", "main", split="test")
print(len(ds))
print(ds[0]["question"])
print(ds[0]["answer"])

problems = []
for i in range(10):
    q = ds[i]["question"]
    gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
    problems.append({"idx": i, "question": q, "gold": gold})
    print(i, gold)

with open("gsm8k_10.json", "w") as f:
    json.dump(problems, f, indent=2)