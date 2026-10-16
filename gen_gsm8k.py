import torch
import json, os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to("mps")

with open("gsm8k_10.json") as f:
    problems = json.load(f)

end_think = tok.convert_tokens_to_ids("</think>")

def gen_trace(question):
    messages = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = tok(text, return_tensors="pt").input_ids.to("mps")
    torch.manual_seed(0)
    out = model.generate(ids, max_new_tokens=8000)
    return out.cpu(), ids.shape[1]

for p in problems:
    if os.path.exists(f"trace_{p['idx']}.pt"):
        continue
    start = time.time()
    out, prompt_len = gen_trace(p["question"])
    torch.save({"out": out, "prompt_len": prompt_len, "gold": p["gold"]}, f"trace_{p['idx']}.pt")
    gen = out[0, prompt_len:]
    has_end = (gen == end_think).any().item()
    print(p["idx"], "tokens:", gen.shape[0], "</think>:", has_end, "secs:", round(time.time() - start))