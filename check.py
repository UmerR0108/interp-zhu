import torch
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

def load(i):
    d = torch.load(f"trace_{i}.pt")
    return d["out"][0, d["prompt_len"]:].tolist(), d["gold"]

gen, gold = load(2)
comma = f"{int(gold):,}"
print("2: gold", gold, "| comma version", comma, "first at char", tok.decode(gen).find(comma))

gen, gold = load(4)
print("4: gold", gold, "| text around token 68:", tok.decode(gen[50:90]))

#gen, gold = load(7)
#print("7: gold", gold, "| end of trace:", tok.decode(gen)[-400:])