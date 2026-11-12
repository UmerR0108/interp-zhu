import torch
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
end_think = tok.convert_tokens_to_ids("</think>")
words = ["wait", "hmm", "let me check", "double-check", "double check", "let me verify", "actually"]
firsts = {0: 248, 1: 164, 2: 449, 3: 346, 5: 714, 6: 442, 8: 2131, 9: 284}

before_tokens = 0
after_tokens = 0
before_counts = {w: 0 for w in words}
after_counts = {w: 0 for w in words}

for i, first in firsts.items():
    d = torch.load(f"trace_{i}.pt")
    gen = d["out"][0, d["prompt_len"]:].tolist()
    think_end = gen.index(end_think)
    before = tok.decode(gen[:first]).lower()
    after = tok.decode(gen[first:think_end]).lower()
    print(i, "wait", before.count("wait"), "->", after.count("wait"), "| let me check", before.count("let me check"), "->", after.count("let me check"))
    before_tokens += first
    after_tokens += think_end - first
    for w in words:
        before_counts[w] += before.count(w)
        after_counts[w] += after.count(w)

print("tokens before:", before_tokens, "after:", after_tokens)
for w in words:
    b = round(1000 * before_counts[w] / before_tokens, 1)
    a = round(1000 * after_counts[w] / after_tokens, 1)
    print(w, "| before:", before_counts[w], b, "per 1k | after:", after_counts[w], a, "per 1k")