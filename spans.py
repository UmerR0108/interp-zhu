import torch
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
end_think = tok.convert_tokens_to_ids("</think>")

for i in range(10):
    d = torch.load(f"trace_{i}.pt")
    gen = d["out"][0, d["prompt_len"]:].tolist()
    gold = d["gold"]

    if end_think not in gen:
        print(i, "cut off, no </think>")
        continue
    think_end = gen.index(end_think)
    thinking = gen[:think_end]
    answer_text = tok.decode(gen[think_end + 1:])
    correct = gold in answer_text

    first = None
    for v in [gold, f"{int(gold):,}"]:
        v_ids = tok(v, add_special_tokens=False).input_ids
        for j in range(len(thinking) - len(v_ids) + 1):
            if thinking[j:j + len(v_ids)] == v_ids:
                if first is None or j < first:
                    first = j
                break
    

    if first is None:
        print(i, "think_end", think_end, "correct", correct, "gold never appears in thinking")
    else:
        post = round(100 * (think_end - first) / think_end)
        print(i, "think_end", think_end, "first", first, "post-answer %", post, "correct", correct)