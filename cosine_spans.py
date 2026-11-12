import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to("mps")
end_think = tok.convert_tokens_to_ids("</think>")

path = hf_hub_download("Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50", "layer14.sae.pt")
sae = torch.load(path)
print(sae.keys())
W_enc = sae["W_enc"].float()
W_dec = sae["W_dec"].float()
b_enc = sae["b_enc"].float()
b_dec = sae["b_dec"].float()

def recon_cos(X):
    cos_all = []
    for s in range(0, X.shape[0], 512):
        x = X[s:s + 512]
        pre = x @ W_enc.T + b_enc
        vals, idx = pre.topk(50, dim=-1)
        f = torch.zeros_like(pre)
        f.scatter_(-1, idx, vals)
        x_hat = f @ W_dec.T + b_dec
        cos_all.append(torch.nn.functional.cosine_similarity(x, x_hat, dim=-1))
    return torch.cat(cos_all)

firsts = {0: 248, 1: 164, 2: 449, 3: 346, 5: 714, 6: 442, 8: 2131, 9: 284}

for i, first in firsts.items():
    d = torch.load(f"trace_{i}.pt")
    out = d["out"]
    p = d["prompt_len"]
    t = out[0, p:].tolist().index(end_think)
    with torch.no_grad():
        hs = model.model(out.to("mps"), output_hidden_states=True).hidden_states
        X = hs[15][0].float().cpu()
        del hs
        cos = recon_cos(X)
    spans = [cos[:p], cos[p:p + first], cos[p + first:p + t], cos[p + t:]]
    print(i, "prompt / before / after / answer:", [round(s.mean().item(), 3) for s in spans])