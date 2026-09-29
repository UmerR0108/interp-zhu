import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
#converts tokens to and from token IDs
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to("mps")
#loads teh 2 bytes per weight model onto the mac's GPU

messages = [{"role": "user", "content": "What is 17 x 24?"}]
# the question that sets the role as the user asking a question
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
# wraps the question in chat markets so we can see where the assistant begins
ids = tok(text, return_tensors="pt").input_ids.to("mps")
# converts the text into token IDs and moves them into the gpu

torch.manual_seed(0)
# makes the randomness repeatable by selecting the same prob
out = model.generate(ids, max_new_tokens=2000)
# produces the prompt and the 2k token limit
torch.save(out.cpu(), "trace_17x24.pt")
# copies the result and saves it to the disk

gen = out[0, ids.shape[1]:]
end_think = tok.convert_tokens_to_ids("</think>")
pos = (gen == end_think).nonzero()
print(gen.shape,pos)
