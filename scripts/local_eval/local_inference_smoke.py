import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None

print("[1] device:", "mps" if torch.backends.mps.is_available() else "cpu")
print("[2] loading Qwen3-0.6B-Base tokenizer...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", token=HF_TOKEN)
print(f"    tokenizer ok ({time.time()-t0:.1f}s)")

print("[3] loading model bf16 (mps)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B-Base",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
).to("mps")
print(f"    model loaded ({time.time()-t0:.1f}s) params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

print("[4] inference smoke test...")
prompt = "달빛알바 호빠 후기:"
inputs = tok(prompt, return_tensors="pt").to("mps")
in_len = inputs["input_ids"].shape[1]
t0 = time.time()
out = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=1.0, top_p=0.9, pad_token_id=tok.eos_token_id)
dt = time.time() - t0
new_tok = out[0].shape[0] - in_len
text = tok.decode(out[0], skip_special_tokens=True)
tok_per_sec = new_tok / dt if dt > 0 else 0
print(f"    ok ({dt:.1f}s, {new_tok} new tokens, {tok_per_sec:.1f} tok/s)")
print(f"    [output] {text[:300]}")
