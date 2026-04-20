from gen import generate_solutions
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset

MODEL = "unsloth/Qwen3-8B-Base-unsloth-bnb-4bit"

model, tok = FastLanguageModel.from_pretrained(
    model_name = MODEL,
    max_seq_length = 1024, # Alzato per sicurezza sui prompt lunghi
    dtype = None,          # Unsloth rileva automaticamente float16/bfloat16
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Carichiamo il dataset
ds = load_dataset("openai/openai_humaneval", split="test")

random_index = 123

test_prompt = ds[random_index]['prompt']
test_entry = ds[random_index]['entry_point']

# Generazione
soluzioni = generate_solutions(test_prompt, model, tok, max_new_tokens=300, do_sample=True, n_solutions=1, temperature=0.2)

print(f"Prompt:\n{test_prompt}\n")
print(f"Soluzioni generate:\n{soluzioni[0]}\n")

soluzioni = generate_solutions(test_prompt, model, tok, max_new_tokens=300, do_sample=True, n_solutions=1, temperature=0.15)

print(f"Soluzioni generate:\n{soluzioni[0]}\n")