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
ds = load_dataset("stefanocarrera/autophagy_D_mercury")

test_prompt = ds['train']['prompt'][0]
test_entry = ds['train']['entry_point'][0]

# Generazione
soluzioni = generate_solutions(test_prompt, test_entry, model, tok, n_solutions=1)
print(soluzioni[0])