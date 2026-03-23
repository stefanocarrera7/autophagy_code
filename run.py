import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from huggingface_hub import login
from datasets import load_dataset
import autophagy_clean as autophagy

login(token="xxx")

# Modelli
base_models = {
    'llama': "unsloth/Meta-Llama-3.1-8B",
    "qwen_06b" : "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
    "qwen_4b" : "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit",
    "qwen_8b" : "unsloth/Qwen3-8B-Base-unsloth-bnb-4bit",
    "qwen_14b" : "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit"
}

# Funzione filtro
def is_valid_test(example):
    return isinstance(example['test'], str) and len(example['test'].strip()) > 0

# 2. Caricamento dati
real_data = load_dataset("stefanocarrera/autophagy_D_mercury", split='train')
# # TEST
# real_data = real_data.shuffle(seed=42).select(range(5))
# prev_adapter_repo = "stefanocarrera/autophagycode_M_unsloth__Qwen3-14B-Base-unsloth-bnb-4bit_lr0.0001_chunk142_gen8"

print(f"Dataset originale: {len(real_data)} righe")
real_data = real_data.filter(is_valid_test)
print(f"Dataset filtrato: {len(real_data)} righe")

# autofagia
base_model_id = base_models.get('qwen_14b')
autophagy.autophagy(
    base_model_id=base_model_id,
    real_data_train=real_data,
    real_data_test= "he",
    model_type = "qwen",
    g=10,
    n_solutions=1,
    lr=1e-4,
    real_data_strategy="trust",
    start_round=0,                      # se maggiore di 0, deve esserci anche resume_model_id
    resume_model_id=None,
    skip_first_test=False
)