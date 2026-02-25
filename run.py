from huggingface_hub import login
from datasets import load_dataset
import autophagy

login(token="xxxx")

# Modelli
base_models = {
    'llama': "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    'qwen' : 'unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit',
    "qwen_06b" : "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
    "qwen_4b" : "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit",
    "qwen_8b" : "unsloth/Qwen3-8B-Base-unsloth-bnb-4bit",
}

# Funzione filtro
def is_valid_test(example):
    return isinstance(example['test'], str) and len(example['test'].strip()) > 0

# 2. Caricamento dati
real_data = load_dataset("stefanocarrera/D_mercury_to_he", split='train')

print(f"Dataset originale: {len(real_data)} righe")
real_data = real_data.filter(is_valid_test)
print(f"Dataset filtrato: {len(real_data)} righe")

# autofagia
base_model_id = base_models.get('qwen_8b')
autophagy.autophagy(
    base_model_id=base_model_id,
    real_data_train=real_data,
    model_type = 'qwen',
    g=5,        
    n_solutions=1,
    lr=1e-4
)