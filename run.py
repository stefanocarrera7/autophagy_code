import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from huggingface_hub import login
from datasets import load_dataset
import autophagy_clean as autophagy 

# Funzione filtro (può stare fuori dall'if __name__)
def is_valid_test(example):
    return isinstance(example['test'], str) and len(example['test'].strip()) > 0


if __name__ == '__main__':
    
    login(token="xxx")

    # Modelli
    base_models = {
        "qwen_06b" : "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
        "qwen_4b" : "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit",
        "qwen_8b" : "unsloth/Qwen3-8B-Base-unsloth-bnb-4bit",
        "qwen_14b" : "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit"
    }

    # Caricamento dati
    real_data = load_dataset("stefanocarrera/autophagy_D_mercury", split='train')
    # text_data = load_dataset("stefanocarrera/autophagy_D_text_S", split='train')
    # # TEST
    # real_data = real_data.shuffle(seed=42).select(range(5))
    prev_adapter_repo = "stefanocarrera/autophagycode_M_Qwen3-14B_lr0.0001_c142_trust_g8"

    print(f"Dataset originale: {len(real_data)} righe")
    real_data = real_data.filter(is_valid_test)
    print(f"Dataset filtrato: {len(real_data)} righe")

    # autofagia
    base_model_id = base_models.get('qwen_06b')
    autophagy.autophagy(
        base_model_id=base_model_id,
        real_data_train=real_data,
        real_data_test="he",
        model_type="qwen",
        g=10,
        n_solutions=1,
        lr=1e-4,
        real_data_strategy="scm",
        start_round=0,                      
        resume_model_id=None,
        skip_first_test=False
    )