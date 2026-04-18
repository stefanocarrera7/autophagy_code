import unsloth
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from huggingface_hub import login
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
        "qwen_14b" : "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit",
        "qwen_4b_instruct" : "unsloth/Qwen3-4B",
        "qwen_8b_instruct" : "unsloth/Qwen3-8B",
    }

    STRATEGY = 'trust'
    
    # # TEST
    # real_data = real_data.shuffle(seed=42).select(range(5))

    # prev_adapter_repo = "stefanocarrera/autophagycode_M_Qwen3-14B_lr0.0001_c142_trust_g8"

    # autofagia
    base_model_id = base_models.get('qwen_8b_instruct')
    autophagy.autophagy(
        base_model_id=base_model_id,
        is_instruct = True,
        real_data_train='mercury',
        real_data_test="he",
        model_type="qwen",
        g=10,
        n_solutions=1,
        lr=1e-4,
        real_data_strategy=STRATEGY,
        start_round=0,                      
        resume_model_id=None,
        skip_first_test=True
    )