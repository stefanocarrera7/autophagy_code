import unsloth
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from huggingface_hub import login
import autophagy_clean as autophagy 
from transformers import set_seed


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

    run_configs = [
        # {"run_id": "run0", "seed": 42},
        {"run_id": "run1", "seed": 123},
        {"run_id": "run2", "seed": 999}
    ]

    STRATEGY = 'trust' # 'correct', 'trust', 'text', 'scm', 'sem'

    # prev_adapter_repo = "stefanocarrera/autophagycode_M_Qwen3-14B_lr0.0001_c142_trust_g8"

    for config in run_configs:
        current_run = config["run_id"]
        current_seed = config["seed"]

        set_seed(current_seed)

        print(f"\n{'='*50}")
        print(f"INIZIO ESECUZIONE: {current_run} (Seed: {current_seed})")
        print(f"{'='*50}\n")

        # autofagia
        base_model_id = base_models.get('qwen_06b')
        autophagy.autophagy(
            base_model_id=base_model_id,
            is_instruct = False,
            real_data_train='mercury',
            real_data_test="he",
            model_type="qwen",
            g=10,
            n_solutions=1,
            lr=1e-4,
            real_data_strategy=STRATEGY,
            start_round=0,                      
            resume_model_id=None,
            skip_first_test=False,
            temperature=0.2,
            top_p=0.99,
            save_token_log=True,
            run_id=current_run,
            seed=current_seed
        )