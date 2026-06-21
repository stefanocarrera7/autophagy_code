import unsloth
import os
import shutil
import gc
import torch
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from huggingface_hub import login
import autophagy as autophagy 
from transformers import set_seed

HF_TOKEN = os.getenv("token_write")

def is_valid_test(example):
    return isinstance(example['test'], str) and len(example['test'].strip()) > 0

def clean_system_and_cache():
    """Pulisce la cache di HuggingFace su disco, le cartelle temporanee e la VRAM."""
    print("\n[Pulizia] Avvio liberazione disco e memoria GPU...")
    
    # 1. HuggingFace Hub Cache (~/.cache/huggingface/hub)
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(hf_cache_dir):
        shutil.rmtree(hf_cache_dir, ignore_errors=True)
        print(f" - Eliminata directory: {hf_cache_dir}")
    
    # 2. Cartella logs /runs
    runs_dir = "runs"
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir, ignore_errors=True)
        print(f" - Eliminata directory: {runs_dir}")
        
    # 3. Unsloth compiled cache
    unsloth_cache = "unsloth_compiled_cache"
    if os.path.exists(unsloth_cache):
        shutil.rmtree(unsloth_cache, ignore_errors=True)
        print(f" - Eliminata directory: {unsloth_cache}")

    # 4. Svuotamento memoria GPU
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(" - Memoria VRAM di PyTorch svuotata.")
        
    print("[Pulizia] Completata con successo.\n")


if __name__ == '__main__':
    
    login(token=HF_TOKEN)

    # Dizionario dei Modelli
    base_models = {
        "qwen_06b": "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
        # "qwen_4b": "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit",
        # "qwen_8b": "unsloth/Qwen3-8B-Base-unsloth-bnb-4bit",
        # "qwen_14b": "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit"
    }

    # Configurazioni delle Run
    run_configs = [
        {"run_id": "run1", "seed": 123},
        {"run_id": "run2", "seed": 999}
    ]

    temperatures = [#0.2,
                    1
                    ]

    STRATEGY = 'trust'

    # ---------------------------------------------------------
    # CICLI DI ESECUZIONE
    # ---------------------------------------------------------
    for model_key, base_model_id in base_models.items():
        print(f"\n{'='*70}")
        print(f"AVVIO SPERIMENTAZIONE PER IL MODELLO: {model_key} ({base_model_id})")
        print(f"{'='*70}")

        # prev_adapter = "stefanocarrera/autophagycode_M_mercury_Qwen3-8B_lr0.0001_c142_trust_t1_g5_run2"

        for temp in temperatures:
            for config in run_configs:
                current_run = config["run_id"]
                current_seed = config["seed"]

                set_seed(current_seed)

                print(f"\n{'-'*60}")
                print(f"ESECUZIONE: Modello = {model_key} | Temp = {temp} | Run = {current_run} (Seed: {current_seed})")
                print(f"{'-'*60}\n")

                try:
                    autophagy.autophagy(
                        base_model_id=base_model_id,
                        is_instruct=False,
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
                        temperature=temp,
                        top_p=0.99,
                        save_token_log=True,
                        run_id=current_run,
                        seed=current_seed,
                    )
                except Exception as e:
                    print(f"!!! ERRORE CRITICO durante l'esecuzione di {model_key} (Temp: {temp}, Run: {current_run}) !!!")
                    print(f"Dettaglio Errore: {e}")
                
                # A fine funzione, a prescindere dal successo o fallimento, ripuliamo tutto
                finally:
                    clean_system_and_cache()
                    
    print("\n INTERA PIPELINE DI SPERIMENTAZIONE COMPLETATA! ")