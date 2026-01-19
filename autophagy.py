from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_sample import generate_sample
# Importa la nuova funzione dal file modificato (assumendo l'abbia chiamato train_accelerate.py)
from train_unsloth import finetune_model_unsloth
from huggingface_hub import HfApi
import torch

def _sanitize_repo_name(text: str) -> str:
    return text.replace("/", "__").replace(" ", "_")

def autophagy(
    base_model_id: str,
    real_data_train: Dataset,
    real_data_test: Dataset,
    g: int = 10,
    n_solutions: int = 1,
    data_format: str = "he",
    pass_at_k: int = 1
    ):

    # 0) Starting model - Standard HF Loading
    print(f"Loading base model: {base_model_id}")
    gen_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16, # o bfloat16
        trust_remote_code=True
    )
    gen_tok = AutoTokenizer.from_pretrained(base_model_id)

    sample = real_data_train
    base_tag = _sanitize_repo_name(base_model_id)
    prev_adapter_repo = None

    for t in range(g):
        print(f"=== Generation round {t+1}/{g} ===")
        print("\nStarting sample generation...")
        
        # NOTA: Assicurati che generate_sample accetti un modello HF standard 
        # (La tua funzione gen.py usa model.generate(), quindi è già compatibile!)
        synth = generate_sample(sample, gen_model, gen_tok, n_solutions=n_solutions)

        print("\nStarting finetuning...")
        ft_dir = f"runs/gen_{t:02d}/adapters"
        
        # Chiamata alla nuova funzione di training
        ft_model, ft_tok = finetune_model_unsloth(
            dataset = synth,
            base_model_id = base_model_id,
            output_dir = ft_dir,
            num_train_epochs = 2,
            lr = 2e-4,
            batch_size = 1,
            grad_accum = 16,
            resume_adapter_repo=prev_adapter_repo # Passiamo l'adapter precedente per continuare il training
        )

        print("\nEnd Finetuning...")

        model_id = f"stefanocarrera/autophagycode_M_{base_tag}_gen{t+1}"
        data_id  = f"stefanocarrera/autophagycode_D_{base_tag}_gen{t+1}"

        print("\nPushing to HuggingFace Hub...")
        api = HfApi()
        synth.push_to_hub(data_id)
        
        # Con PEFT, ft_model è un PeftModel, quindi save_pretrained/push_to_hub salva solo l'adapter
        ft_model.push_to_hub(model_id)
        ft_tok.push_to_hub(model_id)
        print(f"Pushed model (adapter) to {model_id}")

        # Aggiorniamo i riferimenti per il prossimo giro
        prev_adapter_repo = model_id
        
        # Per il prossimo giro di generazione, dobbiamo assicurarci di usare il modello base + il nuovo adapter.
        # ft_model è già (Base + Adapter), quindi possiamo usarlo direttamente per la generazione.
        gen_model, gen_tok = ft_model, ft_tok
    

    return gen_model, gen_tok