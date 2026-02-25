from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_sample import generate_sample
from train_unsloth import finetune_model
from huggingface_hub import HfApi
import torch

def _sanitize_repo_name(text: str) -> str:
    return text.replace("/", "__").replace(" ", "_")

def autophagy(
    base_model_id: str,
    real_data_train: Dataset,
    model_type: str = "llama", # 'llama' o 'qwen'
    g: int = 10,
    n_solutions: int = 1,
    lr: float = 1e-4
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
    chunk_size = 138

    for t in range(g):
        print(f"=== Generation round {t+1}/{g} ===")

        # 1. Calcolo degli indici per il subset di questo round
        start_idx = t * chunk_size
        # Se è l'ultimo giro, prendiamo tutto quello che resta per non perdere dati
        end_idx = (t + 1) * chunk_size
        
        # 2. Estrazione del subset
        current_subset = sample.select(range(start_idx, end_idx))
        
        print("\nStarting sample generation...")
        
        synth = generate_sample(current_subset, gen_model, gen_tok, n_solutions=n_solutions)

        print("\nStarting finetuning...")
        ft_dir = f"runs/gen_{t:02d}/adapters"
        
        # Chiamata alla nuova funzione di training
        ft_model, ft_tok = finetune_model(
            dataset = synth,
            base_model_id = base_model_id,
            output_dir = ft_dir,
            model_type=model_type,  # 'llama' o 'qwen,  # Specifica il tipo di modello per il templete del prompt
            num_train_epochs = 2,
            lr = lr,
            batch_size = 1,
            grad_accum = 16,
            resume_adapter_repo=prev_adapter_repo # Passiamo l'adapter precedente per continuare il training
        )

        print("\nEnd Finetuning...")

        model_id = f"stefanocarrera/autophagycode_M_{base_tag}_lr{lr}_gen{t+1}"
        data_id  = f"stefanocarrera/autophagycode_D_{base_tag}_lr{lr}_gen{t+1}"

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