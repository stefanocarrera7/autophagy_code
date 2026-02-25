from datasets import Dataset
from generate_sample import generate_sample
from train_unsloth import finetune_model
from huggingface_hub import HfApi
import torch
import gc
from unsloth import FastLanguageModel

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

    sample = real_data_train
    base_tag = _sanitize_repo_name(base_model_id)
    prev_adapter_repo = None
    chunk_size = 138

    for t in range(g):
        print(f"\n{'='*40}")
        print(f"   === Generation round {t+1}/{g} ===")
        print(f"{'='*40}")

        # --- 1. Caricamento Modello per la Generazione ---
        # Se siamo al primo giro usiamo il base model, altrimenti l'ultimo adapter addestrato
        current_model_id = prev_adapter_repo if prev_adapter_repo else base_model_id
        
        print(f"Loading model for generation via Unsloth: {current_model_id}")
        gen_model, gen_tok = FastLanguageModel.from_pretrained(
            model_name = current_model_id,
            max_seq_length = 2048, # Assicurati che corrisponda alla lunghezza usata in train_unsloth
            dtype = torch.float16,
            load_in_4bit = True,
        )
        # Abilita l'inferenza nativa ottimizzata di Unsloth
        FastLanguageModel.for_inference(gen_model)

        # 2. Calcolo degli indici per il subset di questo round
        start_idx = t * chunk_size
        end_idx = (t + 1) * chunk_size
        
        # 3. Estrazione del subset
        current_subset = sample.select(range(start_idx, end_idx))
        
        print("\nStarting sample generation...")
        # test_synth = generate_sample(he, gen_model, gen_tok, n_solutions=n_solutions)
        # he_data_id = f"stefanocarrera/autophagycode_D_HE_{base_tag}_lr{lr}_chunk{chunk_size}_gen{t+1}"
        # test_synth.push_to_hub(f"stefanocarrera/autophagycode_D_{base_tag}_lr{lr}_gen{t+1}_test")
        synth = generate_sample(current_subset, gen_model, gen_tok, n_solutions=n_solutions)

        # --- 4. PULIZIA DELLA VRAM (PRE-TRAINING) ---
        print("\nPulizia della VRAM in corso prima del finetuning...")
        del gen_model
        del gen_tok
        gc.collect()
        torch.cuda.empty_cache()

        # --- 5. Finetuning ---
        print("\nStarting finetuning...")
        ft_dir = f"runs/gen_{t:02d}/adapters"
        
        ft_model, ft_tok = finetune_model(
            dataset = synth,
            base_model_id = base_model_id,
            output_dir = ft_dir,
            model_type = model_type,
            num_train_epochs = 2,
            lr = lr,
            batch_size = 1,
            grad_accum = 16,
            resume_adapter_repo = prev_adapter_repo 
        )

        print("\nEnd Finetuning...")

        model_id = f"stefanocarrera/autophagycode_M_{base_tag}_lr{lr}_gen{t+1}"
        data_id  = f"stefanocarrera/autophagycode_D_{base_tag}_lr{lr}_gen{t+1}"

        # --- 6. Salvataggio su Hugging Face ---
        print("\nPushing to HuggingFace Hub...")
        synth.push_to_hub(data_id)
        
        # Salviamo l'adapter addestrato
        ft_model.push_to_hub(model_id)
        ft_tok.push_to_hub(model_id)
        print(f"Pushed model (adapter) to {model_id}")

        # Aggiorniamo i riferimenti per il prossimo giro
        prev_adapter_repo = model_id
        
        # --- 7. PULIZIA DELLA VRAM (POST-TRAINING) ---
        print("\nPulizia della VRAM in corso prima del prossimo round di generazione...")
        del ft_model
        del ft_tok
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\nProcesso di Autophagy completato con successo!")
    
    # Restituiamo l'ID dell'ultimo modello addestrato anziché il modello stesso,
    return prev_adapter_repo