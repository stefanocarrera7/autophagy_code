from datasets import Dataset, load_dataset
from generate_sample import generate_sample
from train_unsloth import finetune_model
from evaluate_metrics import evaluate_and_push_metrics
from huggingface_hub import HfApi
import torch
import gc
from unsloth import FastLanguageModel

def _sanitize_repo_name(text: str) -> str:
    """Sanitizza il nome del repository sostituendo caratteri non validi con underscore."""
    return text.replace("/", "__").replace(" ", "_")

def autophagy(
    base_model_id: str,
    real_data_train: Dataset,
    real_data_test: str = "he",
    model_type: str = "llama", # 'llama' o 'qwen'
    g: int = 10,
    n_solutions: int = 1,
    lr: float = 1e-4
    ):

    sample = real_data_train
    base_tag = _sanitize_repo_name(base_model_id)
    prev_adapter_repo = None
    chunk_size = 138

    if real_data_test == "he":
        print("\nLoading HumanEval test set...")
        test_data = load_dataset("stefanocarrera/autophagy_D_evalplus")

    for t in range(g):
        print(f"\n{'='*40}")
        print(f"   === Generation round {t+1}/{g} ===")
        print(f"{'='*40}")

        # --- Caricamento Modello ---
        current_model_id = prev_adapter_repo if prev_adapter_repo else base_model_id
        
        print(f"Loading model for generation via Unsloth: {current_model_id}")
        gen_model, gen_tok = FastLanguageModel.from_pretrained(
            model_name = current_model_id,
            max_seq_length = 1024,
            dtype = torch.float16,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(gen_model)

        # --- Calcolo sicuro degli indici per il subset del round --- 
        start_idx = t * chunk_size
        end_idx = min((t + 1) * chunk_size, len(sample)) # Impedisce l'Out of Bounds
        
        # Se start_idx supera la lunghezza del dataset, interrompiamo il ciclo
        if start_idx >= len(sample):
            print("\nDati di training esauriti per il prossimo round. Interruzione della pipeline.")
            break
            
        current_subset = sample.select(range(start_idx, end_idx))

        # --- Generazione del dataset sintetico per il test (HumanEval) ---
        print(f"\nGenerating synthetic test set for generation {t+1}...")
        test_synth = generate_sample(test_data, gen_model, gen_tok, n_solutions=n_solutions)
        he_data_id = f"stefanocarrera/autophagycode_D_{real_data_test}_{base_tag}_lr{lr}_chunk{chunk_size}_gen{t+1}_test"
        test_synth.push_to_hub(he_data_id)

        # --- Valutazione delle metriche sul test ---
        evaluate_and_push_metrics(test_synth, real_data_test, base_tag, lr, t+1, verbose = False)

        print("\nStarting sample generation...")
        synth = generate_sample(current_subset, gen_model, gen_tok, n_solutions=n_solutions)

        # --- PULIZIA DELLA VRAM (PRE-TRAINING) ---
        print("\nPulizia della VRAM in corso prima del finetuning...")
        del gen_model
        del gen_tok
        gc.collect()
        torch.cuda.empty_cache()

        # --- Finetuning ---
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

        # --- Salvataggio su Hugging Face ---
        print("\nPushing to HuggingFace Hub...")
        synth.push_to_hub(data_id)
        ft_model.push_to_hub(model_id)
        ft_tok.push_to_hub(model_id)
        print(f"Pushed model (adapter) to {model_id}")

        # Aggiorniamo i riferimenti per il prossimo giro
        prev_adapter_repo = model_id
        
        # --- PULIZIA DELLA RAM E DELLA VRAM (POST-TRAINING) ---
        print("\nPulizia totale della memoria prima del prossimo round...")
        del ft_model
        del ft_tok
        del test_synth
        del synth
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\n[PIPELINE COMPLETATA]")
    return prev_adapter_repo