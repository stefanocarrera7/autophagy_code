from datasets import Dataset, load_dataset
from generate_sample import generate_sample, original_correct_replace, synth_correct_replace
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
    lr: float = 1e-4,
    runs = 1,
    start_round: int = 0,                           # per riprendere da un round specifico in caso di interruzioni
    resume_model_id: str = None,                     # ultimo modello addestrato
    real_data_strategy: str = None,                    # 'correct' rimpiazza le soluzioni errate
    real_data_per_generation: float = None,          # se specificato, indica la percentuale di dati reali da utilizzare per ogni generazione
    ):

    sample = real_data_train
    base_tag = _sanitize_repo_name(base_model_id)
    prev_adapter_repo = resume_model_id
    chunk_size = int(len(sample) / g)     # cambiare per renderlo dinamico in base alla dimensione del dataset e al numero di generazioni, a 138 solo per test, dato che stiamo runnando solo 5 generazioni

    if real_data_test == "he":
        print("\nLoading HumanEval test set...")
        test_data = load_dataset("openai/openai_humaneval", split="test")

    if real_data_test == "evalplus":
        print("\nLoading EvalPlus test set...")
        test_data = load_dataset("stefanocarrera/autophagy_D_evalplus", split="train")
        # # TEST
        # test_data = test_data.select(range(5))

    for t in range(start_round, g):
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
            device_map = {"": 0},
        )
        FastLanguageModel.for_inference(gen_model)

        # --- Calcolo degli indici per il subset del round --- 
        start_idx = t * chunk_size
        end_idx = min((t + 1) * chunk_size, len(sample)) # Impedisce l'Out of Bounds
        
        # Se start_idx supera la lunghezza del dataset, interrompiamo il ciclo
        if start_idx >= len(sample):
            print("\nDati di training esauriti per il prossimo round. Interruzione della pipeline.")
            break
            
        current_subset = sample.select(range(start_idx, end_idx))


        # --- Generazione del dataset sintetico per il test (HumanEval) ---
        print(f"\nGenerating synthetic test set for generation {t+1}...")
        for r in range(1,runs+1):
            test_synth = generate_sample(test_data, gen_model, gen_tok, n_solutions=n_solutions)
            test_data_id = f"stefanocarrera/autophagycode_D_{real_data_test}_{base_tag}_lr{lr}_chunk{chunk_size}_gen{t+1}_test_run{r}"
            test_synth.push_to_hub(test_data_id)

            # Valutazione Metriche
            evaluate_and_push_metrics(test_synth, real_data_test, base_tag, lr, t+1, verbose = False)

        if t == g - 1:
            print("\nUltima generazione completata, Pipeline terminata.")
            break

        # --- Generazione del dataset sintetico per il train ---
        print("\nStarting sample generation...")
        synth = generate_sample(current_subset,
                                gen_model, gen_tok,
                                n_solutions=n_solutions,
                                real_data_strategy=real_data_strategy,
                                real_data_prop=real_data_per_generation)
        
        # --- Correct Replacemet (if chosen) ---
        if real_data_strategy == 'correct':
            synth = original_correct_replace(synth, current_subset, real_data_test, base_tag, lr, gen_round = t+1)

        if real_data_strategy == 'synth_correct':
            synth = synth_correct_replace(synth)


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

        model_id = f"stefanocarrera/autophagycode_M_{base_tag}_lr{lr}_chunk{chunk_size}_strategy_{real_data_strategy}_gen{t+1}"
        data_id  = f"stefanocarrera/autophagycode_D_train_{base_tag}_lr{lr}_chunk{chunk_size}strategy_{real_data_strategy}_gen{t+1}"

        # --- Salvataggio su HF ---
        print("\nPushing to HuggingFace Hub...")
        synth.push_to_hub(data_id)
        ft_model.push_to_hub(model_id)
        ft_tok.push_to_hub(model_id)
        print(f"Pushed model (adapter) to {model_id}")

        # Aggiorniamo i riferimenti per il prossimo giro
        prev_adapter_repo = model_id
        
        # --- 7. PULIZIA DELLA VRAM (POST-TRAINING) EXTREME ---
        print("\nPulizia della VRAM in corso prima del prossimo round di generazione...")
        
        # Sposta esplicitamente il modello sulla CPU per liberare subito la VRAM
        ft_model.cpu()
        
        del ft_model
        del ft_tok
        
        # Doppio giro di garbage collection
        gc.collect()
        gc.collect() 
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        print("Memoria liberata con successo. VRAM attuale allocata:", torch.cuda.memory_allocated() / 1e9, "GB")
    
    print("\n[PIPELINE COMPLETATA]")
    return prev_adapter_repo