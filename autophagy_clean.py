from datasets import Dataset, load_dataset
from generate_sample import generate_sample, original_correct_replace, synth_correct_replace
from train_unsloth import finetune_model
from evaluate_metrics import evaluate_and_push_metrics
from huggingface_hub import HfApi
import os
import shutil
import torch
import gc
from unsloth import FastLanguageModel

def _sanitize_repo_name(text: str) -> str:
    """Sanitizza e accorcia il nome del repository per rispettare il limite di 96 char di HF."""
    # 1. Rimuove l'organizzazione (es. 'unsloth/')
    if "/" in text:
        text = text.split("/")[-1]
    
    # 2. Rimuove suffissi lunghi standard
    text = text.replace("-unsloth-bnb-4bit", "")
    text = text.replace("-bnb-4bit", "")
    text = text.replace("-Base", "")
    text = text.replace("-Instruct", "")
    
    return text.replace(" ", "_")

def autophagy(
    base_model_id: str,
    is_instruct: bool = False,
    real_data_train: str = 'mercury',    # 'taco', 'mercury', 'all_train'
    real_data_test: str = "he",    # 'all_test'
    model_type: str = "llama", # 'llama' o 'qwen'
    g: int = 10,
    n_solutions: int = 1,
    lr: float = 1e-5,
    start_round: int = 0,                           # per riprendere da un round specifico in caso di interruzioni
    resume_model_id: str = None,                     # ultimo modello addestrato
    real_data_strategy: str = 'trust',              # 'correct' rimpiazza le soluzioni errate, 'trust' rimpiazza qualsiasi sia la soluzione, 'text' fa il ft con dati testuali
    skip_first_test = False,
    temperature: float = 1,
    top_p: float = 0.95,
    save_token_log: bool = False
    ):

    base_tag = _sanitize_repo_name(base_model_id)
    if is_instruct:
        base_tag += "-instr"
        
    prev_adapter_repo = resume_model_id

    if real_data_strategy == 'sc':
        n_sol = n_solutions
    else:
        n_sol = 1

    # ===== SCARICARE I DATI  ======
    # --- TEST ---
    if real_data_test == "he":
        print("\nLoading HumanEval test set...")
        test_data = load_dataset("openai/openai_humaneval", split="test")

    if real_data_test == "evalplus":
        print("\nLoading EvalPlus test set...")
        test_data = load_dataset("stefanocarrera/autophagy_D_evalplus", split="train")

    if (real_data_test == "all_test" and real_data_train != "all_train") or (real_data_test != "all_test" and real_data_train == "all_train"):
        print("\nERROR: Inconsistent data selection. If you choose 'all_test' for real_data_test, you must also choose 'all_train' for real_data_train, and vice versa.")
        return

    if real_data_test == "all_test":
        print("\nLoading Merged All test set...")
        test_data = load_dataset("stefanocarrera/autophagy_D_all", split="test")

    # --- TRAIN ---
    if real_data_train == "mercury":
        print("\nLoading Mercury training set...")
        sample = load_dataset("stefanocarrera/autophagy_D_mercury", split="train")

    if real_data_train == "taco":
        print("\nLoading Taco training set...")
        sample = load_dataset("stefanocarrera/autophagy_D_taco", split="train")

    if real_data_train == "all_train":
        print("\nLoading Merged All training set...")
        sample = load_dataset("stefanocarrera/autophagy_D_all", split="train")
    
    chunk_size = int(len(sample) / g)

    

    # ======= MAIN LOOP ========
    for t in range(start_round, g):
        print(f"\n{'='*40}")
        print(f"   === Generation round {t+1}/{g} ===")
        print(f"{'='*40}")

        # --- Caricamento Modello ---
        current_model_id = prev_adapter_repo if prev_adapter_repo else base_model_id
        
        print(f"Loading model for generation via Unsloth: {current_model_id}")
        gen_model, gen_tok = FastLanguageModel.from_pretrained(
            model_name = current_model_id,
            max_seq_length = 2048,
            dtype = torch.float16,
            load_in_4bit = True,
            device_map = {"": 0},
        )              
        FastLanguageModel.for_inference(gen_model)


        # --- Generazione del dataset per il test (HumanEval) ---
        if skip_first_test == False or (skip_first_test == True and t != start_round):
            
            print(f"\nGenerating synthetic test set for generation {t+1}...")
            test_synth = generate_sample(test_data, gen_model, gen_tok,
                                         n_solutions=n_solutions,
                                         real_data_strategy='trust',
                                         is_instruct=is_instruct, model_type=model_type,
                                         temperature=temperature, top_p=top_p,
                                         save_token_log=save_token_log)

            test_data_id = f"stefanocarrera/autophagycode_D_{real_data_test}_train-{real_data_train}_{base_tag}_strategy_{real_data_strategy}_t{temperature}_g{t+1}"
            test_synth.push_to_hub(test_data_id)

            # --- Valutazione Metriche ----   # da vedere per il text
            evaluate_and_push_metrics(test_synth, real_data_test, tokenizer=gen_tok, synth_repo = test_data_id, verbose = False)

        if t == g - 1:
            print("\nUltima generazione completata, Pipeline terminata.")
            break


        # --- Calcolo degli indici per il subset del round --- 
        start_idx = t * chunk_size
        end_idx = min((t + 1) * chunk_size, len(sample)) # Impedisce l'Out of Bounds
        
        # Se start_idx supera la lunghezza del dataset, interrompiamo il ciclo
        if start_idx >= len(sample):
            print("\nDati di training esauriti per il prossimo round. Interruzione della pipeline.")
            break

                
        current_subset = sample.select(range(start_idx, end_idx))

        # --- Generazione del dataset sintetico per il train ---
        print("\nStarting sample generation...")
        synth = generate_sample(current_subset,
                                gen_model, gen_tok,
                                n_solutions=n_sol,
                                real_data_strategy=real_data_strategy,
                                is_instruct=is_instruct, model_type=model_type,
                                temperature=temperature, top_p=top_p,
                                save_token_log=save_token_log)

        # --- Correct Replacemet (if chosen) ---
        if real_data_strategy == 'correct':
            synth = original_correct_replace(synth, current_subset, real_data_test)

        if real_data_strategy == 'sc':
            synth, _ = synth_correct_replace(synth, real_data_test)


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
            is_instruct=is_instruct,
            output_dir = ft_dir,
            model_type = model_type,
            num_train_epochs = 3,
            lr = lr,
            batch_size = 1,
            grad_accum = 16,
            pack_to_max=False,
            resume_adapter_repo = prev_adapter_repo 
        )

        print("\nEnd Finetuning...")

        model_id = f"stefanocarrera/autophagycode_M_{real_data_train}_{base_tag}_lr{lr}_c{chunk_size}_{real_data_strategy}_t{temperature}_g{t+1}"
        data_id  = f"stefanocarrera/autophagycode_D_{real_data_train}_{base_tag}_lr{lr}_c{chunk_size}_{real_data_strategy}_t{temperature}_g{t+1}"

        # --- Salvataggio su HF ---
        print("\nPushing to HuggingFace Hub...")
        synth.push_to_hub(data_id)
        ft_model.push_to_hub(model_id)
        ft_tok.push_to_hub(model_id)
        print(f"Pushed model (adapter) to {model_id}")

        # Aggiorniamo i riferimenti per il prossimo giro
        prev_adapter_repo = model_id

        # Pulizia del disco
        print("\nCancellazione dei file locali dell'adapter per liberare spazio su disco...")
        if os.path.exists(ft_dir):
            shutil.rmtree(ft_dir)
            print(f"Cartella {ft_dir} eliminata con successo.")
        
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