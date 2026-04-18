from generate_sample import generate_sample
import torch
import gc
from datasets import load_dataset
from unsloth import FastLanguageModel
from huggingface_hub import login

# 1. Autenticazione (Usa un NUOVO token con permessi di WRITE)
login("xxx") 

# 2. Carichiamo HumanEval una volta sola
print("Caricamento dataset base (HumanEval)...")
base_data = load_dataset("openai/openai_humaneval", split="test")

num_generations = 1
n_sol_per_prompt = 1
max_seq_length = 750
test_data_id = "he"

B = 8
MODEL = f"Qwen3-{B}B"
real_data_strategy = 'trust'

for g in range(num_generations):
    print(f"\n{'='*50}")
    print(f"   AVVIO GENERAZIONE E VALUTAZIONE [{g}]")
    print(f"{'='*50}")

    # --- LOGICA DI SELEZIONE DEL MODELLO ---
    if g == 0:
        # Generazione 0: Modello originale
        model_repo = f"unsloth/Qwen3-{B}B-Base-unsloth-bnb-4bit"
    else:
        model_repo = f"stefanocarrera/autophagycode_M_Qwen3-{B}B_lr0.0001_c200_trust_g{g}"

    dataset_repo = f"stefanocarrera/autophagycode_D_he_{MODEL}_strategy_trust_g{g+1}"

    print(f"Scaricamento e caricamento modello tramite Unsloth: {model_repo}")
    
    # 3. Caricamento del modello e tokenizer con Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_repo,
        max_seq_length = 750,
        dtype = torch.float16,
        load_in_4bit = True,
        device_map = {"": 0},
    )
    FastLanguageModel.for_inference(model)

    # 4. Generazione del dataset
    print(f"Generazione delle soluzioni in corso per la gen {g}...")
    gen_dataset = generate_sample(
        data=base_data, 
        model=model, 
        tokenizer=tokenizer, 
        n_solutions=n_sol_per_prompt,
        real_data_strategy=real_data_strategy
    )

    # 5. Push diretto del dataset generato sul tuo Hugging Face Hub
    print(f"Caricamento del dataset su Hugging Face: {dataset_repo}")
    gen_dataset.push_to_hub(dataset_repo, private=False)
    
    print(f"-> Generazione {g} completata con successo!")

    # 6. PULIZIA DELLA MEMORIA (Critico per evitare CUDA Out of Memory)
    print("Pulizia della VRAM per il prossimo modello...")
    
    # 1. Sposta esplicitamente il modello sulla CPU per svuotare brutalmente la VRAM
    model.cpu()
    
    # 2. Elimina i riferimenti Python
    del model
    del tokenizer
    
    # 3. Doppio giro di garbage collection per intercettare riferimenti circolari ostinati
    gc.collect()
    gc.collect()
    
    # 4. Svuota la cache interna di PyTorch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # Pulisce la memoria condivisa tra processi

print("\nPipeline di autofagia completata per tutte le generazioni!")