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

num_generations = 10
n_sol_per_prompt = 3
max_seq_length = 300
test_data_id = "he"

B = 8
MODEL = f"Qwen3-{B}B"
real_data_strategy = 'correct'

for g in range(1,num_generations):
    print(f"\n{'='*50}")
    print(f"   AVVIO GENERAZIONE E VALUTAZIONE [{g}]")
    print(f"{'='*50}")

    # --- LOGICA DI SELEZIONE DEL MODELLO ---
    if g == 0:
        # Generazione 0: Modello originale
        model_repo = f"unsloth/Qwen3-{B}B-Base-unsloth-bnb-4bit"
    else:
        model_repo = f"stefanocarrera/autophagycode_M_{MODEL}_lr1e-05_c142_correct_g{g}"

    dataset_repo = f"stefanocarrera/autophagycode_D_he_{MODEL}_strategy_{real_data_strategy}_g{g+1}"

    print(f"Scaricamento e caricamento modello tramite Unsloth: {model_repo}")
    
    # 3. Caricamento del modello e tokenizer con Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_repo,
        max_seq_length = max_seq_length,
        dtype = torch.float16,
        load_in_4bit = True,
    )
    # Abilita l'inferenza nativa 2x più veloce di Unsloth
    FastLanguageModel.for_inference(model)

    # 4. Generazione del dataset
    print(f"Generazione delle soluzioni in corso per la gen {g}...")
    gen_dataset = generate_sample(
        data=base_data, 
        model=model, 
        tokenizer=tokenizer, 
        n_solutions=n_sol_per_prompt
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