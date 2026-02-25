from generate_sample import generate_sample
import torch
import gc
from datasets import load_dataset
from unsloth import FastLanguageModel # <--- Importazione Unsloth
from huggingface_hub import login

# 1. Autenticazione (Usa un NUOVO token con permessi di WRITE)
login("xxxx") 

# 2. Carichiamo HumanEval una volta sola
print("Caricamento dataset base (HumanEval)...")
base_data = load_dataset("openai_humaneval", split="test")

num_generations = 5
n_sol_per_prompt = 1
max_seq_length = 1024 # Parametro richiesto da Unsloth

HF_USERNAME = "stefanocarrera"
DATASET_BASE_NAME = "autophagycode_D_HE_unsloth__Qwen3-4B-Base-unsloth-bnb-4bit_lr0.0001_chunck138_gen"

for g in range(num_generations + 1):
    print(f"\n{'='*50}")
    print(f"   AVVIO GENERAZIONE E VALUTAZIONE [{g}]")
    print(f"{'='*50}")

    # --- LOGICA DI SELEZIONE DEL MODELLO ---
    if g == 0:
        # Generazione 0: Modello originale di Meta
        model_repo = "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit"
    else:
        model_repo = f"stefanocarrera/autophagycode_M_unsloth__Qwen3-4B-Base-unsloth-bnb-4bit_lr0.0001_gen{g}"

    dataset_repo = f"{HF_USERNAME}/{DATASET_BASE_NAME}{g+1}"

    print(f"Scaricamento e caricamento modello tramite Unsloth: {model_repo}")
    
    # 3. Caricamento del modello e tokenizer con Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_repo,
        max_seq_length = max_seq_length,
        dtype = torch.float16, # <-- FORZATO per la tua Tesla T4 (Turing non supporta bfloat16)
        load_in_4bit = True,   # <-- FONDAMENTALE per non superare i 15 GB di VRAM
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
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

print("\nPipeline di autofagia completata per tutte le generazioni!")