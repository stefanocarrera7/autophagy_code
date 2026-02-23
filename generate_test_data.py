from generate_sample import generate_sample
import torch
import gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# 1. Autenticazione (Assicurati di avere il token impostato come variabile d'ambiente 
# o passalo direttamente qui se non usi huggingface-cli login)
login("xxxxx") 

# 2. Carichiamo HumanEval una volta sola
print("Caricamento dataset base (HumanEval)...")
base_data = load_dataset("openai_humaneval", split="test")

num_generations = 10
n_sol_per_prompt = 1

HF_USERNAME = "stefanocarrera"
# Il nome base dei tuoi modelli fine-tunati (es. MIO_MODELLO_gen1, gen2, ecc.)
MODEL_BASE_NAME = "stefanocarrera/autophagycode_M_meta-llama__Meta-Llama-3.1-8B-Instruct_gen10_TEST" 
DATASET_BASE_NAME = "autophagycode_D_HE_meta-llama__Meta-Llama-3.1-8B-Instruct_gen"

for g in range(num_generations+1):
    print(f"\n{'='*50}")
    print(f"   AVVIO GENERAZIONE E VALUTAZIONE [{g}]")
    print(f"{'='*50}")

    # --- LOGICA DI SELEZIONE DEL MODELLO ---
    if g == 0:
        # Generazione 1: Modello originale di Meta
        model_repo = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    else:
        model_repo = f"stefanocarrera/autophagycode_M_meta-llama__Meta-Llama-3.1-8B-Instruct_gen{g}_TEST"

    dataset_repo = f"{HF_USERNAME}/{DATASET_BASE_NAME}{g}_TEST"

    print(f"Scaricamento e caricamento modello: {model_repo}")
    
    # 3. Caricamento del modello e tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        device_map="auto",
        torch_dtype=torch.float16
    )

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