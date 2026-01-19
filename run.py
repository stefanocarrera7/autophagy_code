from datasets import load_dataset
import autophagy
from huggingface_hub import login

HF_TOKEN = "xxxx"
login(token=HF_TOKEN)

# 1. Modello di partenza
base_model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# 2. Dati reali: HumanEval in questo esempio
#    Deve avere i campi: ['task_id', 'prompt', 'entry_point', 'test']
real_data = load_dataset("stefanocarrera/D_mercury_to_he", split="train")

# 1. Funzione per verificare se il test è valido (è una stringa e non è vuota)
def is_valid_test(example):
    return isinstance(example['test'], str) and len(example['test'].strip()) > 0

# 2. Applica il filtro al dataset
print(f"Dataset originale: {len(real_data)} righe")
real_data = real_data.filter(is_valid_test)
print(f"Dataset filtrato: {len(real_data)} righe")

real_data_train = real_data
real_data_test = real_data


# 4. Lancia l’autofagia per g generazioni
autophagy.autophagy(
    base_model_id=base_model_id,
    real_data_train=real_data_train,
    real_data_test=real_data_test,
    g=10,               # numero di generazioni
    n_solutions=1      # soluzioni generate per prompt
)
