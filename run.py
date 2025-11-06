from datasets import load_dataset
import autophagy
from huggingface_hub import login

HF_TOKEN = "xxxxxxxxx"
login(token=HF_TOKEN)

# 1. Modello di partenza
base_model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# 2. Dati reali: HumanEval in questo esempio
#    Deve avere i campi: ['task_id', 'prompt', 'entry_point', 'test']
real_data = load_dataset("openai_humaneval", split="test")
real_data = real_data.train_test_split(test_size=0.33, seed=42)
real_data_train = real_data["train"]
real_data_test = real_data["test"]


# 4. Lancia l’autofagia per g generazioni
autophagy.autophagy(
    base_model_id=base_model_id,
    real_data_train=real_data_train,
    real_data_test=real_data_test,
    g=2,               # numero di generazioni
    n_solutions=5      # soluzioni generate per prompt
)
