# Script per estrarre le metriche di valutazione direttamente dai dataset sintetici generati
from evaluate_metrics import evaluate_and_push_metrics
from datasets import load_dataset, Dataset
from autophagy_clean import _sanitize_repo_name
from huggingface_hub import login

login(token = "xxx")

g = 10
model_id = "unsloth__Qwen3-8B-Base-unsloth-bnb-4bit"
lr = 0.0001

for t in range(1, g + 1):

    test_data = load_dataset(f"stefanocarrera/autophagycode_D_he_{model_id}_lr0.0001_chunk142_gen{t}_test", split="train")
    evaluate_and_push_metrics(test_data, "he", model_id, 1e-4, t)