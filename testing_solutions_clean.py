# Script per estrarre le metriche di valutazione direttamente dai dataset sintetici generati
from evaluate_metrics import evaluate_and_push_metrics
from datasets import load_dataset, Dataset
from autophagy_clean import _sanitize_repo_name
from huggingface_hub import login

login(token = "xxxx")

g = 5
MODEL = "Qwen3-8B"

for t in range(1, g + 1):

    # test_data = load_dataset(f"stefanocarrera/autophagycode_D_he_{MODEL}_strategy_sc_g{t}", split="train")
    test_data = load_dataset(f"stefanocarrera/autophagycode_D_train_{MODEL}_lr0.0001_c142_sc_g{t}", split = "train")
    evaluate_and_push_metrics(test_data, "he", MODEL, 1e-4, t, test_or_train='train')