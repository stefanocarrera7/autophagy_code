# Script per estrarre le metriche di valutazione direttamente dai dataset sintetici generati
from evaluate_metrics import evaluate_and_push_metrics
from datasets import load_dataset, Dataset
from autophagy_clean import _sanitize_repo_name
from huggingface_hub import login

login(token = "xxx")

g = 5
base_model_id = "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit"
lr = 0.0001
chunk = 138
base_tag = _sanitize_repo_name(base_model_id)

for t in range(4, g + 1):

    # test_data = load_dataset(f"stefanocarrera/autophagycode_D_he_{base_tag}_lr{lr}_chunk{chunk}_gen{t}_test", split="train")
    test_data = load_dataset(f"stefanocarrera/autophagycode_D_{base_tag}_lr{lr}_gen{t}", split="train")

    evaluate_and_push_metrics(test_data, "he", base_tag, 1e-4, t)