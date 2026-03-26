# Script per estrarre le metriche di valutazione direttamente dai dataset sintetici generati
from evaluate_metrics import evaluate_and_push_metrics
from datasets import load_dataset, Dataset
from huggingface_hub import login

login(token = "xxx")

g = 10
MODEL = "Qwen3-14B"
# strategy = 'text'
strategies = ['trust', 'correct', 'text']

for strategy in strategies:
    for t in range(1, g + 1):

        # test_data = load_dataset(f"stefanocarrera/autophagycode_D_train_unsloth__Qwen3-8B-Base-unsloth-bnb-4bit_lr0.0001_chunk142_gen{t}", split="train")
        test_data = load_dataset(f"stefanocarrera/autophagycode_D_he_{MODEL}_strategy_{strategy}_g{t}", split = "train")
        evaluate_and_push_metrics(test_data, "he", MODEL, 1e-4, t, test_or_train='test', strategy=strategy)