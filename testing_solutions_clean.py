# Script per estrarre le metriche di valutazione direttamente dai dataset sintetici generati
from evaluate_metrics import evaluate_and_push_metrics
from datasets import load_dataset, Dataset
from huggingface_hub import login

login(token = "xxx")


MODELS = ["Qwen3-8B"]
strategies = ['trust']
test_or_train = 'test'

if test_or_train == 'test':
    g = 10
else: 
    g = 9

for MODEL in MODELS:
    for strategy in strategies:
        for t in range(1, g + 1):

            if test_or_train == 'test':
                data = load_dataset(f"stefanocarrera/autophagycode_D_he_Qwen3-8B_strategy_trust_g{t}", split = "train")
                evaluate_and_push_metrics(data, "he", MODEL, 1e-4, t, test_or_train=test_or_train, strategy=strategy)
            else: 
                data = load_dataset(f"stefanocarrera/autophagycode_D_train_{MODEL}_lr0.0001_c142_{strategy}_g{t}", split="train")
                evaluate_and_push_metrics(data, "mbpp", MODEL, 1e-4, t, test_or_train=test_or_train, strategy=strategy)

            