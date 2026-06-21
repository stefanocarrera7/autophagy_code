# Script per estrarre le metriche di valutazione direttamente dai dataset sintetici generati
from evaluate_metrics import evaluate_and_push_metrics
from datasets import load_dataset, Dataset
from huggingface_hub import login
from transformers import AutoTokenizer

login(token = "xxx")

MODELS = ["Qwen3-0.6B"]
strategies = ['trust']
test_or_train = 'test'

if test_or_train == 'test':
    g = 10
else: 
    g = 9


for MODEL in MODELS:
    
    tok = AutoTokenizer.from_pretrained(f"unsloth/{MODEL}")

    for strategy in strategies:

        for t in range(1, 2):

            repo = f"stefanocarrera/autophagycode_D_he_train-mercury_{MODEL}_strategy_trust_t0.2_g{t}"
            data = load_dataset(repo, split = "train")
            evaluate_and_push_metrics(data, "he", tokenizer=tok, synth_repo=repo)


            