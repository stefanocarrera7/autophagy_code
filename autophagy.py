from datasets import Dataset
from huggingface_hub import login
from unsloth import FastLanguageModel
from data import generate_sample
from train_unsloth import finetune_model_unsloth
from eval import test_model
from huggingface_hub import HfApi

HF_TOKEN = "xxxxxxxxx"
login(token=HF_TOKEN)
api = HfApi()
HF_USER = "username" 
PROJECT = "autophagy-coding"
PRIVATE = True   


def autophagy(
    base_model_id: str,
    real_data_train: Dataset,
    real_data_test: Dataset,
    hf_token: str,
    g: int = 10,
    n_solutions: int = 10
    ):

    # 0) Starting model
    gen_model, gen_tok = FastLanguageModel.from_pretrained(
        model_name = base_model_id,
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
        device_map = "auto",
    )
    FastLanguageModel.for_inference(gen_model)

    sample = real_data_train

    for t in range(g):
        synth = generate_sample(sample, gen_model, gen_tok, n_solutions=n_solutions)

        ft_dir = f"runs/gen_{t:02d}/adapters"
        ft_model, ft_tok = finetune_model_unsloth(
            dataset = synth,
            base_model_id = gen_model,
            output_dir = ft_dir,
            num_train_epochs = 2,
            lr = 2e-4,
            batch_size = 1,
            grad_accum = 16,
            max_length = 2048,
        )

        perf = test_model(real_data_test, ft_model, ft_tok, n_solutions=n_solutions, k=1)
        print(f"[gen {t}] metrics: {perf}")

        # naming
        model_id = f"{HF_USER}/{PROJECT}_M_{base_model_id}_gen{t+1}"
        data_id  = f"{HF_USER}/{PROJECT}_D_{base_model_id}_gen{t+1}"

        # push to hub
        synth.push_to_hub(data_id, private=PRIVATE)
        api.create_repo(model_id, repo_type="model", private=PRIVATE, exist_ok=True)
        ft_model.push_to_hub(model_id)
        ft_tok.push_to_hub(model_id)

        gen_model, gen_tok = ft_model, ft_tok
        sample = synth

    return gen_model, gen_tok