from datasets import Dataset
from unsloth import FastLanguageModel
from generate_sample import generate_sample
from train_unsloth import finetune_model_unsloth
from eval import test_model 
from huggingface_hub import HfApi

def _sanitize_repo_name(text: str) -> str:
    return text.replace("/", "__").replace(" ", "_")

def autophagy(
    base_model_id: str,
    real_data_train: Dataset,
    real_data_test: Dataset,
    g: int = 10,
    n_solutions: int = 10,
    data_format: str = "he",
    pass_at_k: int = 1
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
    base_tag = _sanitize_repo_name(base_model_id)
    prev_adapter_repo = None

    for t in range(g):
        synth = generate_sample(sample, gen_model, gen_tok, n_solutions=n_solutions)

        ft_dir = f"runs/gen_{t:02d}/adapters"
        ft_model, ft_tok = finetune_model_unsloth(
            dataset = synth,
            base_model_id = base_model_id,
            output_dir = ft_dir,
            num_train_epochs = 2,
            lr = 2e-4,
            batch_size = 1,
            grad_accum = 16,
            max_length = 2048,
            resume_adapter_repo=prev_adapter_repo
        )

        # perf = test_model(real_data_test, ft_model, ft_tok, n_solutions=n_solutions, data_format=data_format, k=pass_at_k)
        # print("Average Correct solutions per task: ", perf['avg_c'])
        # print(f"[gen {t}] metrics: {perf}")

        # naming
        model_id = f"stefanocarrera/autophagycode_M_{base_tag}_gen{t+1}"
        data_id  = f"stefanocarrera/autophagycode_D_{base_tag}_gen{t+1}"

        # push to hub
        api = HfApi()
        synth.push_to_hub(data_id)
        print(f"Pushed data to {data_id}")
        api.create_repo(model_id, repo_type="model", private=True, exist_ok=True)
        ft_model.push_to_hub(model_id)
        ft_tok.push_to_hub(model_id)
        print(f"Pushed model to {model_id}")

        prev_adapter_repo = model_id
        gen_model, gen_tok = ft_model, ft_tok
        sample = synth

    return gen_model, gen_tok