import json
from datasets import Dataset
from generate_solutions import generate_solutions
from evaluate_metrics import evaluate_correctness_only, evaluate_executable_only


def generate_sample(data,
                    model,
                    tokenizer,
                    n_solutions:int = 1,
                    real_data_strategy: str = 'trust',  # 'replace', 'trust', 'text'
                    is_instruct: bool = False,
                    model_type: str = "qwen",
                    temperature: float = 1,
                    top_p: float = 0.95,
                    save_token_log: bool = False
                    ) -> Dataset:

    sample = []

    if real_data_strategy != 'text':
        max_new_t = 300
    else:
        max_new_t = 64

    for row in range(len(data)):

        prompt = data[row]['prompt']

        if prompt.endswith('..'):
            prompt = prompt[:-1] + '\n'
        # Forza il ritorno a capo se non c'è già
        if not prompt.endswith('\n'):
            prompt += '\n'


        gen_prompt = prompt
        if is_instruct:
            if model_type.lower() == "qwen":
                gen_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                gen_prompt = f"### Prompt:\n{prompt}\n\n### Completion:\n"
        

        solutions, top_k_progs = generate_solutions(gen_prompt, model, tokenizer,
                                       n_solutions=n_solutions, 
                                       max_new_tokens=max_new_t, 
                                       do_sample=True,
                                       temperature=temperature,
                                       top_p=top_p,
                                       save_token_log=save_token_log)

        # Aggiunta delle soluzioni
        for s in range(n_solutions):
            sample.append({
                            "task_id": data[row]["task_id"],
                            "entry_point": data[row]['entry_point'],
                            "prompt": prompt,
                            "completion": solutions[s],
                            "top_k_progression": json.dumps(top_k_progs[s]) if top_k_progs is not None else None,
                            "test": data[row]['test'],
                        })
        
        if (row+1) % 10 == 0:
            print(f"{row+1} / {len(data)} tasks processed.")

    return Dataset.from_list(sample)



