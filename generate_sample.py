from datasets import Dataset
from gen import generate_solutions
from eval import test_solutions


def generate_sample(data,
                    model,
                    tokenizer,
                    n_solutions:int = 10):

    sample = []
    for row in range(len(data)):

        entry = data[row]['entry_point']
        prompt = data[row]['prompt']

        solutions = generate_solutions(prompt, entry, model, tokenizer, n_solutions)
        perf = test_solutions(solutions, data[row]['test'], entry)

        sample.append({
                        "task_id": data[row]["task_id"],
                        "prompt": data[row]['prompt'],
                        "completion": perf['best_sol'],
                        "test": data[row]['test'],
                    })

    return Dataset.from_list(sample)