from datasets import Dataset
from gen import generate_solutions


def generate_sample(data,
                    model,
                    tokenizer,
                    n_solutions:int = 1):

    sample = []
    for row in range(len(data)):

        entry = data[row]['entry_point']
        prompt = data[row]['prompt']

        solutions = generate_solutions(prompt, entry, model, tokenizer, n_solutions=n_solutions)

        # Qua si potrebbero provare diversi approcci per considerare le soluzioni multiple:
        # - scegliere la soluzione migliore
        # - scegliere una soluzione a caso
        # - includerle tutte nel dataset sintetico (come fatto qui sotto)
        
        for s in range(n_solutions):
            sample.append({
                            "task_id": data[row]["task_id"],
                            "entry_point": entry,
                            "prompt": prompt,
                            "completion": solutions[s],
                            "test": data[row]['test'],
                        })
        
        if (row+1) % 10 == 0:
            print(f"{row+1} / {len(data)} tasks processed.")

    return Dataset.from_list(sample)