from datasets import Dataset
from gen import generate_solutions
import random
from eval import test_solutions
from evaluate_metrics import evaluate_and_push_metrics


def generate_sample(data,
                    model,
                    tokenizer,
                    n_solutions:int = 1,
                    real_data_strategy: str = None,  # 'replace', 'augment'
                    real_data_prop: float = 0):

    sample = []
    index_to_insert = None

    if real_data_strategy in ['augment', 'replace'] and real_data_prop > 0:
        index_to_insert = random.sample(range(len(data)), int(len(data)*real_data_prop))

    for row in range(len(data)):

        if index_to_insert and (row in index_to_insert) and (real_data_strategy == 'replace'):
            sample.append({
                "task_id": data[row]["task_id"],
                "entry_point": data[row]["entry_point"],
                "prompt": data[row]["prompt"],
                "completion": data[row]["completion"],
                "test": data[row]['test'],
            })
            continue

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

    if index_to_insert and (real_data_strategy == 'augment') and (real_data_prop > 0):
        for i in index_to_insert:
            sample.append({
                "task_id": data[i]["task_id"],
                "entry_point": data[i]["entry_point"],
                "prompt": data[i]["prompt"],
                "completion": data[i]["completion"],
                "test": data[i]['test'],
                })

    return Dataset.from_list(sample)


def correct_replace(data: Dataset, original_data: Dataset, real_data_str: str, base_tag: str, lr: float, gen_round: int) -> Dataset:
    """
    Sostituisce le soluzioni errate con quelle originali
    """
    metrics = evaluate_and_push_metrics(data, real_data_str, base_tag, lr, gen_round, push=False)
    is_correct_list = [m['is_correct'] for m in metrics]

    # dizionario di mappatura veloce: task_id -> completion originale
    original_mapping = {row['task_id']: row['completion'] for row in original_data}

    def replacement_logic(example, idx):
        if not is_correct_list[idx]:
            # recuperiamo la soluzione corretta usando il task_id
            example['completion'] = original_mapping[example['task_id']]
        return example

    return data.map(replacement_logic, with_indices=True)
