from datasets import Dataset
from gen import generate_solutions
from post_processing import remove_markdown
import random
from eval import test_solutions
from evaluate_metrics import evaluate_correctness_only


def generate_sample(data,
                    model,
                    tokenizer,
                    n_solutions:int = 1,
                    real_data_strategy: str = None,  # 'replace', 'augment', 'sc'
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


def original_correct_replace(data: Dataset, original_data: Dataset, real_data_str: str, base_tag: str, lr: float, gen_round: int) -> Dataset:
    """
    Sostituisce le soluzioni errate usando il task_id per garantire l'allineamento.
    """
    # mappa della correttezza {task_id: True/False}
    correctness_map = evaluate_correctness_only(data, real_data_str)
    original_mapping = {row['task_id']: row['completion'] for row in original_data}

    def replacement_logic(example):
        tid = example['task_id']
        
        if not correctness_map.get(tid, False):
            example['completion'] = original_mapping[tid]
        return example

    return data.map(replacement_logic)


def synth_correct_replace(synth_data: Dataset, real_data_test: str = 'he') -> Dataset:
    """
    Takes the synth data with n_solutions solutions generated per task and takes only the first correct among the generated per task.
    If no correct solution has been generated for a given task, then the first is given
    """
    # raggruppiamo le righe per task_id mantenendo l'ordine di generazione
    tasks_grouped = {}
    for row in synth_data:
        tid = row['task_id']
        if tid not in tasks_grouped:
            tasks_grouped[tid] = []
        tasks_grouped[tid].append(row)

    filtered_sample = []

    solution_replaced = {}
    # iteriamo su ogni gruppo di task
    for tid, rows in tasks_grouped.items():
        selected_row = rows[0]  # prendiamo la prima per default se falliscono tutte
        solution_replaced[tid] = False
        
        count = 0
        # valutiamo le soluzioni una per una
        for row in rows:
            sol = remove_markdown(str(row["completion"]))
            entry = str(row["entry_point"])
            test_cell = str(row["test"])
            
            is_correct = False
            if test_cell.strip() and test_cell != "nan":
                # eseguiamo il test solo su questa specifica soluzione
                res = test_solutions([sol], entry, test_cell, data_format=real_data_test)
                if res['solutions_summary']:
                    summary = res['solutions_summary'][0]
                    if summary.get('fail', 0) == 0 and summary.get('ok', 0) > 0:
                        is_correct = True
            
            # Se la soluzione passa i test, la selezioniamo e INTERROMPIAMO il ciclo
            if is_correct:
                selected_row = row
                if count > 0:
                    solution_replaced[tid] = True
                break

            count += 1
                
        filtered_sample.append(selected_row)

    # 4. Ricostruiamo e restituiamo il Dataset filtrato
    return Dataset.from_list(filtered_sample), solution_replaced


