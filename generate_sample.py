from datasets import Dataset
from gen import generate_solutions
from post_processing import remove_markdown2
import random
from eval import test_solutions
from evaluate_metrics import evaluate_correctness_only, evaluate_executable_only


def generate_sample(data,
                    model,
                    tokenizer,
                    n_solutions:int = 1,
                    real_data_strategy: str = 'trust',  # 'replace', 'trust', 'text'
                    ) -> Dataset:

    sample = []

    if real_data_strategy != 'text':
        max_new_t = 512
    else:
        max_new_t = 64

    for row in range(len(data)):

        prompt = data[row]['prompt']

        if prompt.endswith('..'):
            prompt = prompt[:-1] + '\n'
        # Forza il ritorno a capo se non c'è già
        if not prompt.endswith('\n'):
            prompt += '\n' 
        

        solutions = generate_solutions(prompt, model, tokenizer, n_solutions=n_solutions, max_new_tokens=max_new_t)

        # Aggiunta delle soluzioni
        for s in range(n_solutions):
            sample.append({
                            "task_id": data[row]["task_id"],
                            "entry_point": data[row]['entry_point'],
                            "prompt": prompt,
                            "completion": solutions[s],
                            "test": data[row]['test'],
                        })
        
        if (row+1) % 10 == 0:
            print(f"{row+1} / {len(data)} tasks processed.")

            
    if real_data_strategy == 'scm':
        sample = synth_correct_mantain(Dataset.from_list(sample), real_data_test='he')
        return sample
    
    if real_data_strategy == 'sem':
        sample = synth_executable_mantain(Dataset.from_list(sample), real_data_test='he')
        return sample

    return Dataset.from_list(sample)



def original_correct_replace(data: Dataset, original_data: Dataset, real_data_str: str) -> Dataset:
    """
    Sostituisce le soluzioni errate usando il task_id per garantire l'allineamento.
    """
    # mappa della correttezza {task_id: True/False}
    correctness_map = evaluate_correctness_only(data, real_data_str)
    original_mapping = {row['task_id']: row['completion'] for row in original_data}

    def replacement_logic(example):
        tid = example['task_id']
        if not correctness_map.get(tid, False) and original_mapping[tid]:
            example['completion'] = original_mapping[tid]
        return example

    return data.map(replacement_logic)



def synth_executable_mantain(synth_data: Dataset, real_data_test: str) -> Dataset:
    """
    Filtra il dataset mantenendo solo i campioni con task_id corretti.
    """
    executable_map = evaluate_executable_only(synth_data, real_data_test)

    filtered_dataset = synth_data.filter(
        lambda row: executable_map.get(row['task_id'], False)
    )

    print(f"Soluzioni eseguibili mantenute: {len(filtered_dataset)} su {len(synth_data)}")
    
    return filtered_dataset



def synth_correct_mantain(synth_data: Dataset, real_data_test: str) -> Dataset:
    """
    Filtra il dataset mantenendo solo i campioni con task_id corretti.
    """
    correctness_map = evaluate_correctness_only(synth_data, real_data_test)

    filtered_dataset = synth_data.filter(
        lambda row: correctness_map.get(row['task_id'], False)
    )

    print(f"Soluzioni corrette mantenute: {len(filtered_dataset)} su {len(synth_data)}")
    
    return filtered_dataset



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
            sol = remove_markdown2(str(row["completion"]))
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


