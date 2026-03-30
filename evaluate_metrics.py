from datasets import Dataset
import metrics
from post_processing import remove_markdown2, light_cleanup, remove_repetition, remove_check
from eval import test_solutions
import gc

def evaluate_and_push_metrics(
    test_synth: Dataset,      # dataset da testare
    real_data_test: str,
    base_tag: str, 
    lr: float, 
    gen_round: int,
    verbose = False,
    test_or_train = 'test',
    strategy = 'trust'
) -> None:
    """
    Valuta il dataset generato, estrae le metriche Halstead e il Maintainability Index,
    e pusha i risultati direttamente su Hugging Face.
    """
    print(f"\nEvaluating synthetic test set and extracting metrics for generation {gen_round}...")
    generation_results = []
    
    for j in range(len(test_synth)):
        sol = str(test_synth["completion"][j])
        sol = remove_markdown2(sol, str(test_synth["prompt"][j]))
        sol = remove_check(sol)
        # sol = light_cleanup(sol)
        
        entry = str(test_synth["entry_point"][j])
        test_cell = str(test_synth["test"][j])

        # Quante funzioni ha definito la soluzione? - Per capire la possibile ripetizione
        n_def = sol.count("def ")
        n_entry = sol.count("def " + entry + "(")

        # Se c'e stata una ripetizione, usiamo la logica definita in remove_repetitions
        if n_def > 2 or n_entry > 1:
            sol = remove_repetition(sol, entry)
        
        row_metrics = {
            "task_id": test_synth[j]['task_id'],
            "entry_point": entry,
            "is_executable": False,
            "is_correct": False,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_run_time_ms": None,
            "error_type": None,
            "halstead_vocabulary": None,
            "halstead_length": None,
            "halstead_volume": None,
            "halstead_difficulty": None,
            "halstead_effort": None,
            "maintainability_index": None,
            "n_func_defined": n_def,
            "entry_point_repeated": n_entry > 1
        }

        if test_cell == "nan" or not test_cell.strip():
            row_metrics["error_type"] = "NoTestData"
            generation_results.append(row_metrics)
            continue


        # Esecuzione test
        res = test_solutions([sol], entry, test_cell, data_format=real_data_test, verbose=verbose)
        
        # Analisi Eseguibilità e Correttezza
        if res.get("prop_correct_defined", 0) > 0.99:
            row_metrics["is_executable"] = True
            
        if not res['solutions_summary']:
            # Caso 1: SyntaxError
            row_metrics["error_type"] = res.get("errors", [None])[0]
            
        else:
            # Caso 2: Il codice è stato eseguito, analizziamo i test
            summary = res['solutions_summary'][0]
            
            ok_count = summary.get('ok', 0)
            fail_count = summary.get('fail', 0)
            
            row_metrics["tests_passed"] = ok_count
            row_metrics["tests_failed"] = fail_count
            row_metrics["test_run_time_ms"] = summary.get("time_ms", None)
            
            # Recuperiamo l'errore di runtime
            row_metrics["error_type"] = summary.get("error_type")
            
            if fail_count == 0 and ok_count > 0:
                row_metrics["is_correct"] = True
                row_metrics["error_type"] = None # Se è corretto, non ci sono errori

        # Calcolo Metriche Halstead
        metrics_result = metrics.halstead_metrics(sol)
        if metrics_result:
            row_metrics["halstead_vocabulary"] = metrics_result['vocabulary']
            row_metrics["halstead_length"] = metrics_result['length']
            row_metrics["halstead_volume"] = metrics_result['volume']
            row_metrics["halstead_difficulty"] = metrics_result['difficulty']
            row_metrics["halstead_effort"] = metrics_result['effort']
        
        # Calcolo Maintainability Index
        mi_result = metrics.original_MI(sol)
        if mi_result is not None:
            row_metrics["maintainability_index"] = mi_result

        if row_metrics['error_type'] == 'SyntaxError' and (row_metrics["entry_point_repeated"] or row_metrics['n_func_defined'] > 3):
            row_metrics['error_type'] = 'MaxToken'

        generation_results.append(row_metrics)

    # Salvataggio delle metriche su Hugging Face
    metrics_dataset = Dataset.from_list(generation_results)
    if test_or_train == 'test':
        metrics_data_id = f"stefanocarrera/autophagycode_D_metrics_{real_data_test}_{base_tag}_lr{lr}_{strategy}_g{gen_round}"
        metrics_dataset.push_to_hub(metrics_data_id)
    elif test_or_train == 'train':
        metrics_data_id = f"stefanocarrera/autophagycode_D_metrics_train_{base_tag}_lr{lr}_{strategy}_g{gen_round}"
        metrics_dataset.push_to_hub(metrics_data_id)
    print(f"Pushed metrics to {metrics_data_id}")

    del generation_results
    del metrics_dataset
    gc.collect()


def evaluate_correctness_only(test_synth: Dataset, real_data_test: str) -> dict:
    """
    Ritorna un dizionario {task_id: is_correct:bool} per un allineamento sicuro.
    """
    correctness_map = {}
    
    for row in test_synth:
        sol = remove_markdown2(str(row["completion"]))
        entry = str(row["entry_point"])
        test_cell = str(row["test"])
        tid = row["task_id"]
        
        is_correct = False
        if test_cell.strip() and test_cell != "nan":
            res = test_solutions([sol], entry, test_cell, data_format=real_data_test)
            if res['solutions_summary']:
                summary = res['solutions_summary'][0]
                if summary.get('fail', 0) == 0 and summary.get('ok', 0) > 0:
                    is_correct = True
        
        correctness_map[tid] = is_correct
        
    return correctness_map