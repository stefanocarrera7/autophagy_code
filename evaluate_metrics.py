from datasets import Dataset
import metrics
from post_processing import remove_markdown, light_cleanup
from eval import test_solutions
import gc

def evaluate_and_push_metrics(
    test_synth: Dataset,      # dataset da testare
    real_data_test: str,      # only used in naming the metrics dataset for hf
    base_tag: str, 
    lr: float, 
    gen_round: int,
    verbose = False
) -> None:
    """
    Valuta il dataset generato, estrae le metriche Halstead e il Maintainability Index,
    e pusha i risultati direttamente su Hugging Face.
    """
    print(f"\nEvaluating synthetic test set and extracting metrics for generation {gen_round}...")
    generation_results = []
    
    for j in range(len(test_synth)):
        raw_sol = str(test_synth["completion"][j])
        sol = remove_markdown(raw_sol)
        sol = light_cleanup(sol)
        
        entry = str(test_synth["entry_point"][j])
        test_cell = str(test_synth["test"][j])
        
        row_metrics = {
            "task_index": j,
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
            "maintainability_index": None
        }

        if test_cell == "nan" or not test_cell.strip():
            row_metrics["error_type"] = "NoTestData"
            generation_results.append(row_metrics)
            continue

        # Esecuzione test
        res = test_solutions([sol], entry, test_cell, "human_eval", verbose=verbose)

        row_metrics["test_run_time_ms"] = res.get("solutions_summary", {}).get("time_ms", None)
        
        # Analisi Eseguibilità e Correttezza
        if res.get("prop_correct_defined", 0) > 0.99:
            row_metrics["is_executable"] = True
            
        if not res['solutions_summary']:
            if "EntryPointNotFound" in res.get("errors", []):
                row_metrics["error_type"] = "EntryPointNotFound"
            else:
                row_metrics["error_type"] = "SyntaxError"
        else:
            summary = res['solutions_summary'][0]
            ok_count = summary.get('ok', 0)
            fail_count = summary.get('fail', 0)
            
            row_metrics["tests_passed"] = ok_count
            row_metrics["tests_failed"] = fail_count
            
            if fail_count == 0 and ok_count > 0:
                row_metrics["is_correct"] = True

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

        generation_results.append(row_metrics)

    # Salvataggio delle metriche su Hugging Face
    metrics_dataset = Dataset.from_list(generation_results)
    metrics_data_id = f"stefanocarrera/autophagycode_metrics_D_metrics_{real_data_test}_{base_tag}_lr{lr}_gen{gen_round}"
    metrics_dataset.push_to_hub(metrics_data_id)
    print(f"Pushed metrics to {metrics_data_id}")

    del generation_results
    del metrics_dataset
    gc.collect()