from datasets import Dataset
import metrics
from post_processing import remove_markdown2, remove_repetition, remove_check, extract_functions
from radon.visitors import HalsteadVisitor
from radon.metrics import halstead_visitor_report
from radon.visitors import ComplexityVisitor
from radon.raw import analyze
from eval import test_solutions
import gc
import json

def evaluate_and_push_metrics(
    test_synth: Dataset,      # dataset da testare
    real_data_test: str,
    tokenizer,
    synth_repo: str,
    verbose = False
    ) -> None:
    """
    Valuta il dataset generato, estrae le metriche Halstead e il Maintainability Index,
    e pusha i risultati direttamente su Hugging Face.
    """

    generation_results = []
    
    for j in range(len(test_synth)):
        sol = str(test_synth["completion"][j])

        sol = extract_functions(sol)
        sol = remove_markdown2(sol, str(test_synth["prompt"][j]))
        sol = remove_check(sol)
        
        entry = str(test_synth["entry_point"][j])
        test_cell = str(test_synth["test"][j])

        # Quante funzioni ha definito la soluzione? - Per capire la possibile ripetizione
        n_def = sol.count("def ")
        n_entry = sol.count("def " + entry + "(")

        # Se c'e stata una ripetizione, usiamo la logica definita in remove_repetitions
        if n_def > 1 or n_entry > 1:
            sol = remove_repetition(sol, entry)

        raw_token_dict = metrics.token_dictionary(sol, tokenizer)
        string_key_token_dict = {str(k): v for k, v in raw_token_dict.items()} if raw_token_dict else {}
        
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
            "halstead_time": None,
            "cyclomatic_complexity": None,
            "maintainability_index": None,
            "loc": None,
            "sloc": None,
            "comment_percentage": None,
            "TTR": metrics.ttr(sol, tokenizer),
            "token_dict": json.dumps(string_key_token_dict),
            "shannon_entropy": metrics.token_entropy(sol, tokenizer),
            "n_func_defined": n_def,
            "entry_point_repeated": n_entry > 1
        }

        if test_cell == "nan" or not test_cell.strip():
            row_metrics["error_type"] = "NoTestData"
            generation_results.append(row_metrics)
            continue


        # Esecuzione test
        res = test_solutions([sol], entry, test_cell, test_format=real_data_test, verbose=verbose)
        
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

        # # Calcolo Metriche Halstead
        # try:
        #     visitor = HalsteadVisitor.from_code(sol)
        #     report = halstead_visitor_report(visitor)

        #     row_metrics["halstead_vocabulary"] = report.vocabulary
        #     row_metrics["halstead_length"] = report.length
        #     row_metrics["halstead_volume"] = report.volume
        #     row_metrics["halstead_difficulty"] = report.difficulty
        #     row_metrics["halstead_effort"] = report.effort
        #     row_metrics["halstead_time"] = report.time

        #     row_metrics["cyclomatic_complexity"] = ComplexityVisitor.from_code(sol).total_complexity
            
        # except SyntaxError:
        #     pass
        # except Exception as e:
        #     pass
        
        # # Calcolo Maintainability Index
        # try:
        #     mi_result = metrics.original_MI(sol)
        #     if mi_result is not None:
        #         row_metrics["maintainability_index"] = mi_result
        # except SyntaxError:
        #     pass
        # except Exception:
        #     pass


        # Calcolo Metriche con Multimetric
        multi_stats = metrics.get_multimetric_from_string(sol)
        
        if multi_stats:
            # Multimetric restituisce i valori in un dizionario con queste chiavi:
            row_metrics["halstead_vocabulary"] = multi_stats.get('operands_uniq', 0) + multi_stats.get('operators_uniq', 0)
            row_metrics["halstead_length"] = multi_stats.get('operands_sum', 0) + multi_stats.get('operators_sum', 0)
            
            # Valori diretti di Halstead
            row_metrics["halstead_volume"] = multi_stats.get('halstead_volume')
            row_metrics["halstead_difficulty"] = multi_stats.get('halstead_difficulty')
            row_metrics["halstead_effort"] = multi_stats.get('halstead_effort')
            row_metrics["halstead_time"] = multi_stats.get('halstead_timerequired')
            
            # Puoi anche estrarre altre metriche utilissime già che ci sei!
            row_metrics["cyclomatic_complexity"] = multi_stats.get('cyclomatic_complexity')
            row_metrics["maintainability_index"] = multi_stats.get('maintainability_index')

            # row_metrics["loc"] = multi_stats.get('loc', 0)
            # row_metrics["sloc"] = multi_stats.get('sloc', 0)
            # comment_ratio = multi_stats.get('comment_ratio', 0)
            # row_metrics["comment_percentage"] = round(comment_ratio, 2)
        
        else:
            # Se multi_stats è None (es. codice rotto), le metriche restano None
            pass

        try:
            # Radon raw parser è ultraveloce e specifico per Python
            raw_metrics = analyze(sol)
            row_metrics["loc"] = raw_metrics.loc
            row_metrics["sloc"] = raw_metrics.sloc
            
            righe_di_commento = raw_metrics.comments + raw_metrics.multi
            
            if raw_metrics.loc > 0:
                row_metrics["comment_percentage"] = round((righe_di_commento / raw_metrics.loc) * 100, 2)
            else:
                row_metrics["comment_percentage"] = 0.0

        except SyntaxError:
            pass


        generation_results.append(row_metrics)

    # Salvataggio delle metriche su Hugging Face
    metrics_dataset = Dataset.from_list(generation_results)

    metrics_data_id = synth_repo + "_metrics"
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
        sol = remove_markdown2(str(row["completion"]), str(row["prompt"]))
        sol = remove_check(sol)
        entry = str(row["entry_point"])
        test_cell = str(row["test"])
        tid = row["task_id"]
        
        is_correct = False
        if test_cell.strip() and test_cell != "nan":
            res = test_solutions([sol], entry, test_cell, test_format=real_data_test)
            if res['solutions_summary']:
                summary = res['solutions_summary'][0]
                if summary.get('fail', 0) == 0 and summary.get('ok', 0) > 0:
                    is_correct = True
        
        correctness_map[tid] = is_correct
        
    return correctness_map


def evaluate_executable_only(test_synth: Dataset, real_data_test: str) -> dict:
    """
    Ritorna un dizionario {task_id: is_executable:bool} per un allineamento sicuro.
    """
    executable_map = {}
    
    for row in test_synth:
        sol = remove_markdown2(str(row["completion"]), str(row["prompt"]))
        entry = str(row["entry_point"])
        test_cell = str(row["test"])
        tid = row["task_id"]
        
        is_executable = False
        if test_cell.strip() and test_cell != "nan":
            res = test_solutions([sol], entry, test_cell, test_format=real_data_test)
            if res['prop_correct_defined'] > 0.99:
                    is_executable = True
        
        executable_map[tid] = is_executable
        
    return executable_map