import ast
from statistics import mean
from metrics import passatk
from metrics import halstead_metrics, original_MI
from gen import generate_solutions
import datasets
import time
import json
import signal
import functools
import timeit

# ---------------- Timeout handler ----------------
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

# Registra il signal per SIGALRM (solo su UNIX/Linux/Mac)
signal.signal(signal.SIGALRM, timeout_handler)

#################################

# def extract_inputs_results(code_str, target_names=("inputs", "results")):
#     """Estrae input e risultati per il formato HumanEval (he)."""
#     try:
#         tree = ast.parse(code_str)
#     except:
#         return {}
#     found = {}
#     class Finder(ast.NodeVisitor):
#         def _capture(self, targets, value):
#             try:
#                 lit = ast.literal_eval(value)
#             except Exception:
#                 return
#             for t in targets:
#                 if isinstance(t, ast.Name) and t.id in target_names:
#                     found[t.id] = lit
#         def visit_Assign(self, node):
#             self._capture(node.targets, node.value)
#             self.generic_visit(node)
#         def visit_AnnAssign(self, node):
#             targets = [node.target] if node.target else []
#             if node.value is not None:
#                 self._capture(targets, node.value)
#             self.generic_visit(node)
#     Finder().visit(tree)
#     return found

def wrapper_func_time(func, run, *args):
    frozen_func = functools.partial(func, *args)
    t_runs = timeit.repeat(frozen_func, repeat=run, number=1)
    return min(t_runs)

def test_solutions(solutions, entry_point, test_cell, data_format="he", test_runs=1, verbose=False):
    """
    Testa le soluzioni generate gestendo sia funzioni globali che metodi di classe Solution.
    """
    n_executable = 0
    n_correct = 0
    errors = []
    best_ok = 0
    best_sol_time = None
    best_sol = solutions[0] if solutions else ""
    solutions_sum = []

    # Preparazione dei dati di test
    if data_format == 'evalplus':
        n_tests = None
    #     td = extract_inputs_results(test_cell)
    #     inputs_list = td.get("inputs", [])
    #     results_list = td.get("results", [])
    #     n_tests = min(len(inputs_list), len(results_list))
    else:    # mbpp o he
        asserts = [
            line.strip() 
            for line in test_cell.split("\n") 
            if line.strip().startswith("assert ")
        ]
        n_tests = len(asserts)

    if n_tests == 0 and n_tests is not None:
        if verbose:
            print(f"Nessun test trovato nella stringa:\n {test_cell}")
        return {
            "solutions_summary": [], 
            "best_sol": best_sol, 
            "best_sol_time_ms": None,
            "prop_correct_defined": 0,
            "c": 0,
            "errors": ["NoTestsFound"]
        }

    for i, sol in enumerate(solutions):
        timeouts = 0
        ns = {}
        # Inseriamo import comuni per evitare NameError
        import math, collections, heapq, bisect, itertools
        ns.update({
            'math': math, 'collections': collections, 'heapq': heapq, 
            'bisect': bisect, 'itertools': itertools,
            'List': list, 'Dict': dict, 'Tuple': tuple, 'Optional': lambda x: x
        })
        
        try:
            # --- FASE 1: DEFINIZIONE ---
            signal.alarm(1)
            try:
                exec(sol.strip(), ns, ns)
            finally:
                signal.alarm(0)

            # se exec() non ha dato errori allora il modello ha generato una soluzione almeno eseguibile:
            n_executable += 1

            # --- FASE 1.5: PROMOZIONE METODI CLASSE SOLUTION ---
            # Se il modello ha usato 'class Solution', portiamo i suoi metodi nel namespace globale
            # if 'Solution' in ns:
            #     try:
            #         sol_instance = ns['Solution']()
            #         for attr_name in dir(sol_instance):
            #             if not attr_name.startswith("__"):
            #                 ns[attr_name] = getattr(sol_instance, attr_name)
            #     except:
            #         pass

            if entry_point not in ns:
                errors.append("EntryPointNotFound")
                if verbose:
                    keys = [k for k in ns.keys() if not k.startswith('__')]
                    print(f"[SOL {i}] Errore: '{entry_point}' non trovata. Trovato: {keys}")
                continue
            
            candidate_func = ns[entry_point]

            # --- FASE 2: ESECUZIONE DEI TEST ---
            ok = 0
            fail = 0
            sol_time = False
            ratio_he = 0.0
            
            if data_format == 'evalplus':
                # Compila la cella di test per caricare le funzioni (es. check) nel namespace ns
                try:
                    exec(test_cell, ns, ns)
                except Exception:
                    pass
                
                check_func = ns.get('check')
                
                try:
                    # Copriamo sia la correttezza sia il tempo con un singolo allarme
                    signal.alarm(5) 
                    try:
                        # 1. Verifica la correttezza
                        if check_func:
                            ratio_he = check_func(candidate_func)
                            if ratio_he is None: 
                                ratio_he = 1.0 # Sicurezza se check non ritorna nulla
                        else:
                            ratio_he = 1.0 # Caso in cui gli assert siano globali in test_cell
                            
                        # 2. Misura il tempo SOLO se ha passato i test (sotto l'ombrello dell'allarme!)
                        if ratio_he >= 0.999:
                            if check_func:
                                sol_time = wrapper_func_time(check_func, test_runs, candidate_func)
                            else:
                                sol_time = wrapper_func_time(exec, test_runs, test_cell, ns, ns)
                                
                    finally:
                        signal.alarm(0) # Spegne sempre e comunque l'allarme
                        
                except TimeoutException:
                    timeouts += 1
                    fail = 1
                    ratio_he = 0.0
                except Exception as e:
                    fail = 1
                    ratio_he = 0.0
                    if verbose: print(f"[SOLUTION {i}] Errore test: {e}")

            elif data_format == 'mbpp':
                for a in asserts:
                    if timeouts >= 1: break
                    try:
                        signal.alarm(3)
                        try:
                            exec(a, ns, ns)
                        finally:
                            signal.alarm(0)
                        ok += 1
                    except:
                        fail += 1

                if fail == 0:
                    sol_time = 0
                    for a in asserts:
                        try:
                            t_run = wrapper_func_time(exec, test_runs, a, ns, ns)
                            sol_time += t_run
                        except:
                            continue
            
            else: # data_format == 'he NON PLUS'
                
                ns['candidate'] = candidate_func 

                assert_times = []
                t_start = None
                for a in asserts:
                    if timeouts >= 1: break
                    try:
                        signal.alarm(3)
                        t_start = time.perf_counter()
                        try:
                            exec(a, ns, ns)
                            # t_end e append vengono eseguiti solo se exec ha successo
                            t_duration = time.perf_counter() - t_start
                            assert_times.append(t_duration)
                            ok += 1
                        finally:
                            signal.alarm(0)
                    except Exception:
                        fail += 1
                        # if verbose: print(f"Test fallito: {a} | Errore: {e}")
                
                if fail == 0 and assert_times:
                    sol_time = sum(assert_times)

            # --- FASE 3: STATISTICHE ---
            sol_time_ms = sol_time * 1000 if sol_time is not False else None
            if data_format == 'evalplus':
                ok = ratio_he
                fail = 1 - ratio_he
                total = 1
                ratio = ratio_he
            else:
                total = ok + fail
                ratio = ok / total if total > 0 else 0
            solutions_sum.append({"sol": sol, "ok": ok, "fail": fail, "ratio": ratio, "time_ms": sol_time_ms})

            if ok > best_ok:
                best_ok = ok
                best_sol = sol

            if fail <= 0.001 and total > 0:
                if verbose: print(f"[SOLUTION {i}] OK: All tests passed ({ok}/{total})")
                n_correct += 1
                if best_sol_time is None or (sol_time_ms is not None and sol_time_ms < best_sol_time):
                    best_sol_time = sol_time_ms
                    best_sol = sol
            elif verbose:
                print(f"[SOLUTION {i}] {fail}/{total} tests failed")

        except Exception as e:
            errors.append(type(e).__name__)
            if verbose: print(f"[SOLUTION {i}] ERRORE SINTASSI: {e}")
            continue

    return {
        "solutions_summary": solutions_sum, 
        "best_sol": best_sol, 
        "best_sol_time_ms": best_sol_time,
        "prop_correct_defined": n_executable / len(solutions) if solutions else 0,
        "c": n_correct,
        "errors": errors
    }