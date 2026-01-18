import ast
from statistics import mean
from .metrics import passatk
from .metrics import halstead_metrics, original_MI
from .gen import generate_solutions
import datasets
import time
import json
import signal


# ---------------- Timeout handler ----------------
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

# Registra il signal per SIGALRM (solo su UNIX/Linux/Mac)
signal.signal(signal.SIGALRM, timeout_handler)

#################################

def extract_inputs_results(code_str, target_names=("inputs", "results")):
    """Estrae input e risultati per il formato HumanEval (he)."""
    tree = ast.parse(code_str)
    found = {}
    class Finder(ast.NodeVisitor):
        def _capture(self, targets, value):
            try:
                lit = ast.literal_eval(value)
            except Exception:
                return
            for t in targets:
                if isinstance(t, ast.Name) and t.id in target_names:
                    found[t.id] = lit
        def visit_Assign(self, node):
            self._capture(node.targets, node.value)
            self.generic_visit(node)
        def visit_AnnAssign(self, node):
            targets = [node.target] if node.target else []
            if node.value is not None:
                self._capture(targets, node.value)
            self.generic_visit(node)
    Finder().visit(tree)
    return found



############################
#### USING PERF_COUNTER ####
############################

# def test_solutions(solutions, entry_point, test_data, data_format="he", test_timeout=1):
#     """
#     Testa le soluzioni generate gestendo correttamente timeout e scope.
#     """
#     n_correct = 0
#     prop_test_passed = []
#     run_times = []
#     best_ok = 0
#     best_sol = solutions[0] if solutions else ""

#     # Preparazione dei dati di test
#     if data_format == 'he':
#         td = extract_inputs_results(test_data)
#         inputs_list = td.get("inputs", [])
#         results_list = td.get("results", [])
#         n_tests = min(len(inputs_list), len(results_list))
#     else:
#         # Per MBPP puliamo le righe vuote
#         asserts = [line.strip() for line in test_data.split("\n") if line.strip()]
#         n_tests = len(asserts)

#     if n_tests == 0:
#         print(f"Nessun test trovato nella stringa:\n {test_data}")
#         return {"best_sol": best_sol,
#                 "c": 0,
#                 "prop_test_passed": [],
#                 "running_time": []}

#     for i, sol in enumerate(solutions):
#         sol_runtime = 0
#         ns = {}
#         try:
#             # --- FASE 1: DEFINIZIONE DELLA FUNZIONE ---
#             # Non usiamo timeout qui perché è solo definizione ed è immediata.
#             exec(sol.strip(), ns, ns)

#             if entry_point not in ns:
#                 # Se l'entry point non c'è, proviamo a stampare cosa c'è per debug
#                 keys = [k for k in ns.keys() if not k.startswith('__')]
#                 print(f"[SOL {i}] Errore: La funzione '{entry_point}' non è stata definita. Trovato: {keys}")
#                 continue
            
#             candidate_func = ns[entry_point]

#             # --- FASE 2: ESECUZIONE DEI TEST ---
#             ok = 0
#             fail = 0
            
#             if data_format == 'he':
#                 for inp, expected in zip(inputs_list[:n_tests], results_list[:n_tests]):
#                     t_start = time.perf_counter() # <<< START TIMER
#                     try:
#                         if isinstance(inp, (list, tuple)):
#                             out = func_timeout(test_timeout, candidate_func, args=inp)
#                         else:
#                             out = func_timeout(test_timeout, candidate_func, args=(inp,))

#                         t_end = time.perf_counter() # <<< STOP TIMER
#                         sol_runtime += (t_end - t_start)
                        
#                         if out == expected:
#                             ok += 1
#                         else:
#                             fail += 1
#                     except FunctionTimedOut:
#                         fail += 1
#                         sol_runtime += test_timeout
#                     except Exception:
#                         fail += 1

#             else: # data_format == 'mbpp' (basato su assert)
#                 for a in asserts:
#                     t_start = time.perf_counter() # <<< START TIMER
#                     try:
#                         func_timeout(test_timeout, exec, args=(a, ns, ns))
#                         t_end = time.perf_counter() # <<< STOP TIMER
#                         sol_runtime += (t_end - t_start)

#                         ok += 1
#                     except FunctionTimedOut:
#                         print(f"Timeout test: {a}")
#                         fail += 1
#                         sol_runtime += (test_timeout)
#                     except AssertionError:
#                         # print(f"   Assert fallito: {a}")
#                         fail += 1
#                     except Exception as e:
#                         # print(f"   Errore runtime: {e}")
#                         fail += 1

#             # --- FASE 3: STATISTICHE ---
#             total = ok + fail
#             ratio = ok / total if total > 0 else 0
#             prop_test_passed.append(round(ratio, 2))
#             run_times.append(round((sol_runtime/ok)*1000, 4))    # ms

#             if ok > best_ok:
#                 best_ok = ok
#                 best_sol = sol

#             if fail == 0 and total > 0:
#                 print(f"[SOLUTION {i}] ✅ All tests passed ({ok}/{total})")
#                 n_correct += 1
#             else:
#                 print(f"[SOLUTION {i}] {fail}/{total} tests failed")
#                 pass

#         except SyntaxError:
#             print(f"[SOLUTION {i}] ERRORE SINTASSI: Il codice generato è malformato.")
#             prop_test_passed.append(None)
#             run_times.append(None)
#             continue
#         except Exception as e:
#             print(f"[SOLUTION {i}] Errore generico durante il setup: {e}")
#             prop_test_passed.append(None)
#             run_times.append(None)
#             continue

#     return {"best_sol": best_sol,
#             "c": n_correct, 
#             "prop_test_passed": prop_test_passed,
#             "running_time": run_times}



######################
#### USING TIMEIT ####
######################
import functools
import timeit

def wrapper_func_time(func, run, *args):
    frozen_func = functools.partial(func, *args)
    t_runs = timeit.repeat(frozen_func, repeat = run, number=1)
    return min(t_runs)


def test_solutions(solutions, entry_point, test_data, data_format="he", test_runs = 5, verbose = False):
    """
    Testa le soluzioni generate
    """
    n_correct = 0
    best_ok = 0
    best_sol_time = None
    best_sol = solutions[0] if solutions else ""
    solutions_sum = []

    # Preparazione dei dati di test
    if data_format == 'he':
        td = extract_inputs_results(test_data)
        inputs_list = td.get("inputs", [])
        results_list = td.get("results", [])
        n_tests = min(len(inputs_list), len(results_list))
    else:
        # Per MBPP puliamo le righe vuote
        asserts = [line.strip() for line in test_data.split("\n") if line.strip()]
        n_tests = len(asserts)

    if n_tests == 0:
        if verbose:
            print(f"Nessun test trovato nella stringa:\n {test_data}")
        return {"best_sol": best_sol,
                "c": 0,
                "prop_test_passed": [],
                "running_time": []}

    for i, sol in enumerate(solutions):
        timeouts = 0
        ns = {}
        try:
            # --- FASE 1: DEFINIZIONE DELLA FUNZIONE ---
            # Anche la definizione dovrebbe essere protetta da timeout (per loop infiniti globali)
            signal.alarm(1)
            try:
                exec(sol.strip(), ns, ns)
            finally:
                signal.alarm(0) # DISATTIVA ALLARME

            if entry_point not in ns:
                keys = [k for k in ns.keys() if not k.startswith('__')]
                if verbose:
                    print(f"[SOL {i}] Errore: La funzione '{entry_point}' non è stata definita. Trovato: {keys}")
                continue
            
            candidate_func = ns[entry_point]

            # --- FASE 2: ESECUZIONE DEI TEST ---
            ok = 0
            fail = 0
            sol_time = False
            
            if data_format == 'he':
                for inp, expected in zip(inputs_list[:n_tests], results_list[:n_tests]):

                    if timeouts >= 1:
                        if verbose:
                            print('Too much timeouts for the same solution in the test. Skipping this solution\n')
                        break
                    
                    try:
                        # --- INIZIO LOGICA SIGNAL CORRETTA ---
                        signal.alarm(5) # Imposta timer
                        try:
                            if isinstance(inp, (list, tuple)):
                                out = candidate_func(*inp)
                            else:
                                out = candidate_func(inp)
                        finally:
                            signal.alarm(0) # DISATTIVA timer (fondamentale!)
                        # --- FINE LOGICA SIGNAL CORRETTA ---

                        if out == expected:
                            ok += 1
                        else:
                            fail += 1
                            
                    except TimeoutException:
                        if verbose:
                            print('TIMEOUT\n')
                        timeouts += 1
                        fail += 1 # Aggiunto fail qui per coerenza
                    except Exception:
                        fail += 1

                # Estrapoliamo il running time solo se la correttezza funzionale è garantita
                if fail == 0:
                    sol_time = 0
                    sample_size = min(20, n_tests)

                    for inp, expected in zip(inputs_list[:sample_size], results_list[:sample_size]):
                        if isinstance(inp, (list, tuple)):
                            t_run = wrapper_func_time(candidate_func, test_runs, *inp)
                        else:
                            t_run  = wrapper_func_time(candidate_func, test_runs, inp)
                        sol_time += t_run 

            else: # data_format == 'mbpp'
                for a in asserts:
                    if timeouts >= 1:
                        if verbose:
                            print('Too much timeouts for the same solution in the test. Skipping this solution\n')
                        break
                    try:
                        signal.alarm(5)
                        try:
                            exec(a, ns, ns)
                        finally:
                            signal.alarm(0) # DISATTIVA timer
                        
                        ok += 1
                    except AssertionError:
                        fail += 1
                    except TimeoutException:
                        if verbose:
                            print('TIMEOUT\n')
                        timeouts += 1
                        fail += 1
                    except Exception as e:
                        fail += 1

                if fail == 0:
                    sol_time = 0
                    for a in asserts:
                        t_run = wrapper_func_time(exec, test_runs, a, ns, ns)
                        sol_time += t_run

            # --- FASE 3: STATISTICHE ---
            if sol_time is not False:
                sol_time_ms = sol_time * 1000
            else:
                sol_time_ms = None

            total = ok + fail
            ratio = ok / total if total > 0 else 0
            solutions_sum.append({"sol": sol, "ok": ok, "fail": fail, "total": total, "ratio": ratio, "time_ms": sol_time_ms, 'timeout': True if timeouts > 0 else False})

            if ok > best_ok:
                best_ok = ok
                best_sol = sol
                # Non aggiorniamo il sol_time qui perché potrebbe non essere definito se nessuna soluzione ha ancora passato tutti i test

            if fail == 0 and total > 0:
                if verbose:
                    print(f"[SOLUTION {i}] ✅ All tests passed ({ok}/{total})")
                n_correct += 1
                if best_sol_time == None:
                    best_sol_time = sol_time_ms
                    best_sol = sol
                elif sol_time_ms < best_sol_time:
                    best_sol_time = sol_time_ms
                    best_sol = sol
            else:
                if verbose:
                    print(f"[SOLUTION {i}] {fail}/{total} tests failed")
                pass

        except SyntaxError:
            if verbose:
                print(f"[SOLUTION {i}] ERRORE SINTASSI: Il codice generato è malformato.")
            continue
        except TimeoutException:
            # Cattura timeout durante la fase di definizione (exec iniziale)
            if verbose:
                print(f"[SOLUTION {i}] TIMEOUT durante la definizione.")
            continue
        except Exception as e:
            if verbose:
                print(f"[SOLUTION {i}] Errore generico durante il setup: {e}")
            continue

    return {"solutions_summary": solutions_sum,
            "best_sol": best_sol,
            "best_sol_time_ms": best_sol_time,
            "c": n_correct
            }


def test_model(test_split, model, tokenizer, n_solutions, save_path, data_format:str = "he", k: int = 1, start_index = 0, end_index = 146):

    # pass_at_k, tpr, best_sols_acc, times = [], [], [], []
    # pass_at_k = []

    # tot_c = []
    for row in range(start_index, end_index):   # len(test_split)
        print(f"Testing Solution {row}")
        ex = test_split[row]
        solutions = generate_solutions(ex['prompt'], ex['entry_point'], model, tokenizer, n_solutions=n_solutions)
        perf = test_solutions(solutions, ex['entry_point'], ex['test'], data_format=data_format)
        res = {"task_id":ex['task_id'],
               'best_solution':perf['best_sol'],
               'correct_ratio':perf['c']/n_solutions,
               'run_time': perf['best_sol_time'],
               'best_solution_halstead': halstead_metrics(perf['best_sol']),
               'pass_at_k': passatk(n_solutions, perf['c'], k),
               'MI': original_MI(perf['best_sol'])}
        # n, c, t = perf['n'], perf['c'], perf['best_time']
        # pass_at_k.append(passatk(n_solutions, perf['c'], k))
        # tpr.append(c / n if n else 0.0)
        # if t is not None:
        #     times.append(t)

        with open(save_path, "a") as f:
          f.write(json.dumps(res) + "\n")


    return print(f"Risultati salvati in {save_path}")
    #
    #     {
    #     "best_sol_acc_avg": mean(best_sols_acc) if best_sols_acc else 0.0,
    #     "avg_TPR": mean(tpr) if tpr else 0.0,
    #     "avg_passatk": mean(pass_at_k) if pass_at_k else 0.0,
    #     "avg_time": mean(times) if times else None,
    # }

        