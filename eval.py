from statistics import mean
from metrics import passatk
from metrics import halstead_metrics, original_MI
from gen import generate_solutions
import time
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


    asserts = [
        line.strip() 
        for line in test_cell.split("\n") 
        if line.strip().startswith("assert ")
    ]
    n_tests = len(asserts)

    # ---- CASO ERRORE: Nessun test trovato nella cella fornuta alla funzione
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
    
    # ===== MAIN LOOP =====
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
            sol_error = None 

            
            if data_format == 'mbpp':
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
                            t_duration = time.perf_counter() - t_start
                            assert_times.append(t_duration)
                            ok += 1
                        finally:
                            signal.alarm(0)
                    except AssertionError:
                        fail += 1
                        if not sol_error: sol_error = "AssertionError"
                    except TimeoutException:
                        fail += 1
                        timeouts += 1
                        if not sol_error: sol_error = "TimeoutException"
                    except Exception as e:
                        fail += 1
                        if not sol_error: sol_error = type(e).__name__
                
                if fail == 0 and assert_times:
                    sol_time = sum(assert_times)


            else:  # data_format == 'he'
                try:
                    exec(test_cell, ns, ns)
                    
                    if 'check' in ns:
                        signal.alarm(3)
                        try:
                            ns['check'](candidate_func)
                            ok += 1 
                        finally:
                            signal.alarm(0)

                except TimeoutException:
                    fail += 1
                    timeouts += 1
                    sol_error = "TimeoutException"
                except AssertionError:
                    fail += 1
                    sol_error = "AssertionError"
                except Exception as e:
                    fail += 1
                    sol_error = type(e).__name__

            # --- FASE 3: STATISTICHE ---
            sol_time_ms = sol_time * 1000 if sol_time is not False else None

            total = ok + fail
            ratio = ok / total if total > 0 else 0
                
            solutions_sum.append({
                "sol": sol, 
                "ok": ok, 
                "fail": fail, 
                "ratio": ratio, 
                "time_ms": sol_time_ms,
                "error_type": sol_error 
            })

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