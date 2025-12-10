import ast
from statistics import mean
from metrics import passatk
from gen import generate_solutions
import datasets
from func_timeout import func_timeout, FunctionTimedOut

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

def test_solutions(solutions, entry_point, test_data, data_format="he", test_timeout=1):
    """
    Testa le soluzioni generate gestendo correttamente timeout e scope.
    """
    n_correct = 0
    prop_test_passed = []
    best_ok = 0
    best_sol = solutions[0] if solutions else ""

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
        print(f"Nessun test trovato nella stringa:\n {test_data}")
        return {"best_sol": best_sol, "c": 0, "prop_test_passed": []}

    for i, sol in enumerate(solutions):
        ns = {}
        try:
            # --- FASE 1: DEFINIZIONE DELLA FUNZIONE ---
            # Non usiamo timeout qui perché è solo definizione ed è immediata.
            # Eseguirla nel thread principale garantisce che 'ns' venga popolato.
            # .strip() è sicuro qui, rimuove solo spazi inizio/fine stringa.
            exec(sol.strip(), ns, ns)

            if entry_point not in ns:
                # Se l'entry point non c'è, proviamo a stampare cosa c'è per debug
                keys = [k for k in ns.keys() if not k.startswith('__')]
                print(f"[SOL {i}] Errore: La funzione '{entry_point}' non è stata definita. Trovato: {keys}")
                continue
            
            candidate_func = ns[entry_point]

            # --- FASE 2: ESECUZIONE DEI TEST ---
            ok = 0
            fail = 0
            
            if data_format == 'he':
                for inp, expected in zip(inputs_list[:n_tests], results_list[:n_tests]):
                    try:
                        # CORREZIONE TIMEOUT: Passiamo funzione e argomenti separati
                        if isinstance(inp, (list, tuple)):
                            out = func_timeout(test_timeout, candidate_func, args=inp)
                        else:
                            out = func_timeout(test_timeout, candidate_func, args=(inp,))
                        
                        if out == expected:
                            ok += 1
                        else:
                            fail += 1
                    except FunctionTimedOut:
                        fail += 1
                    except Exception:
                        fail += 1

            else: # data_format == 'mbpp' (basato su assert)
                for a in asserts:
                    try:
                        func_timeout(test_timeout, exec, args=(a, ns, ns))
                        ok += 1
                    except FunctionTimedOut:
                        print(f"Timeout test: {a}")
                        fail += 1
                    except AssertionError:
                        # print(f"   Assert fallito: {a}")
                        fail += 1
                    except Exception as e:
                        # print(f"   Errore runtime: {e}")
                        fail += 1

            # --- FASE 3: STATISTICHE ---
            total = ok + fail
            ratio = ok / total if total > 0 else 0
            prop_test_passed.append(round(ratio, 2))

            if ok > best_ok:
                best_ok = ok
                best_sol = sol

            if fail == 0 and total > 0:
                print(f"[SOLUTION {i}] ✅ All tests passed ({ok}/{total})")
                n_correct += 1
            else:
                print(f"[SOLUTION {i}] {fail}/{total} tests failed")
                pass

        except SyntaxError:
            print(f"[SOLUTION {i}] ERRORE SINTASSI: Il codice generato è malformato.")
        except Exception as e:
            print(f"[SOLUTION {i}] Errore generico durante il setup: {e}")
            continue

    return {"best_sol": best_sol, "c": n_correct, "prop_test_passed": prop_test_passed}



def test_model(test_split, model, tokenizer, n_solutions, data_format:str = "he", k: int = 1):

    # pass_at_k, tpr, best_sols_acc, times = [], [], [], []
    pass_at_k = []

    tot_c = []
    for row in range(len(test_split)):
        print(f"Testing Solution {row}")
        ex = test_split[row]
        solutions = generate_solutions(ex['prompt'], ex['entry_point'], model, tokenizer, n_solutions=n_solutions)
        perf = test_solutions(solutions, ex['entry_point'], ex['test'], data_format=data_format)
        tot_c.append({"task_id":ex['task_id'], 'best_solution':perf['best_sol'], 'c':perf['c']})
        # best_sols_acc.append(perf['best_sol_acc'])
        # n, c, t = perf['n'], perf['c'], perf['best_time']
        pass_at_k.append(passatk(n_solutions, perf['c'], k))
        # tpr.append(c / n if n else 0.0)
        # if t is not None:
        #     times.append(t)


    return {
        "generated_sample": datasets.from_list(tot_c),
        "avg_c": mean([c['c'] for c in tot_c]),
        "avg_pass_at_k": mean(pass_at_k)}
    #     {
    #     "best_sol_acc_avg": mean(best_sols_acc) if best_sols_acc else 0.0,
    #     "avg_TPR": mean(tpr) if tpr else 0.0,
    #     "avg_passatk": mean(pass_at_k) if pass_at_k else 0.0,
    #     "avg_time": mean(times) if times else None,
    # }


        