import time
import ast
from statistics import mean
from metrics import passatk
from gen import generate_solutions



def test_solutions(solutions, test_data, entry_point):

    def extract_inputs_results(code_str, target_names=("inputs", "results")):
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



    td = extract_inputs_results(test_data)
    inputs_list = td.get("inputs", [])
    results_list = td.get("results", [])

    n = min(len(inputs_list), len(results_list))
    if n == 0:
        print("no test found, check test data")
        return None

    best_ok = 0
    best_fail = None
    best_sol = None
    best_time = None
    c = 0

    for sol in solutions:
        ns = {}
        try:
            exec(sol, ns, ns)
            candidate_func = ns[entry_point]
        except Exception:
            continue

        ok = fail = 0
        start = time.perf_counter()
        for inp, expected in zip(inputs_list[:n], results_list[:n]):
            try:
                out = candidate_func(*inp) if isinstance(inp, (list, tuple)) else candidate_func(inp)
                if out == expected:
                    ok += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
        elapsed = time.perf_counter() - start

        if ok >= best_ok:
            best_sol = sol
            best_ok = ok
            best_fail = fail
            if best_time is None or elapsed < best_time:
                best_time = elapsed

        if fail == 0:
            c += 1

    best_sol_acc = best_ok / (best_ok + best_fail)
    n = len(solutions)

    return {"best_sol": best_sol, "best_sol_acc": best_sol_acc, "n":n, "c":c, "best_time": best_time}





def test_model(test_split, model, tokenizer, n_solutions, k: int = 1):

    pass_at_k, tpr, best_sols_acc, times = [], [], [], []

    for row in range(len(test_split)):
        ex = test_split[row]
        solutions = generate_solutions(ex['prompt'], ex['entry_point'], model, tokenizer, n_solutions=n_solutions)
        perf = test_solutions(solutions, ex['test'], ex['entry_point'])
        if perf is None:  # sicurezza
            continue
        best_sols_acc.append(perf['best_sol_acc'])
        n, c, t = perf['n'], perf['c'], perf['best_time']
        pass_at_k.append(passatk(n, c, k))
        tpr.append(c / n if n else 0.0)
        if t is not None:
            times.append(t)

    return {
        "best_sol_acc_avg": mean(best_sols_acc) if best_sols_acc else 0.0,
        "avg_TPR": mean(tpr) if tpr else 0.0,
        "avg_passatk": mean(pass_at_k) if pass_at_k else 0.0,
        "avg_time": mean(times) if times else None,
    }