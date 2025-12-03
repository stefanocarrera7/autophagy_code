import time
import ast
from statistics import mean
from metrics import passatk
from gen import generate_solutions
import datasets

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



def test_solutions(solutions, entry_point, test_data, data_format:str = "he"):

    if data_format == 'he':
        td = extract_inputs_results(test_data)
        inputs_list = td.get("inputs", [])
        results_list = td.get("results", [])

        n = min(len(inputs_list), len(results_list))
        if n == 0:
            print("no test found, check test data")
            return None

    n_correct = 0
    prop_test_passed = []
    for i in range(len(solutions)):
        sol = solutions[i]
        ns = {}
        try:
            exec(sol, ns, ns)
            candidate_func = ns[entry_point]

            if data_format == 'he':
                ok = fail = 0
                for inp, expected in zip(inputs_list[:n], results_list[:n]):
                    try:
                        out = candidate_func(*inp) if isinstance(inp, (list, tuple)) else candidate_func(inp)
                        if out == expected:
                            ok += 1
                        else:
                            fail += 1
                    except Exception:
                        print(f"Problem on execution of the function {entry_point} on input:\n {inp} and output:\n {out}")
                if fail == 0:
                    print(f"[SOLUTION {i}] All tests passed")
                    n_correct += 1
                else:
                    print(f"[SOLUTION {i}] {fail} / {fail + ok} tests not passed \n Solution: \n {sol}")
                prop_test_passed.append(ok/(ok+fail))

            else:
                asserts = [line.strip() for line in test_data.split("\n") if line.strip()]
                t_correct = 0
                for a in asserts:
                    try:
                        exec(a, ns, ns)
                        t_correct += 1
                    except:
                        continue
                if t_correct == len(asserts):
                    n_correct += 1
                prop_test_passed.append(t_correct/len(asserts))

        except:
            print(f"Generated solution not executable:\n{sol}")
            continue


    return {"c": n_correct, "prop_test_passed": prop_test_passed}




def test_model(test_split, model, tokenizer, n_solutions, data_format:str = "he", k: int = 1):

    # pass_at_k, tpr, best_sols_acc, times = [], [], [], []
    pass_at_k = []

    tot_c = []
    for row in range(len(test_split)):
        ex = test_split[row]
        solutions = generate_solutions(ex['prompt'], ex['entry_point'], model, tokenizer, n_solutions=n_solutions)
        perf = test_solutions(solutions, ex['test'], ex['entry_point'], data_format=data_format)
        tot_c.append({"task_id":ex['task_id'], 'c':perf['c']})
        # best_sols_acc.append(perf['best_sol_acc'])
        # n, c, t = perf['n'], perf['c'], perf['best_time']
        pass_at_k.append(passatk(n_solutions, perf['c'], k))
        # tpr.append(c / n if n else 0.0)
        # if t is not None:
        #     times.append(t)


    return {
        "correct_solutions_list": datasets.from_list(tot_c),
        "avg_c": mean([c['c'] for c in tot_c]),
        "avg_pass_at_k": mean(pass_at_k)}
    #     {
    #     "best_sol_acc_avg": mean(best_sols_acc) if best_sols_acc else 0.0,
    #     "avg_TPR": mean(tpr) if tpr else 0.0,
    #     "avg_passatk": mean(pass_at_k) if pass_at_k else 0.0,
    #     "avg_time": mean(times) if times else None,
    # }


        