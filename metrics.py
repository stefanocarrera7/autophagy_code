import time
import tokenize
import io
import ast
import math

def passatk(n:int, c:int, k:int):
  k = min(n, k)
  if k <= 0 or n <= 0: return 0.0
  if c <= 0: return 0.0
  if c >= n: return 1.0
  return 1.0 - (math.comb(n - c, k) / math.comb(n, k))






HALSTEAD_OPERATORS = {
    # Parole chiave di controllo e struttura
    "if", "elif", "else", "for", "while", "break", "continue", "return", "yield",
    "try", "except", "finally", "raise", "with", "as", "assert", "del", "pass",

    # Definizione e organizzazione
    "def", "class", "lambda", "import", "from", "global", "nonlocal",

    # Operatori aritmetici
    "+", "-", "*", "/", "//", "%", "**",

    # Operatori di confronto
    "==", "!=", ">", "<", ">=", "<=",

    # Operatori logici
    "and", "or", "not", "in", "is",

    # Operatori bitwise
    "&", "|", "^", "~", "<<", ">>",

    # Operatori di assegnamento
    "=", "+=", "-=", "*=", "/=", "//=", "%=", "**=", "&=", "|=", "^=", ">>=", "<<=",

    # Operatori di accesso / slicing / chiamata
    ".", "[", "]", "(", ")",  ":", ",", "}", "{",

    # Funzioni built-in considerate "operatori" (eseguono azioni)
    "print", "len", "range", "input", "open", "sum", "max", "min", "map", "filter", "zip", "sorted", "enumerate"
}


def h_vocavulary(n1, n2):
    return (n1 + n2)

def h_length(N1, N2):
    return (N1 + N2)

def h_volume(h_voc, h_len):
    return h_len * math.log2(h_voc)

def h_difficulty(n1, N1, n2):
    return (n1/2) * (N1/n2)

def h_effort(h_vol, h_diff):
    return h_vol * h_diff


def halstead_metrics(source):

    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    operators, operands = set(), set()
    N1 = N2 = 0

    for t in tokens:
        if t.type in (tokenize.OP, tokenize.NAME, tokenize.NUMBER):
            if t.string in HALSTEAD_OPERATORS:
                operators.add(t.string)
                N1 += 1
            else:
                operands.add(t.string)
                N2 += 1

    n1, n2 = len(operators), len(operands)

    if n1 == 0 or n2 == 0:
        raise ValueError("n1 == 0 or n2 == 0")
    
    vocabulary = h_vocavulary(n1, n2)
    length = h_length(N1, N2)
    volume = h_volume(vocabulary, length)
    difficulty = h_difficulty(n1, N1, n2)
    effort = h_effort(volume, difficulty)

    return {"vocabulary": vocabulary,
            "length": length,
            "volume": volume,
            "difficulty": difficulty,
            "effort": effort
            }


def cyclomatic_complexity_simplified(source: str) -> int:
    """
    Calcola la complessità ciclomatica (McCabe) approssimata per codice Python.
    Complexity = 1 + numero di punti decisionali (if, for, while, and/or, try/except, ecc.)
    """
    tree = ast.parse(source)
    complexity = 1  # valore minimo

    for node in ast.walk(tree):
        # punti di decisione
        if isinstance(node, (ast.If, ast.For, ast.While, ast.AsyncFor,
                             ast.With, ast.AsyncWith, ast.Try, ast.ExceptHandler,
                             ast.IfExp, ast.BoolOp, ast.comprehension, ast.Match)):
            complexity += 1

    return complexity


def loc(source: str) -> int:
    in_block = False
    count = 0

    for line in source.splitlines():
        i = 0
        has_code = False  # True if the line contains code (no comment)

        while i < len(line):
            if line.startswith('"""', i) or line.startswith("'''", i):
                in_block = not in_block
                i += 3
                continue

            if not in_block:
                ch = line[i]
                if ch == '#':
                    break
                if not ch.isspace():
                    has_code = True

            i += 1

        if has_code:
            count += 1

    return count



def perc_of_comments(source):
    count = 0
    flag = False
    for line in source.splitlines():
        if not line.strip():
            continue

        if line.strip().startswith('#'):
            count += 1
            continue

        if line.strip().startswith('"""') or line.strip().startswith("'''"):
            if flag:
                flag = False
                count += 1
            else: flag = True

        if flag == True:
            count += 1

        if line.strip().endswith('"""') or line.strip().endswith("'''") and not (line == "'''" or line == '"""'):
            if flag:
                flag = False

    return (count / loc(source))*100


def original_MI(source: str) -> float:
    V = halstead_metrics(source).get("volume")
    G = cyclomatic_complexity_simplified(source)
    L = loc(source)
    return 171 - 5.2 * math.log(V) - 0.23 * G - 16.2 * math.log(L)

def radon_MI(source: str) -> float:
    V = halstead_metrics(source).get("volume")
    G = cyclomatic_complexity_simplified(source)
    L = loc(source)
    C = perc_of_comments(source)
    return max(0, 100 * (171 - 5.2 * math.log(V) - 0.23 * G - 16.2 * math.log(L) + 50 * math.sin(math.sqrt(2.4 * C))) / 171)



def tpr(source: str, entry_point: str, test_inputs, expected_outputs) -> float:

    if len(test_inputs) != len(expected_outputs):
        raise ValueError("test_inputs ed expected_outputs must be of same length.")

    ns = {}
    exec(source, ns, ns)

    func = ns.get(entry_point)
    if func is None or not callable(func):
        raise ValueError(f"Function '{entry_point}' not found or not callable.")

    if not test_inputs:
        return 0.0  # nessun test -> TPR definito 0.0

    correct = 0

    for x, y_true in zip(test_inputs, expected_outputs):
        try:
            if isinstance(x, tuple):
                y_pred = func(*x)
            elif isinstance(x, dict):
                y_pred = func(**x)
            else:
                y_pred = func(x)
        except Exception:
            continue

        if y_pred == y_true:
            correct += 1

    tpr = correct / len(expected_outputs)
    return tpr




def mean_runtime(source: str, func_name: str, test_inputs) -> float:

    if not test_inputs:
        raise ValueError("test_inputs can't be empty")

    ns = {}
    exec(source, ns, ns)

    func = ns.get(func_name)
    if func is None or not callable(func):
        raise ValueError(f"Function '{func_name}' not found or not callable.")

    total_time = 0.0

    for x in test_inputs:
        start = time.perf_counter()

        try:
            if isinstance(x, tuple):
                func(*x)
            elif isinstance(x, dict):
                func(**x)
            else:
                func(x)
        except Exception:
            continue

        end = time.perf_counter()
        total_time += (end - start)

    # Tempo medio sulle esecuzioni riuscite
    return total_time / len(test_inputs)

