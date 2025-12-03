from datasets import load_dataset
import re
from datasets import Dataset
from huggingface_hub import login

HF_TOKEN = ""
login(token=HF_TOKEN)

def make_executable_assert(assert_list):
    """ Make assert statements of MBPP executable """
    norm = []
    for s in assert_list:
        s = s.strip()
        norm.append(s)
    return "\n".join(norm) + "\n"  # executable code

def last_def_name(cell: str):
    """Ritorna il nome dell'ultima funzione definita nella cella, oppure None.
        Utile per trovare l'entry point
    """
    matches = re.findall(r'(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(', cell)
    return matches[-1] if matches else None



def mbpp_to_he():
    mbpp = load_dataset("google-research-datasets/mbpp", split="test")
    he = {}

    he['task_id'] = mbpp['task_id']
    he['canonical_solution'] = mbpp['code']

    # tests
    tests = []
    for i in range(len(mbpp['test_list'])):
        tests.append(mbpp['test_list'][i] + mbpp['challenge_test_list'][i])

    he['test'] = [make_executable_assert(t) for t in tests]


    # prompts
    prompts = []

    pattern_def = r'(?m)^\s*def\s+[A-Za-z_]\w*\s*\([^)]*\)\s*:'
    pattern_import = r'(?m)^\s*(?:from\s+\S+\s+import|import\s+\S+)'

    for t, c in zip(mbpp['text'], mbpp['code']):
        righe_import = re.findall(pattern_import, c)
        righe_def = re.findall(pattern_def, c)

        if righe_def:
            # tutte le def, docstring solo sotto l’ultima
            lines = righe_import + righe_def[:-1] + [f"{righe_def[-1]}\n    '''{t}'''"]
            prompt = "\n".join(lines)
        else:
            lines = righe_import + [f"'''{t}'''"]
            prompt = "\n".join(lines)

        prompts.append(prompt)

    he['prompt'] = prompts

    he['entry_point'] = [last_def_name(p) for p in prompts]
    
    return Dataset.from_dict(he)

he = mbpp_to_he()
he.push_to_hub("stefanocarrera7/autophagy_mbpp_to_he")
