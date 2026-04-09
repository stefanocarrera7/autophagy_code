import pandas as pd
import json
import textwrap
from datasets import Dataset
from huggingface_hub import login
login('xxx')

def construct_prompt(starter_code: str, question: str) -> str:
    # 1. Troviamo l'indentazione presente alla fine dello starter_code
    # Prendiamo tutto ciò che segue l'ultimo carattere di "a capo" (\n)
    indent = starter_code.split('\n')[-1]
    
    # 2. Applichiamo la stessa indentazione a ogni riga della question
    # (Tranne la prima, che è già "posizionata" dal concatenamento)
    indented_question = textwrap.indent(question, indent).lstrip()
    
    # 3. Costruiamo il prompt
    prompt = f'{starter_code}"""\n{indent}{indented_question}\n{indent}"""\n\t'
    return prompt


def extract_asserts(input_output:str, entry_point:str) -> str:
    input = json.loads(input_output)['inputs']
    output = json.loads(input_output)['outputs']

    # abbiamo verificato che gli input sono tutte liste, queste liste contengono i parametri

    tests = ''
    for inp, out in zip(input, output):
        inp = str(inp)[1:-1]
        out = str(out)[1:-1] if type(out) == list else str(out)
        single_test = 'assert ' + entry_point + '(' + inp + ') == ' + out + '\n'
        tests = tests + single_test
    
    return tests

df = pd.read_json("hf://datasets/likaixin/TACO-verified/taco_verified.json")
df = df[df['difficulty']=='EASY']
df = df[df['starter_code']!='']
df = df[df['starter_code'].apply(lambda x: x.startswith('def '))]
df = df[df['question'].apply(lambda x: len(x)<2000)]
df['entry_point'] = df['starter_code'].apply(lambda x: x.split('(')[0][4:])
df['completion'] = df['solutions'].apply(lambda x: x[0])
df['test'] = df.apply(lambda row: extract_asserts(row['input_output'], row['entry_point']), axis=1)
df['prompt'] = df.apply(lambda row: construct_prompt(row['starter_code'], row['question']), axis=1)

df = df.rename(columns={
    'id': 'task_id'
})

df = df[['task_id', 'entry_point', 'prompt', 'completion', 'test']]


hf_dataset = Dataset.from_pandas(df)
hf_dataset.push_to_hub("stefanocarrera/autophagy_D_TACO")
