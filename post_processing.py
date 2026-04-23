# -*- coding: latin-1 -*-
import re
import pandas as pd


def find_idxs(text: str, pattern: str) -> list:
    """Find all occurrences of a pattern in a text and return their starting indices."""
    idxs = []
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx == -1:
            break
        idxs.append(idx)
        start = idx + len(pattern)
    return idxs

def remove_markdown2(text: str, prompt: str) -> str:

    len_prompt = len(prompt)
    backticks_idxs = find_idxs(text, "```")
    python_idxs = find_idxs(text, "```python")

    if python_idxs:
        for i in backticks_idxs:
            if len_prompt+5 < i < python_idxs[0]:
                return text[:i]
            if python_idxs[0] < i:
                candidate_func = text[python_idxs[0]+9:i]
                if 'def ' in candidate_func:
                    return candidate_func
    
    else:
        if backticks_idxs:
            return text[:backticks_idxs[0]]
    
    return text
        


import re

def remove_repetition(text: str, entry_point: str) -> str:
    """
    Remove repetition behaviour of LLMs keeping the entry_point function.
    Safely ignores nested functions by looking only at top-level definitions (no indentation).
    """
    pattern = r'^(?:@[^\n]+\n)*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    top_level_defs = list(re.finditer(pattern, text, flags=re.MULTILINE))
    
    # 1. Troviamo quale di queste funzioni top-level è il nostro entry_point
    entry_idx = -1
    for i, match in enumerate(top_level_defs):
        if match.group(1) == entry_point:
            entry_idx = i
            break
            
    if entry_idx == -1:
        return text  
        
    if entry_idx == len(top_level_defs) - 1:
        return text

    # 2. Identifichiamo la funzione successiva
    helper_match = top_level_defs[entry_idx + 1]
    helper_name = helper_match.group(1)
    helper_start = helper_match.start()
    
    # FIX: Se la funzione successiva ha lo stesso nome dell'entry point, è una copia. Tagliamo.
    if helper_name == entry_point:
        return text[:helper_start]
    
    # 3. Estraiamo il corpo dell'entry_point 
    entry_start = top_level_defs[entry_idx].start()
    entry_point_body = text[entry_start:helper_start]

    # 4. Verifichiamo se l'helper è chiamato nel corpo dell'entry_point
    if re.search(rf'\b{re.escape(helper_name)}\s*\(', entry_point_body):
        if entry_idx + 2 < len(top_level_defs):
            second_helper_start = top_level_defs[entry_idx + 2].start()
            return text[:second_helper_start]
        else:
            return text 
    else:
        return text[:helper_start]


def remove_check(text: str) -> str:
    """
    Rimuove la funzione di test 'check' tipica di HumanEval/MBPP.
    Ignora in modo sicuro eventuali funzioni annidate chiamate 'check' 
    cercando solo definizioni a livello globale (inizio riga).
    """
    pattern = r'^def\s+check\b\s*\('
    
    match = re.search(pattern, text, flags=re.MULTILINE)
    
    if match:
        # Tagliamo esattamente dove inizia il match (ovvero la lettera 'd' di def)
        return text[:match.start()]
        
    return text


def extract_functions(llm_output: str) -> str:
    """
    Estrae definizioni di funzioni multiple (comprese le helper) e import globali,
    ignorando la non-stop generation come test cases e testo discorsivo.
    """
    lines = llm_output.splitlines()
    extracted_lines = []
    inside_target_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # 1. Cattura import globali (spesso il LLM li inserisce in cima, prima delle helper)
        if not inside_target_block and (line.startswith('import ') or line.startswith('from ')):
            extracted_lines.append(line)
            continue
            
        # 2. Inizio di una funzione (o di un decoratore come @lru_cache)
        if line.startswith('def ') or line.startswith('@'):
            inside_target_block = True
            extracted_lines.append(line)
            continue
            
        # 3. Logica di cattura del corpo della funzione
        if inside_target_block:
            # Le righe vuote mantengono la formattazione
            if not stripped:
                extracted_lines.append(line)
            # Usa .isspace() per catturare \t, spazi normali e spazi \xa0
            elif line[0].isspace():
                extracted_lines.append(line)
            # I commenti a livello 0 non devono interrompere la funzione
            elif stripped.startswith('#'):
                extracted_lines.append(line)
            else:
                # Se troviamo testo a livello 0 (es. un altro def, o print), usciamo
                inside_target_block = False
                
    # Pulizia: rimuove eventuali righe vuote accumulate alla fine del codice estratto
    while extracted_lines and not extracted_lines[-1].strip():
        extracted_lines.pop()
        
    return '\n'.join(extracted_lines)