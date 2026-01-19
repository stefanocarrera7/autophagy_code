import re
import ast

def remove_markdown(text: str) -> str:
    """Rimuove i backticks del markdown se presenti."""
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def extract_clean_code(prompt: str, generation: str) -> str:
    """
    Combina prompt e generazione, parsa il codice, ed estrae
    solo la PRIMA class Solution valida e gli import.
    Ignora duplicati successivi della stessa classe.
    """
    
    # 1. Pulizia preliminare
    generation_clean = remove_markdown(generation)
    
    # 2. Ricostruzione del codice completo
    if generation_clean.strip().startswith(prompt.strip()):
        full_source = generation_clean
    else:
        full_source = prompt + "\n" + generation_clean

    # 3. Parsing Robusto (gestione del taglio da max_tokens)
    lines = full_source.split('\n')
    tree = None
    
    while lines:
        try:
            current_code = "\n".join(lines)
            tree = ast.parse(current_code)
            break 
        except SyntaxError:
            lines.pop() 
            
    if tree is None:
        return "" 

    # 4. Estrazione Selettiva (Logica "First Match")
    valid_blocks = []
    found_solution_class = False # Flag per tracciare se abbiamo già preso la classe
    
    # -- A. Prima passata per gli IMPORT (li vogliamo sempre in cima) --
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            valid_blocks.append(ast.unparse(node))

    # -- B. Seconda passata per CLASSI e FUNZIONI --
    for node in tree.body:
        # Se è una Classe
        if isinstance(node, ast.ClassDef):
            # Se è la classe "Solution"
            if node.name == "Solution":
                if found_solution_class:
                    continue # SKIP: Ne abbiamo già trovata una, questa è un duplicato!
                else:
                    valid_blocks.append(ast.unparse(node))
                    found_solution_class = True
            else:
                # Se è una classe helper (es. "ListNode", "TreeNode"), la teniamo
                valid_blocks.append(ast.unparse(node))
        
        # Se è una Funzione top-level (non dentro una classe)
        elif isinstance(node, ast.FunctionDef):
            # Le teniamo, a meno che non siano test (spesso i test hanno nomi come main o test_...)
            # Per sicurezza teniamo tutto ciò che è funzione se il prompt non chiedeva esplicitamente classi
            valid_blocks.append(ast.unparse(node))

    return "\n\n".join(valid_blocks)


def generate_solutions(prompt: str,
                       entry_point:str,
                       model,
                       tokenizer,
                       temperature:float = 0.2,
                       max_new_tokens:int = 512,
                       top_p = 0.9,
                       n_solutions: int = 1,
                       verbose : bool = False):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=n_solutions,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id 
    )
    
    raw_solutions = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    
    final_solutions = []
    for sol in raw_solutions:
        clean_code = extract_clean_code(prompt, sol)
        
        if not clean_code.strip():
             final_solutions.append(remove_markdown(sol))
        else:
             final_solutions.append(clean_code)

    return final_solutions