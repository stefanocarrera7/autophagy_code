import re
import ast

def remove_markdown(text: str) -> str:
    """Rimuove i backticks del markdown se presenti."""
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def get_calls_in_function(func_node: ast.FunctionDef) -> set[str]:
    """
    Analizza il corpo di una funzione per trovare i nomi di altri metodi chiamati.
    Cerca pattern come 'self.method_name()' o 'method_name()' (se statico/annidato).
    """
    called_names = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            # Caso: self.metodo(...)
            if isinstance(node.func, ast.Attribute) and \
               isinstance(node.func.value, ast.Name) and \
               node.func.value.id == 'self':
                called_names.add(node.func.attr)
            # Caso: metodo(...) - raro in classi, ma possibile per funzioni annidate o globali
            elif isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
    return called_names

def extract_clean_code(prompt: str, generation: str, entry_point: str) -> str:
    """
    Combina prompt e generazione.
    Usa AST per:
    1. Trovare la class Solution.
    2. Identificare la funzione 'entry_point'.
    3. Mantenere SOLO l'entry_point e le funzioni helper che esso chiama.
    4. Rimuovere duplicati o funzioni allucinate non usate.
    """
    
    # 1. Preparazione
    generation_clean = remove_markdown(generation)
    
    # Unione Prompt + Generazione
    if generation_clean.strip().startswith(prompt.strip()):
        full_source = generation_clean
    else:
        full_source = prompt + "\n" + generation_clean

    # 2. Parsing con gestione errori (taglio dal fondo)
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

    valid_blocks = []
    
    # --- A. IMPORT (Sempre in cima) ---
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            valid_blocks.append(ast.unparse(node))

    # --- B. CLASSI (Logica Smart Entry Point) ---
    found_solution_class = False
    
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # Gestione specifica per class Solution
            if node.name == "Solution":
                if found_solution_class: 
                    continue # Ignora classi Solution duplicate
                
                found_solution_class = True
                
                # Dizionario dei metodi disponibili nella classe
                methods = {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
                
                # Set dei metodi da mantenere
                methods_to_keep = set()
                
                # 1. Cerchiamo l'entry point
                if entry_point in methods:
                    queue = [entry_point]
                    methods_to_keep.add(entry_point)
                    
                    # 2. Dependency Walking: Troviamo tutte le funzioni helper usate
                    while queue:
                        current_method_name = queue.pop(0)
                        current_method_node = methods[current_method_name]
                        
                        # Trova chi chiama questo metodo
                        called_funcs = get_calls_in_function(current_method_node)
                        
                        for called in called_funcs:
                            # Se la funzione chiamata esiste nella classe e non l'abbiamo ancora processata
                            if called in methods and called not in methods_to_keep:
                                methods_to_keep.add(called)
                                queue.append(called)
                
                else:
                    # FALLBACK: Se l'entry point non c'è (nome sbagliato dal modello?), 
                    # manteniamo il primo metodo e speriamo bene, oppure tutti.
                    # Per pulizia, prendiamo il primo.
                    if methods:
                        first_method = list(methods.keys())[0]
                        methods_to_keep.add(first_method)

                # 3. Ricostruiamo il corpo della classe
                new_body = []
                # Manteniamo docstrings o assegnazioni (es. costanti di classe)
                for item in node.body:
                    if not isinstance(item, ast.FunctionDef):
                        new_body.append(item)
                    elif item.name in methods_to_keep:
                        new_body.append(item)
                
                node.body = new_body
                valid_blocks.append(ast.unparse(node))
            
            else:
                # Altre classi (es. TreeNode, ListNode) le teniamo così come sono
                valid_blocks.append(ast.unparse(node))
                
        # Funzioni Top-Level (se non c'è classe)
        elif isinstance(node, ast.FunctionDef):
             # Se il prompt non chiedeva una classe, controlliamo l'entry point anche qui
             if node.name == entry_point or (entry_point not in [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]):
                 valid_blocks.append(ast.unparse(node))

    return "\n\n".join(valid_blocks)


def generate_solutions(prompt: str,
                       entry_point:str,
                       model,
                       tokenizer,
                       temperature:float = 0.6,
                       max_new_tokens:int = 512,
                       top_p = 0.9,
                       n_solutions: int = 1):

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
        # Passiamo l'entry_point per il filtraggio intelligente
        clean_code = extract_clean_code(prompt, sol, entry_point)
        
        if not clean_code.strip():
             final_solutions.append(remove_markdown(sol))
        else:
             final_solutions.append(clean_code)

    return final_solutions