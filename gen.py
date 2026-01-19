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
    Combina prompt e generazione, parsa il codice, ed estrae solo
    la classe Solution, le funzioni e gli import.
    Elimina chirurgicamente test, print e codice 'morto' alla fine.
    """
    
    # 1. Pulizia preliminare della generazione
    generation_clean = remove_markdown(generation)
    
    # 2. Ricostruzione del codice completo
    # Se il modello ha ripetuto il prompt, usiamo solo la generazione.
    # Altrimenti concateniamo Prompt + Generazione per avere un codice sintatticamente valido.
    if generation_clean.strip().startswith(prompt.strip()):
        full_source = generation_clean
    else:
        # Aggiungiamo newline per sicurezza
        full_source = prompt + "\n" + generation_clean

    # 3. Parsing Robusto (gestione del taglio da max_tokens)
    # Se c'è un errore di sintassi (es. codice tagliato alla fine), 
    # togliamo l'ultima riga e riproviamo finché non compila.
    lines = full_source.split('\n')
    tree = None
    
    # Tentativi di parsing riducendo il file dal fondo
    while lines:
        try:
            current_code = "\n".join(lines)
            tree = ast.parse(current_code)
            break # Successo!
        except SyntaxError:
            lines.pop() # Rimuovi l'ultima riga problematica
            
    if tree is None:
        return "" # Non siamo riusciti a recuperare nulla di valido

    # 4. Estrazione Selettiva (Chirurgia)
    # Teniamo solo: Classi, Funzioni, Import.
    # Buttiamo via: Expr (print, chiamate), Assign (variabili globali di test), ecc.
    valid_blocks = []
    
    # Cerca prima gli import per metterli in cima
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            valid_blocks.append(ast.unparse(node))

    # Cerca la classe Solution o funzioni
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            valid_blocks.append(ast.unparse(node))
        elif isinstance(node, ast.FunctionDef):
            # Accettiamo funzioni top-level solo se non sono dentro una classe Solution
            # (utile se il prompt non chiedeva una classe ma una funzione sciolta)
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
        # Usiamo la nuova funzione di estrazione chirurgica
        clean_code = extract_clean_code(prompt, sol)
        
        # Fallback: se la pulizia fallisce (molto raro), restituiamo la raw string pulita dal markdown
        if not clean_code.strip():
             final_solutions.append(remove_markdown(sol))
        else:
             final_solutions.append(clean_code)

    return final_solutions