import re
import ast

def clean_generation(gen_text: str, prompt_text: str) -> str:
    """
    Pulisce la generazione rimuovendo il prompt e i blocchi markdown.
    """
    # 1. Rimuoviamo il prompt se è stato ripetuto all'inizio
    if gen_text.startswith(prompt_text):
        gen_text = gen_text[len(prompt_text):]
    
    # 2. Gestione Markdown (es. ```python ... ```)
    pattern_markdown = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern_markdown, gen_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return gen_text.strip()


def sanitize_code(code: str) -> str:
    """
    Usa l'AST per mantenere solo definizioni valide (Classi, Funzioni, Import).
    Rimuove test, print e codice incompleto alla fine.
    """
    if not code.strip():
        return ""

    # FASE 1: Gestione del codice tagliato (max tokens)
    # Proviamo a parsare. Se fallisce (SyntaxError), togliamo l'ultima riga e riproviamo.
    # Questo serve per il tuo 'Esempio 3' dove il loop si interrompe a metà.
    lines = code.split('\n')
    parsed_tree = None
    
    while lines:
        try:
            current_source = '\n'.join(lines)
            parsed_tree = ast.parse(current_source)
            break # Parsato con successo!
        except SyntaxError:
            lines.pop() # Rimuovi l'ultima riga (probabilmente incompleta) e riprova
    
    if parsed_tree is None:
        return "" # Non siamo riusciti a recuperare nulla di valido

    # FASE 2: Filtraggio dei nodi (Pulizia junk)
    # Teniamo solo Classi, Funzioni e Import. Buttiamo via le chiamate dirette (Expr).
    valid_code_blocks = []
    
    for node in parsed_tree.body:
        # ast.unparse ricostruisce il codice dal nodo (Disponibile da Python 3.9+)
        # Mantiene docstring e logica, rimuove commenti inutili e formattazione strana.
        
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Import, ast.ImportFrom)):
            try:
                valid_code_blocks.append(ast.unparse(node))
            except Exception:
                # Fallback per versioni python vecchie o casi strani: usiamo il source originale
                pass 
                
    return "\n\n".join(valid_code_blocks)


def generate_solutions(prompt: str,
                       entry_point:str,
                       model,
                       tokenizer,
                       temperature:float = 0.6,
                       max_new_tokens:int = 512, # Aumentato per evitare tagli prematuri
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
        # 1. Pulizia base (rimozione prompt/markdown)
        cleaned_text = clean_generation(sol, prompt)
        
        # 2. Sanitizzazione intelligente (AST)
        # Questa funzione rimuoverà automaticamente i test case, i print finali
        # e riparerà i loop tagliati a metà.
        sanitized_sol = sanitize_code(cleaned_text)
        
        # Se l'AST ha cancellato tutto (es. codice troppo rotto), 
        # teniamo il cleaned_text come fallback disperato, ma di solito sanitized è meglio.
        final_sol = sanitized_sol if sanitized_sol else cleaned_text
        
        final_solutions.append(final_sol)

    return final_solutions