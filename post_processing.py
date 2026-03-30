# -*- coding: latin-1 -*-
import re
import pandas as pd

def remove_markdown(text: str) -> str:
    """
    Versione ottimizzata per la tesi: estrae il codice fermandosi 
    al primo segnale di chiusura (```) o di testo naturale (###, Explanation).
    """
    start_idx = 0
    if text.strip().startswith("```python"):
        start_idx = text.find("```python") + 9
    elif text.strip().startswith("```"):
        start_idx = text.find("```") + 3
    
    # Lavoriamo sulla parte che (teoricamente) contiene solo codice
    code_part = text[start_idx:].strip()

    stop_signals = [
        "```",             # Le backtick di chiusura (il tuo caso critico)
        "###",             # Titoli markdown per spiegazioni
    ]

    # Cerchiamo la posizione più vicina (minima) tra tutti i segnali di stop
    end_idx = len(code_part) # Di default, la fine è tutto il testo
    
    for signal in stop_signals:
        pos = code_part.find(signal)
        if pos != -1:
            # Se troviamo un segnale, aggiorniamo end_idx solo se è più vicino dell'attuale
            end_idx = min(end_idx, pos)

    # Tagliamo e puliamo
    final_code = code_part[:end_idx]
    
    return final_code.strip()

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
        


def remove_repetition(text:str, entry_point:str):
    """
    Remove repetition behaviour of LLMs by taking the first entry_point function
    !! Aggressive strategy that cut everrything after the second def starting from def entrty_point()
    """
    start_to_search = max(0, text.find(f'def {entry_point}('))

    end = text.find('def ', start_to_search + len(f'def {entry_point}('))
    if end > 0:
        text = text[:end]
    
    return text


def remove_check(text: str) -> str:
    check_idx = text.find('def check(')
    if check_idx != -1:
        return text[:check_idx]
    return text



def light_cleanup(code: str) -> str:
    """
    Rimuove firme di funzioni duplicate leggendo il codice riga per riga.
    Più sicuro e robusto di qualsiasi espressione regolare.
    """
    lines = code.split('\n')
    new_lines = []
    seen_defs = set()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Intercettiamo le firme delle funzioni
        if stripped.startswith("def "):
            # Estraiamo il nome esatto della funzione
            parts = stripped.split("def ", 1)[1]
            name = parts.split("(", 1)[0].strip()
            
            if name in seen_defs:
                # FIRMA DUPLICATA TROVATA!
                # Saltiamo questa riga e tutte le eventuali righe successive
                # fino alla fine della firma (i due punti ":")
                while i < len(lines) and not lines[i].strip().endswith(":"):
                    i += 1
                i += 1 # Salta l'ultima riga della firma
                continue
            else:
                # Nuova funzione, la salviamo in memoria
                seen_defs.add(name)
                new_lines.append(line)
                i += 1
                continue
                
        new_lines.append(line)
        i += 1
        
    return '\n'.join(new_lines)