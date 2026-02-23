# -*- coding: latin-1 -*-
import re
import pandas as pd

def remove_markdown(text, target="```python"):
    testo_lower = text.lower()
    indice = testo_lower.find(target)
    
    if indice != -1:
        return text[indice + len(target):].strip()
    
    indice_fallback = text.find("```")
    if indice_fallback != -1:
        return text[indice_fallback + 3:].strip()

    return text.strip()

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