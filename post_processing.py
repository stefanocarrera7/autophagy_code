# -*- coding: latin-1 -*-
import re
import pandas as pd

def remove_markdown(text: str) -> str:
    """
    Rimuove tutto ciò che c'è prima di ```python (incluso) e tutto ciò 
    che c'è dopo il ``` di chiusura (se il modello si è ricordato di metterlo).
    """
    # Usiamo lower() solo per trovare l'indice, in caso scriva ```Python
    text_lower = text.lower()
    
    # 1. Trova l'apertura e taglia via il prima
    if "```python" in text_lower:
        start_idx = text_lower.find("```python")
        text = text[start_idx + 9:]  # Taglia tutto fino alla fine di "```python"
    elif "```" in text:
        start_idx = text.find("```")
        text = text[start_idx + 3:]  # Taglia tutto fino alla fine di "```"

    # 2. Se ha messo una chiusura, taglia via tutto il testo discorsivo dopo
    if "```" in text:
        end_idx = text.find("```")
        text = text[:end_idx]
        
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