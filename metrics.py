import math
import tempfile
import subprocess
import json

def passatk(n:int, c:int, k:int):
  k = min(n, k)
  if k <= 0 or n <= 0: return 0.0
  if c <= 0: return 0.0
  if c >= n: return 1.0
  return 1.0 - (math.comb(n - c, k) / math.comb(n, k))



def ttr(code: str, tokenizer) -> float:
    if not str(code).strip():
        return 0.0
        
    tokens = tokenizer.encode(str(code), add_special_tokens=False)
    total_tokens = len(tokens)
    
    if total_tokens == 0:
        return 0.0
        
    unique_tokens = len(set(tokens))
    
    return unique_tokens / total_tokens


def token_dictionary(code: str, tokenizer) -> dict:
    token_ids = tokenizer.encode(str(code), add_special_tokens=False)
    
    text_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    token_freq = {}
    for t_text in text_tokens:

        t_str = str(t_text)
        token_freq[t_str] = token_freq.get(t_str, 0) + 1
        
    return token_freq


def get_multimetric_from_string(code_string: str):
    """
    Prende una stringa di codice Python, la salva in un file temporaneo,
    esegue multimetric e restituisce un dizionario con tutte le metriche.
    """
    # 1. Crea un file temporaneo che si elimina da solo quando viene chiuso (delete=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as temp:
        # Scrive il codice della tua stringa nel file
        temp.write(code_string)
        temp.flush()  # Forza la scrittura sul disco prima di chiamare multimetric
        
        try:
            # 2. Esegue il comando bash: multimetric /tmp/nomefile_temporaneo.py
            result = subprocess.run(
                ['multimetric', temp.name], 
                capture_output=True,  # Cattura l'output invece di stamparlo a schermo
                text=True,            # Restituisce una stringa anziché byte
                check=True
            )
            
            # 3. Multimetric restituisce i dati in formato JSON, quindi li decodifichiamo
            data = json.loads(result.stdout)
            
            # Le metriche globali di tutto il file si trovano sotto la chiave 'overall'
            return data.get('overall', {})
            
        except subprocess.CalledProcessError as e:
            # Cattura errori se multimetric fallisce (es. se il codice è sintatticamente troppo rotto)
            # e restituisce None in modo da non bloccare il tuo ciclo for
            return None
        except json.JSONDecodeError:
            # Sicurezza nel caso l'output non sia un JSON valido
            return None