import re
import ast
import torch
from transformers import LogitsProcessor, LogitsProcessorList

class FP16OverflowClamper(LogitsProcessor):
    """
    Versione ultra-robusta per Tesla T4:
    1. Rimuove NaN e li sostituisce con valori neutri.
    2. Sostituisce Inf con i limiti massimi.
    3. Applica un clamp più stringente.
    """
    def __call__(self, input_ids, scores):
        # Sostituisce NaN con 0.0 e +/- Inf con +/- 100.0
        scores.nan_to_num_(nan=0.0, posinf=100.0, neginf=-100.0)
        # Clamp finale di sicurezza
        scores.clamp_(-100.0, 100.0)
        return scores
    
class Float32LogitsProcessor(LogitsProcessor):
    """ Forza i logit in float32 prima del sampling per stabilità """
    def __call__(self, input_ids, scores):
        return scores.to(torch.float32)


def generate_solutions(prompt: str,
                       entry_point:str,
                       model,
                       tokenizer,
                       temperature:float = 0.2,
                       max_new_tokens:int = 300,
                       top_p = 0.95,
                       n_solutions: int = 1):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    use_sampling = n_solutions > 1
    gen_temperature = temperature if use_sampling else None
    gen_top_p = top_p if use_sampling else None

    # Iniettiamo i salvavita in ordine:
    # 1. Pulizia dei valori folli (NaN/Inf)
    # 2. Casting a Float32 per calcoli precisi nel sampling
    processors = LogitsProcessorList([
        FP16OverflowClamper(),
        Float32LogitsProcessor()
    ])

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=gen_temperature,
        top_p=gen_top_p,
        do_sample=use_sampling,
        num_return_sequences=n_solutions,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=processors
    )
    
    raw_solutions = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    final_solutions = [x.strip() for x in raw_solutions]

    return final_solutions