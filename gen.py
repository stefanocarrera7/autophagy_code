import re

def prepend_prompt_imports(code: str, prompt_text: str) -> str:
    imps = re.findall(r'(?m)^(?:from\s+\S+\s+import\s+.*|import\s+.+)\s*$', prompt_text)
    return (("\n".join(imps) + "\n\n" + code).strip()) if (imps and code) else code or ""

def extract_exec_strict(gen_text: str, prompt_text: str) -> str:
    # nomi funzioni attese dal prompt (signature nel prompt)
    allowed = re.findall(r'(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(', prompt_text)
    if not allowed:
        return ""
    blocks = []
    for name in allowed:
        pat = re.compile(rf'(?ms)^\s*def\s+{re.escape(name)}\s*\(.*?(?=^\S|\Z)')
        m = pat.search(gen_text)
        if m:
            blocks.append(m.group(0).rstrip())
    return "\n\n".join(blocks).strip()


def generate_solutions(prompt: str,
                       entry_point:str,
                       model,
                       tokenizer,
                       temperature:float = 0.6,
                       max_new_tokens:int = 150,
                       top_p = 0.9,
                       n_solutions: int = 10,
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
    )
    # postprocess e ottenimento delle soluzioni come lista
    solutions = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

    # segment = extract_exec_strict(gen_text, prompt)
    # segment = prepend_prompt_imports(segment, prompt)

    for i in range(len(solutions)):
      # solutions[i] = prepend_prompt_imports(extract_exec_strict(solutions[i], prompt), prompt)

      ns = {}
      try:
        if solutions[i]:
          exec(solutions[i], ns, ns)
      except Exception:
        if verbose:
            print(f"Errore nel running di una soluzione per il prompt:\n {prompt}")

      if (entry_point not in ns or not callable(ns[entry_point])) and verbose:
         print(f"La funzione '{entry_point}' non é stata definita correttamente nel codice generato.")

    return solutions