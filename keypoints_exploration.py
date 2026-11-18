import json
from datasets import Dataset, load_dataset
import numpy as np
from transformers import AutoTokenizer

# 1. Definizione degli operatori di Halstead come token BPE
OP_CONTROL_DECL = {
    "def",
    "Ġdef",
    "class",
    "Ġclass",
    "if",
    "Ġif",
    "elif",
    "Ġelif",
    "else",
    "Ġelse",
    "else:",
    "Ġelse:",
    "for",
    "Ġfor",
    "while",
    "Ġwhile",
    "return",
    "Ġreturn",
    "yield",
    "Ġyield",
    "break",
    "Ġbreak",
    "continue",
    "Ġcontinue",
    "pass",
    "Ġpass",
    "try",
    "Ġtry",
    "try:",
    "Ġtry:",
    "except",
    "Ġexcept",
    "except:",
    "Ġexcept:",
    "finally",
    "Ġfinally",
    "finally:",
    "Ġfinally:",
    "raise",
    "Ġraise",
    "with",
    "Ġwith",
    "as",
    "Ġas",
    "import",
    "Ġimport",
    "from",
    "Ġfrom",
    "lambda",
    "Ġlambda",
    "assert",
    "Ġassert",
    "del",
    "Ġdel",
    "global",
    "Ġglobal",
    "nonlocal",
    "Ġnonlocal",
    "async",
    "Ġasync",
    "await",
    "Ġawait",
}

OP_WORD = {"Ġand", "Ġor", "Ġnot", "Ġin", "Ġis"}

OP_SYMBOLIC = {
    "=",
    "Ġ=",
    "+",
    "Ġ+",
    "-",
    "Ġ-",
    "*",
    "Ġ*",
    "/",
    "Ġ/",
    "//",
    "Ġ//",
    "%",
    "Ġ%",
    "**",
    "Ġ**",
    "@",
    "Ġ@",
    "+=",
    "Ġ+=",
    "-=",
    "Ġ-=",
    "*=",
    "Ġ*=",
    "/=",
    "Ġ/=",
    "//=",
    "Ġ//=",
    "%=",
    "Ġ%=",
    "**=",
    "Ġ**=",
    ":=",
    "Ġ:=",
    "==",
    "Ġ==",
    "!=",
    "Ġ!=",
    "<",
    "Ġ<",
    ">",
    "Ġ>",
    "<=",
    "Ġ<=",
    ">=",
    "Ġ>=",
    "&",
    "Ġ&",
    "|",
    "Ġ|",
    "^",
    "Ġ^",
    "~",
    "Ġ~",
    "<<",
    "Ġ<<",
    ">>",
    "Ġ>>",
    "->",
    "Ġ->",
}

OP_PUNCTUATION = {
    "(",
    "Ġ(",
    ")",
    "[",
    "Ġ[",
    "]",
    "{",
    "Ġ{",
    "}",
    ":",
    ",",
    ".",
    ";",
    "):Ċ",
    ":Ċ",
    "),",
    "]:",
    "():",
}

HALSTEAD_OPERATORS = OP_CONTROL_DECL | OP_WORD | OP_SYMBOLIC | OP_PUNCTUATION

INDIVIDUAL_TARGET_TOKENS = [
    "Ġreturn",
    "Ġfor",
    "Ġif",
    "):Ċ",
    ":Ċ",
    "Ġ=",
]


def sol_to_completion(sol: str, prompt: str) -> str:
  """Rimuove solo il prompt esatto senza fare lstrip(), allineando 1:1 con top_k_progression."""
  if sol.startswith(prompt):
    sol = sol[len(prompt) :]
  return sol


def token_to_step(
    completion: str, tokenizer, top_k_dict: dict | None = None
) -> dict[int, str]:
  """Converte la completion in un dizionario {i: token} con indice intero."""
  tok_list = tokenizer.convert_ids_to_tokens(
      tokenizer(completion, add_special_tokens=False)["input_ids"]
  )
  if top_k_dict is not None and len(top_k_dict) == len(tok_list) + 1:
    tok_list.append(tokenizer.eos_token)

  return {i: tok for i, tok in enumerate(tok_list)}


def _safe_mean(values: list[float]) -> float:
  return float(np.mean(values)) if values else 0.0


models = ["Qwen3-4B", "Qwen3-8B"]
temperatures = [0.2, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5]
generations = range(11)
runs = [0, 1, 2]

tokenizer = AutoTokenizer.from_pretrained(
    "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit"
)

records = []

LABELS = [
    "__ALL_OPERATORS__",
    "__NON_OPERATORS__",
    "__ALL_TOKENS__",
    "__OP_CONTROL_DECL__",
    "__OP_WORD__",
    "__OP_SYMBOLIC__",
    "__OP_PUNCTUATION__",
    *INDIVIDUAL_TARGET_TOKENS,
]

for model in models:
  for t in temperatures:
    for g in generations:
      for r in runs:
        data_id = f"stefanocarrera/autophagycode_D_he_{model}_t{t}_g{g}_run{r}"
        ds = load_dataset(data_id, split="train")

        # Accumulatori di macro-medie (una media per soluzione) e conteggio token reali
        buckets = {label: [] for label in LABELS}
        token_counts = {label: 0 for label in LABELS}

        for row in ds:
          completion = sol_to_completion(row["completion"], row["prompt"])
          top_k_dict = json.loads(row["top_k_progression"])
          step_dict = token_to_step(completion, tokenizer, top_k_dict)

          row_buckets = {label: [] for label in LABELS}

          # 1. __ALL_TOKENS__: itera direttamente sugli step salvati (escludendo step_0)
          # Garantisce l'inclusione esatta di EOS senza dipendere dal re-encoding BPE
          for step_key, step_info in top_k_dict.items():
            if step_key == "step_0":
              continue
            if step_info is not None and "pred_entropy" in step_info:
              row_buckets["__ALL_TOKENS__"].append(step_info["pred_entropy"])

          # 2. Categorie di operatori (condizionate al token al passo idx -> entropia a idx+1)
          for idx, token in step_dict.items():
            next_info = top_k_dict.get(f"step_{idx + 1}")
            if next_info is None or "pred_entropy" not in next_info:
              continue

            val_next = next_info["pred_entropy"]

            if token in HALSTEAD_OPERATORS:
              row_buckets["__ALL_OPERATORS__"].append(val_next)
              if token in OP_CONTROL_DECL:
                row_buckets["__OP_CONTROL_DECL__"].append(val_next)
              elif token in OP_WORD:
                row_buckets["__OP_WORD__"].append(val_next)
              elif token in OP_SYMBOLIC:
                row_buckets["__OP_SYMBOLIC__"].append(val_next)
              elif token in OP_PUNCTUATION:
                row_buckets["__OP_PUNCTUATION__"].append(val_next)
            else:
              row_buckets["__NON_OPERATORS__"].append(val_next)

            if token in INDIVIDUAL_TARGET_TOKENS:
              row_buckets[token].append(val_next)

          # Aggrega per singola soluzione (macro-average) solo se la categoria è presente nella row
          for label, vals in row_buckets.items():
            if vals:
              buckets[label].append(_safe_mean(vals))
              token_counts[label] += len(vals)

        for label in LABELS:
          records.append({
              "token": label,
              "model": model,
              "temperature": t,
              "gen": g,
              "run": r,
              "n_solutions": len(
                  buckets[label]
              ),  # quante soluzioni contenevano quel token/categoria
              "n_token": token_counts[label],  # totale occorrenze del token
              "average_entropy": _safe_mean(buckets[label]),
          })

    print(f"COMPLETED MODEL [{model}], TEMPERATURE [{t}]")

final_ds = Dataset.from_list(records)
repo_id = "stefanocarrera/autophagy_code_D_results_entropy_per_next_token_macro_avg"
final_ds.push_to_hub(repo_id)