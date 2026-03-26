from datasets import load_dataset
from huggingface_hub import login

login(token = 'xxx')

df = load_dataset("dgambettaphd/prompt_wxs_5000doc", split='train')

df = df.filter(lambda x: x["dataset"] == "S" and len(x["text"]) < 2000)

df = df.remove_columns(['synt', 'dataset', 'id_doc', 'gen'])

df.push_to_hub('stefanocarrera/autophagy_D_text_S')