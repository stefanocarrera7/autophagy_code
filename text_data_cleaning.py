from datasets import load_dataset
from huggingface_hub import login

login(token = 'xxx')

df = load_dataset("dgambettaphd/prompt_wxs_5000doc", split='train')

df = df.filter(lambda x: x["dataset"] == "S" and len(x["text"]) < 2000)

def transform_row(examples, idx):
    return {
        "task_id": f"task_{idx}",
        "entry_point": "",
        "prompt": examples["text"],
        "completion": "",
        "test": ""
    }

df = df.map(transform_row, with_indices=True)

df = df.remove_columns(['synt', 'dataset', 'id_doc', 'gen', 'text'])

df.push_to_hub('stefanocarrera/autophagy_D_text_S')