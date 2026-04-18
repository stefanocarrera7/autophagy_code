# import
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login
login('xxx')

col_to_del = ['task_id', '__index_level_0__']

# load he
he = load_dataset("openai/openai_humaneval", split="test")
# change canonical solution to completion
he = he.rename_column("canonical_solution", "completion")
he = he.remove_columns([c for c in col_to_del if c in he.column_names])

# load taco
taco = load_dataset("stefanocarrera/autophagy_D_TACO", split="train")
taco = taco.remove_columns([c for c in col_to_del if c in taco.column_names])

# load mercury
mercury = load_dataset("stefanocarrera/autophagy_D_mercury", split="train")
mercury = mercury.remove_columns([c for c in col_to_del if c in mercury.column_names])

# Aggiungere la fonte dei dati
he = he.add_column("source", ["he"] * len(he))
taco = taco.add_column("source", ["taco"] * len(taco))
mercury = mercury.add_column("source", ["mercury"] * len(mercury))

# merging
merged = concatenate_datasets([he, taco, mercury])

new_id = range(len(merged))
merged = merged.add_column("task_id", new_id)

merged = merged.shuffle(seed = 71)

# pushing to hub to a train/test split
merged_split = merged.train_test_split(test_size=0.1, seed=17)

merged_split.push_to_hub("stefanocarrera/autophagy_D_all")