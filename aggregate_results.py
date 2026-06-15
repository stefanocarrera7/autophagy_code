import pandas as pd
import os
from datasets import Dataset
from huggingface_hub import login

HF_TOKEN = os.getenv("token_write")
login(token=HF_TOKEN)


models = ['Qwen3-4B', 'Qwen3-8B']
temperatures = ['0.2', '0.5', '0.75', '1.0', '1.1', '1.25', '1.5']
generations = range(1, 12)
runs = [0, 1, 2] 

all_data = []
print("Scaricamento dei dati in corso con Pandas...")


for model in models:
    for temp in temperatures:
        for r in runs:
            for g in generations:
                repo_name = f"stefanocarrera/autophagycode_D_he_train-mercury_{model}_strategy_trust_t{temp}_g{g}_run{r}_metrics"
                
                try:
                    parquet_url = f"hf://datasets/{repo_name}/data/train-00000-of-00001.parquet"
                    df_gen = pd.read_parquet(parquet_url)
                    

                    df_gen['generation'] = g
                    df_gen['model'] = model
                    df_gen['temperature'] = temp
                    df_gen['run'] = r 
                    df_gen['experiment'] = f"{model} (T={temp})"
                    
                    all_data.append(df_gen)
                    print(f"[{model} - T={temp} - Run {r}] Gen {g} caricata!")
                except Exception as e:
                    print(f"Errore nel caricamento di [{model} - T={temp} - Run {r}] gen {g}: {e}")
    print(f"Temperature {temp} scaricata")


if all_data:
    df_all = pd.concat(all_data, ignore_index=True)
    print("\nDataset completo creato! Dimensioni:", df_all.shape)
else:
    print("\nErrore: Nessun dato è stato caricato. Verifica i nomi dei repo.")
    exit()

df_all_to_push = Dataset.from_pandas(df_all)
df_all_to_push.push_to_hub("stefanocarrera/autophagycode_D_results-per-task_4-8B_02-1.5t_3run")

# 4. Aggregazione delle metriche
metriche_da_aggregare = {
    'is_executable': 'mean',      
    'is_correct': 'mean',           
    'entry_point_repeated': 'mean',  
    'tests_passed': 'mean',           
    'tests_failed': 'mean',           
    'test_run_time_ms': 'median',     
    'halstead_vocabulary': 'mean',       
    'halstead_length': 'mean', 
    'halstead_volume': 'mean', 
    'halstead_difficulty': 'mean',       
    'halstead_effort': 'mean', 
    'halstead_time': 'mean',
    'cyclomatic_complexity': 'mean',
    'maintainability_index': 'mean',
    'loc': 'mean',
    'sloc': 'mean',
    'comment_percentage': 'mean',
    "shannon_entropy": 'mean',
    'mean_predictive_entropy': 'mean',
    'max_predictive_entropy': 'mean',
    'TTR': 'mean',       
    'n_func_defined': 'mean'
}

# Filtriamo solo le colonne che esistono
agg_dict = {col: op for col, op in metriche_da_aggregare.items() if col in df_all.columns}

summary_df = df_all.groupby(['model', 'temperature', 'experiment', 'generation', 'run']).agg(agg_dict).reset_index()
summary_df = summary_df.round(4)

summary_df_to_push = Dataset.from_pandas(summary_df)
summary_df_to_push.push_to_hub("stefanocarrera/autophagycode_D_results-agg_4-8B_02-1.5t_3run")