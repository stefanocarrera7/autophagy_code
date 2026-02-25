# -*- coding: latin-1 -*-
import pandas as pd
import metrics
from post_processing import remove_markdown, light_cleanup
from eval import test_solutions
from statistics import mean
from datasets import Dataset
from huggingface_hub import login

# --- CONFIGURAZIONE HUGGING FACE ---
# Inserisci il tuo username o l'organizzazione dove vuoi salvare i dataset
HF_USERNAME = "stefanocarrera" 
# Assicurati di aver fatto il login da terminale con `huggingface-cli login`
# oppure decommenta le due righe seguenti inserendo il tuo token:
# from huggingface_hub import login
login("xxxxx")
# -----------------------------------

# Liste per il riepilogo finale globale
halstaed_avg = []
stats_per_gen = []

for g in range(5):
    print(f'\n======================================')
    print(f'    EVALUATING GENERATION [{g}]')
    print(f'======================================')
    
    try:
        # Carica il dataset della generazione corrente
        df = pd.read_parquet(f"hf://datasets/stefanocarrera/autophagycode_D_unsloth__Qwen3-0.6B-Base-unsloth-bnb-4bit_lr0.0001_gen{g+1}/data/train-00000-of-00001.parquet")
    except Exception as e:
        print(f"Errore nel download del gen{g}: {e}")
        continue

    # Contatori per tipologia di errore/successo globali della gen
    executable_count = 0
    correct = 0
    syntax_errors = 0
    entry_point_errors = 0
    at_least_1_test = 0
    at_least_2_tests = 0
    
    voc, lng, vol, dif, eff = [], [], [], [], []
    
    # LISTA PER SALVARE I RISULTATI DI QUESTA GENERAZIONE
    generation_results = []

    # Iteriamo direttamente sulle righe del DataFrame corrente
    for j in range(len(df)):
        
        # 1. Estrazione e pulizia del codice
        raw_sol = str(df["completion"].iloc[j])
        sol = remove_markdown(raw_sol)
        sol = light_cleanup(sol)
        
        entry = str(df["entry_point"].iloc[j])
        test_data = str(df["test"].iloc[j])
        
        # Dizionario per memorizzare i risultati della singola riga
        row_metrics = {
            "task_index": j,
            "entry_point": entry,
            "is_executable": False,
            "is_correct": False,
            "tests_passed": 0,
            "tests_failed": 0,
            "error_type": None,
            "halstead_vocabulary": None,
            "halstead_length": None,
            "halstead_volume": None,
            "halstead_difficulty": None,
            "halstead_effort": None
        }

        if test_data == "nan" or not test_data.strip():
            row_metrics["error_type"] = "NoTestData"
            generation_results.append(row_metrics)
            continue

        # 2. Esecuzione test
        res = test_solutions([sol], entry, test_data, "human_eval", verbose=False)
        
        # 3. Analisi di Eseguibilità
        if res.get("prop_correct_defined", 0) > 0:
            executable_count += 1
            row_metrics["is_executable"] = True
            
        # 4. Analisi dei risultati per singola soluzione
        if not res['solutions_summary']:
            # Se la lista è vuota, capiamo il PERCHÉ leggendo gli errori
            if "EntryPointNotFound" in res.get("errors", []):
                entry_point_errors += 1
                row_metrics["error_type"] = "EntryPointNotFound"
            else:
                syntax_errors += 1
                row_metrics["error_type"] = "SyntaxError"
        else:
            summary = res['solutions_summary'][0]
            ok_count = summary.get('ok', 0)
            fail_count = summary.get('fail', 0)
            
            row_metrics["tests_passed"] = ok_count
            row_metrics["tests_failed"] = fail_count
            
            if fail_count == 0 and ok_count > 0:
                correct += 1
                row_metrics["is_correct"] = True
            
            if ok_count >= 1:
                at_least_1_test += 1
            if ok_count >= 2:
                at_least_2_tests += 1

        # 5. Metriche Halstead
        metrics_result = metrics.halstead_metrics(sol)
        if metrics_result:
            voc.append(metrics_result['vocabulary'])
            lng.append(metrics_result['length'])
            vol.append(metrics_result['volume'])
            dif.append(metrics_result['difficulty'])
            eff.append(metrics_result['effort'])
            
            # Salvataggio nella riga corrente
            row_metrics["halstead_vocabulary"] = metrics_result['vocabulary']
            row_metrics["halstead_length"] = metrics_result['length']
            row_metrics["halstead_volume"] = metrics_result['volume']
            row_metrics["halstead_difficulty"] = metrics_result['difficulty']
            row_metrics["halstead_effort"] = metrics_result['effort']
            
        # Aggiungiamo i risultati della riga alla lista
        generation_results.append(row_metrics)

    # --- SALVATAGGIO SU HUGGING FACE ---
    print(f"\n[INFO] Salvataggio metriche per la generazione {g} su Hugging Face...")
    try:
        # Convertiamo la lista di dizionari in un Dataset di HF
        hf_dataset = Dataset.from_list(generation_results)
        
        # Creiamo il nome del repository su HF (es: stefanocarrera/autophagy_metrics_gen0)
        repo_name = f"{HF_USERNAME}/D_metrics_HE_llama_3.1_8B_gen{g}"
        
        # Push al repository (crea o sovrascrive)
        hf_dataset.push_to_hub(repo_name)
        print(f"[OK] Dataset caricato con successo su: hf://datasets/{repo_name}")
    except Exception as e:
        print(f"[ERRORE] Impossibile caricare il dataset su HF per la gen {g}: {e}")

    # Calcolo medie Halstead per la generazione (solo per riepilogo terminale)
    current_metrics = None
    if voc:
        current_metrics = {
            'vol': mean(vol), 'dif': mean(dif), 'eff': mean(eff)
        }
        halstaed_avg.append(current_metrics)

    # Salvataggio statistiche della generazione
    gen_stats = {
        'gen': g,
        'total': len(df),
        'exec': executable_count,
        'correct': correct,
        'entry_err': entry_point_errors,
        'syntax_err': syntax_errors,
        'pass_1+': at_least_1_test,
        'pass_2+': at_least_2_tests
    }
    stats_per_gen.append(gen_stats)

    # --- REPORT GENERAZIONE ---
    print(f"\n---> ANALISI GENERAZIONE {g}:")
    print(f"  [EXEC]    Codice Eseguibile:      {executable_count} / {len(df)}")
    print(f"  [SUCCESS] Completamente corrette: {correct} / {len(df)}")
    print(f"  [PARTIAL] Passano >= 1 test:      {at_least_1_test}")
    print(f"  [PARTIAL] Passano >= 2 test:      {at_least_2_tests}")
    print(f"  [ERROR]   Nome Funzione Errato:   {entry_point_errors}")
    print(f"  [ERROR]   Sintassi/Setup fallito: {syntax_errors}")

# --- REPORT FINALE GLOBALE ---
print("\n\n" + "="*85)
print("                           RIEPILOGO FINALE AUTOFAGIA")
print("="*85)
print(f"{'Gen':<5} | {'Total':<6} | {'Exec':<6} | {'Correct':<8} | {'EntryErr':<9} | {'SyntErr':<8} | {'Pass 1+':<8} | {'Pass 2+':<8}")
print("-" * 85)
for s in stats_per_gen:
    print(f"{s['gen']:<5} | {s['total']:<6} | {s['exec']:<6} | {s['correct']:<8} | {s['entry_err']:<9} | {s['syntax_err']:<8} | {s['pass_1+']:<8} | {s['pass_2+']:<8}")