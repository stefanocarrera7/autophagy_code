# -*- coding: latin-1 -*-
import pandas as pd
import metrics
from post_processing import remove_markdown, light_cleanup
from eval import test_solutions
from statistics import mean

# Liste per il riepilogo finale
halstaed_avg = []
stats_per_gen = []

for g in range(11):
    print(f'\n======================================')
    print(f'    EVALUATING GENERATION [{g}]')
    print(f'======================================')
    
    try:
        # Carica il dataset della generazione corrente
        df = pd.read_parquet(f"hf://datasets/stefanocarrera/autophagycode_D_HE_meta-llama__Meta-Llama-3.1-8B-Instruct_gen{g}_TEST/data/train-00000-of-00001.parquet")
    except Exception as e:
        print(f"Errore nel download del gen{g}: {e}")
        continue

    # Contatori per tipologia di errore/successo
    executable_count = 0
    correct = 0
    syntax_errors = 0
    entry_point_errors = 0
    at_least_1_test = 0
    at_least_2_tests = 0
    
    voc, lng, vol, dif, eff = [], [], [], [], []

    # Iteriamo direttamente sulle righe del DataFrame corrente
    for j in range(len(df)):
        
        # 1. Estrazione e pulizia del codice
        raw_sol = str(df["completion"].iloc[j])
        sol = remove_markdown(raw_sol)
        sol = light_cleanup(sol)
        
        entry = str(df["entry_point"].iloc[j])
        test_data = str(df["test"].iloc[j])
        
        if test_data == "nan" or not test_data.strip():
            continue

        # 2. Esecuzione test
        res = test_solutions([sol], entry, test_data, "human_eval", verbose=False)
        
        # 3. Analisi di Eseguibilità (Codice formalmente valido)
        if res.get("prop_correct_defined", 0) > 0:
            executable_count += 1
            
        # 4. Analisi dei risultati per singola soluzione
        if not res['solutions_summary']:
            # Se la lista è vuota, capiamo il PERCHÉ leggendo gli errori
            if "EntryPointNotFound" in res.get("errors", []):
                entry_point_errors += 1
            else:
                syntax_errors += 1
        else:
            summary = res['solutions_summary'][0]
            ok_count = summary.get('ok', 0)
            fail_count = summary.get('fail', 0)
            
            if fail_count == 0 and ok_count > 0:
                correct += 1
            
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

    # Calcolo medie Halstead per la generazione
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