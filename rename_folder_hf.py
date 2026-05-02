from huggingface_hub import move_repo

# Inserisci qui il tuo token (assicurati che abbia i permessi di scrittura)
HF_TOKEN = "xxx"
USERNAME = "stefanocarrera" # Sostituisci se il tuo username è diverso

models = ['Qwen3-0.6B', 'Qwen3-4B', 'Qwen3-8B']

for model in models:
    for t in [0.2, 1]:
        for g in range(1,11):
            # MODEL
            old_repo = f"{USERNAME}/autophagycode_M_mercury_{model}_lr0.0001_c142_trust_t{t}_g{g}"
            new_repo = f"{USERNAME}/autophagycode_M_mercury_{model}_lr0.0001_c142_trust_t{t}_g{g}_run0"
            try:
                move_repo(from_id=old_repo, to_id=new_repo, repo_type="model", token=HF_TOKEN)
                print(f"✅ Rinominato: {old_repo} -> {new_repo}")
            except Exception as e:
                print(f"❌ Errore con {old_repo}: {e}")

            # HE
            old_repo = f"{USERNAME}/autophagycode_D_he_train-mercury_{model}_strategy_trust_t{t}_g{g}"
            new_repo = f"{USERNAME}/autophagycode_D_he_train-mercury_{model}_strategy_trust_t{t}_g{g}_run0"
            try:
                move_repo(from_id=old_repo, to_id=new_repo, repo_type="dataset", token=HF_TOKEN)
                print(f"✅ Rinominato: {old_repo} -> {new_repo}")
            except Exception as e:
                print(f"❌ Errore con {old_repo}: {e}")

            # TRAIN
            old_repo = f"{USERNAME}/autophagycode_D_mercury_{model}_lr0.0001_c142_trust_t{t}_g{g}"
            new_repo = f"{USERNAME}/autophagycode_D_mercury_{model}_lr0.0001_c142_trust_t{t}_g{g}_run0"
            try:
                move_repo(from_id=old_repo, to_id=new_repo, repo_type="dataset", token=HF_TOKEN)
                print(f"✅ Rinominato: {old_repo} -> {new_repo}")
            except Exception as e:
                print(f"❌ Errore con {old_repo}: {e}")

            # METRICS
            old_repo = f"{USERNAME}/autophagycode_D_he_train-mercury_{model}_strategy_trust_t{t}_g{g}_metrics"
            new_repo = f"{USERNAME}/autophagycode_D_he_train-mercury_{model}_strategy_trust_t{t}_g{g}_metrics_run0"
            try:
                move_repo(from_id=old_repo, to_id=new_repo, repo_type="dataset", token=HF_TOKEN)
                print(f"✅ Rinominato: {old_repo} -> {new_repo}")
            except Exception as e:
                print(f"❌ Errore con {old_repo}: {e}")
