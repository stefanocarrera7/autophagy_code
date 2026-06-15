from huggingface_hub import move_repo

# Inserisci qui il tuo token (assicurati che abbia i permessi di scrittura)
HF_TOKEN = "xxx"
USERNAME = "stefanocarrera" # Sostituisci se il tuo username è diverso

models = ['Qwen3-4B']

for model in models:
    for r in [0,1,2]:
        # for t in [0.2,1]:
        for g in range(1,11):
            # MODEL
            old_repo = f"stefanocarrera/autophagycode_M_mercury_{model}_lr0.0001_c142_trust_t1_g{g}_run{r}"
            new_repo = f"stefanocarrera/autophagycode_M_mercury_{model}_lr0.0001_c142_trust_t1.0_g{g}_run{r}"
            try:
                move_repo(from_id=old_repo, to_id=new_repo, repo_type="model", token=HF_TOKEN)
                print(f"✅ Rinominato: {old_repo} -> {new_repo}")
            except Exception as e:
                print(f"❌ Errore con {old_repo}: {e}")

            # HE
            old_repo = f"{USERNAME}/autophagycode_D_he_train-mercury_{model}_strategy_trust_t1_g{g}_run{r}"
            new_repo = f"{USERNAME}/autophagycode_D_he_train-mercury_{model}_strategy_trust_t1.0_g{g}_run{r}"
            try:
                move_repo(from_id=old_repo, to_id=new_repo, repo_type="dataset", token=HF_TOKEN)
                print(f"✅ Rinominato: {old_repo} -> {new_repo}")
            except Exception as e:
                print(f"❌ Errore con {old_repo}: {e}")

            # TRAIN
            old_repo = f"{USERNAME}/autophagycode_D_mercury_{model}_lr0.0001_c142_trust_t1_g{g}_run{r}"
            new_repo = f"{USERNAME}/autophagycode_D_mercury_{model}_lr0.0001_c142_trust_t1.0_g{g}_run{r}"
            try:
                move_repo(from_id=old_repo, to_id=new_repo, repo_type="dataset", token=HF_TOKEN)
                print(f"✅ Rinominato: {old_repo} -> {new_repo}")
            except Exception as e:
                print(f"❌ Errore con {old_repo}: {e}")

            # METRICS
            old_repo = f"{USERNAME}/autophagycode_D_he_train-mercury_{model}_strategy_trust_t1_g{g}_run{r}_metrics"
            new_repo = f"{USERNAME}/autophagycode_D_he_train-mercury_{model}_strategy_trust_t1.0_g{g}_run{r}_metrics"
            try:
                move_repo(from_id=old_repo, to_id=new_repo, repo_type="dataset", token=HF_TOKEN)
                print(f"✅ Rinominato: {old_repo} -> {new_repo}")
            except Exception as e:
                print(f"❌ Errore con {old_repo}: {e}")
