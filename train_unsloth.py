from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
from datasets import Dataset
import torch
from typing import Dict, Any
from peft import PeftModel

def finetune_model_unsloth(
    dataset: Dataset,
    base_model_id: str,
    output_dir: str = "unsloth-code-ft",
    num_train_epochs: int = 2,
    lr: float = 2e-4,
    batch_size: int = 2,           # Aumentato a 2 come nel codice del collega (se la GPU regge)
    grad_accum: int = 8,           # Bilanciato con il batch size
    max_length: int = 2048,
    max_seq_length: int = 4096,    
    lora_r: int = 16,
    lora_alpha: int = 16,          # Allineato al collega (era 32)
    lora_dropout: float = 0,       # IMPT: Messo a 0 come da best practice Unsloth/codice collega
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    pack_to_max: bool = True,
    resume_adapter_repo: str | None = None
    ) -> tuple[Any, Any]:
    """
    Fine-tuning QLoRA con Unsloth per code generation.
    """
  
    # 1) Modello + tokenizer Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_id,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
        device_map = "auto",
    )

    # 2) PEFT LoRA
    # Abbiamo rimosso task_type="CAUSAL_LM" che dava errore
    # Abbiamo allineato i parametri a quelli del collega (dropout=0, bias="none")
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        target_modules = list(target_modules),
        use_gradient_checkpointing = "unsloth", # Ottimizzazione usata nel codice collega
        random_state = 42,
        bias = "none",
        use_rslora = False,
        loftq_config = None,
    )
    
    # Resume da adapter precedente se fornito (logica tua, mantenuta)
    if resume_adapter_repo:
        print(f"Loading adapter from: {resume_adapter_repo}")
        model = PeftModel.from_pretrained(model, resume_adapter_repo)

    # 3) Preparazione Dati (Integrato FIX EOS TOKEN)
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    
    def formatting_prompts_func(ex):
        # Manteniamo la tua struttura "Prompt/Completion" che è ottima per l'instruction tuning
        # MA aggiungiamo l'EOS token alla fine, essenziale!
        text = f"### Prompt:\n{ex['prompt']}\n\n### Completion:\n{ex['completion']}" + EOS_TOKEN
        return {"text": text}

    # Mappiamo il dataset
    ds = dataset.map(formatting_prompts_func)
    
    # Rimuoviamo colonne vecchie per pulizia, teniamo solo 'text'
    columns_to_keep = ["text"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in columns_to_keep])

    # Split train/test (Manteniamo la tua logica, utile per monitorare l'overfitting)
    split = ds.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    # 4) Argomenti Trainer (Allineati al collega per stabilità)
    args = UnslothTrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_accum,
        learning_rate = lr,
        num_train_epochs = num_train_epochs,
        logging_steps = 1,                 # Più frequente come nel codice collega
        evaluation_strategy = "steps",
        eval_steps = 100,                  # Valuta ogni 100 step
        save_steps = 100,
        save_total_limit = 2,
        bf16 = is_bfloat16_supported(),    # Logica robusta per bf16/fp16
        fp16 = not is_bfloat16_supported(),
        optim = "adamw_8bit",              # Ottimizzatore 8bit standard
        lr_scheduler_type = "linear",      # Il collega usa linear, spesso più stabile del cosine per fine-tuning brevi
        warmup_steps = 5,                  # Warmup fisso invece di ratio, più sicuro
        report_to = "none",
        seed = 42
    )

    # 5) Trainer
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_ds,
        eval_dataset = eval_ds,
        dataset_text_field = "text",       # Specifica il campo testo
        max_seq_length = max_seq_length,
        dataset_num_proc = 4,              # Parallelizza il processing dati
        args = args,
        packing = pack_to_max,
    )

    trainer.train()

    # 6) Salvataggi
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Preparazione per l'uso successivo
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer