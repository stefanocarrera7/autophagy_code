from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
from datasets import Dataset
import torch
from typing import Dict, Any, Union, List
from peft import PeftModel

def finetune_model(
    dataset: Dataset,
    base_model_id: str,
    output_dir: str = "unsloth-code-ft",
    model_type: str = "llama",
    num_train_epochs: int = 2,
    lr: float = 1e-4,
    batch_size: int = 2,
    grad_accum: int = 8,
    max_length: int = 1024,
    max_seq_length: int = 1024,    
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    pack_to_max: bool = True,
    resume_adapter_repo: str | None = None
    ) -> tuple[Any, Any]:
    """
    Fine-tuning universale con gestione dinamica del template di prompt.
    Supporta il training incrementale caricando l'adapter della generazione precedente.
    model_type: accetta 'llama' (default) o 'qwen'.
    """
  
    # 1) Caricamento Modello Base
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_id,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
        device_map = "auto",
    )

    # Gestione specifica per Qwen (Pad Token)
    if model_type.lower() == "qwen" or tokenizer.pad_token is None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # 2) Configurazione PEFT (Modificata per la tua Tesi)
    if resume_adapter_repo:
        print(f"🔄 Riprendo l'addestramento dall'adapter: {resume_adapter_repo}")
        # Carichiamo l'adapter sopra il modello base e abilitiamo esplicitamente l'addestramento sui pesi LoRA
        model = PeftModel.from_pretrained(model, resume_adapter_repo, is_trainable=True)
        
        # Riattiviamo il gradient checkpointing per risparmiare VRAM, che potrebbe perdersi col caricamento diretto di PeftModel
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    else:
        print("🆕 Inizializzazione di un nuovo adapter LoRA...")
        model = FastLanguageModel.get_peft_model(
            model,
            r = lora_r,
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout,
            target_modules = target_modules,
            use_gradient_checkpointing = "unsloth",
            random_state = 42,
            bias = "none",
            use_rslora = False,
            loftq_config = None,
        )

    # 3) Preparazione Dati con Template Dinamico
    EOS_TOKEN = tokenizer.eos_token 
    
    def formatting_prompts_func(ex):
        prompt_text = ex['prompt']
        completion_text = ex['completion']

        if model_type.lower() == "qwen":
            # --- FORMATO CHATML (Ideale per Qwen) ---
            text = (
                f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
                f"<|im_start|>assistant\n{completion_text}<|im_end|>"
            ) + EOS_TOKEN
            
        else:
            # --- FORMATO ALPACA/STANDARD (Ideale per Llama) ---
            text = (
                f"### Prompt:\n{prompt_text}\n\n"
                f"### Completion:\n{completion_text}"
            ) + EOS_TOKEN
            
        return {"text": text}

    ds = dataset.map(formatting_prompts_func)
    
    columns_to_keep = ["text"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in columns_to_keep])

    split = ds.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    # 4) Argomenti Trainer
    args = UnslothTrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_accum,
        learning_rate = lr,
        num_train_epochs = num_train_epochs,
        logging_steps = 1,
        eval_strategy = "steps",
        eval_steps = 100,
        save_steps = 100,
        save_total_limit = 2,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        optim = "adamw_8bit",
        lr_scheduler_type = "linear",
        warmup_steps = 5,
        report_to = "none",
        seed = 42,
        max_grad_norm = 1.0, 
    )

    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_ds,
        eval_dataset = eval_ds,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 4,
        args = args,
        packing = pack_to_max,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer