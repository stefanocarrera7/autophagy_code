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
    batch_size: int = 2,
    grad_accum: int = 8,
    max_length: int = 2048,
    max_seq_length: int = 4096,    
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0,
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
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        target_modules = list(target_modules),
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
        bias = "none",
        use_rslora = False,
        loftq_config = None,
    )
    
    # Resume da adapter precedente se fornito
    if resume_adapter_repo:
        print(f"Loading adapter from: {resume_adapter_repo}")
        model = PeftModel.from_pretrained(model, resume_adapter_repo)

    # 3) Preparazione Dati
    EOS_TOKEN = tokenizer.eos_token 
    
    def formatting_prompts_func(ex):
        text = f"### Prompt:\n{ex['prompt']}\n\n### Completion:\n{ex['completion']}" + EOS_TOKEN
        return {"text": text}

    ds = dataset.map(formatting_prompts_func)
    
    # Rimuoviamo colonne vecchie, teniamo solo 'text'
    columns_to_keep = ["text"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in columns_to_keep])

    # Split train/test
    split = ds.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    # 4) Argomenti Trainer (FIX: eval_strategy)
    args = UnslothTrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_accum,
        learning_rate = lr,
        num_train_epochs = num_train_epochs,
        logging_steps = 1,
        eval_strategy = "steps",           # <--- CORRETTO (prima era evaluation_strategy)
        eval_steps = 100,
        save_steps = 100,
        save_total_limit = 2,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        optim = "adamw_8bit",
        lr_scheduler_type = "linear",
        warmup_steps = 5,
        report_to = "none",
        seed = 42
    )

    # 5) Trainer
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

    # 6) Salvataggi
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    FastLanguageModel.for_inference(model)
    
    return model, tokenizer