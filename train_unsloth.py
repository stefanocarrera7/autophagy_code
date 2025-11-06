from datasets import Dataset
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
import torch
from typing import Dict, Any
from peft import PeftModel



def finetune_model_unsloth(
    dataset: Dataset,
    base_model_id: str,
    output_dir: str = "unsloth-code-ft",
    num_train_epochs: int = 2,
    lr: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 16,
    max_length: int = 2048,
    max_seq_length: int = 4096,     # per il modello
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules = ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"),
    pack_to_max: bool = True,
    resume_adapter_repo: str | None = None
    ) -> tuple[Any, Any]:
    """
    Fine-tuning QLoRA con Unsloth per code generation.
    dataset: HF Dataset con campi 'prompt' e 'completion'
    Ritorna: (model, tokenizer) già pronti per inference.
    """
  
    # 1) Prepara campo testo unico (istruito a produrre "completion" dato il "prompt")
    def join_prompt_completion(ex):
        return {"text": f"### Prompt:\n{ex['prompt']}\n\n### Completion:\n{ex['completion']}\n"}
    ds = dataset.map(join_prompt_completion)
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])

    split = ds.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

  # 2) Modello + tokenizer Unsloth (4-bit pronto)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_id,
        max_seq_length = max_seq_length,
        dtype = None,           # auto fp16/bf16
        load_in_4bit = True,
        device_map = "auto",
    )

  # 3) PEFT LoRA via helper Unsloth
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        target_modules = list(target_modules),
        use_gradient_checkpointing = True,
        random_state = 42,
        bias = "none",
        task_type = "CAUSAL_LM",
    )
    
        # Resume da adapter precedente se fornito
    if resume_adapter_repo:
        model = PeftModel.from_pretrained(model, resume_adapter_repo)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

  # 4) Tokenizzazione
    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)
    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    eval_tok  = eval_ds.map(tok_fn,  batched=True, remove_columns=["text"])

  # 5) Argomenti Trainer Unsloth
    args = UnslothTrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_accum,
        learning_rate = lr,
        num_train_epochs = num_train_epochs,
        logging_steps = 50,
        evaluation_strategy = "steps",
        eval_steps = 500,
        save_steps = 500,
        save_total_limit = 2,
        bf16 = torch.cuda.is_bf16_supported(),
        fp16 = not torch.cuda.is_bf16_supported(),
        optim = "paged_adamw_8bit",          # tipico setup 4-bit
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.03,
        report_to = "none",
    )

  # 6) Trainer Unsloth (packing opzionale)
    trainer = UnslothTrainer(
        model = model,
        args = args,
        train_dataset = train_tok,
        eval_dataset = eval_tok,
        tokenizer = tokenizer,
        packing = pack_to_max,              # True = concatena campioni per riempire i blocchi
    )

    trainer.train()

    # 7) Salvataggi adapter + tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 8) Pronto per inference
    FastLanguageModel.for_inference(model)  # abilita kv-cache ottimizzata
    return model, tokenizer
