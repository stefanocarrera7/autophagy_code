# Puoi usare transformers classico o Unsloth se stai lavorando su pipeline di training/ottimizzazione
from unsloth import FastLanguageModel

# Carichiamo un modello Qwen leggero per fare un test rapido
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 1. VISUALIZZARE I LAYER (Architettura Macro)
# Questo stamperà a schermo la struttura ad albero: Embedding -> Decoder Layers (Attention, MLP) -> RMSNorm
print(model)

# 2. ISPEZIONARE I TENSORI (Architettura Micro)
# Questo itera attraverso ogni singolo tensore mostrandone il nome esatto e la forma (es. torch.Size([1536, 1536]))
for name, param in model.named_parameters():
    print(f"Nome Tensore: {name} | Dimensioni: {param.size()} | Richiede Gradiente: {param.requires_grad}")