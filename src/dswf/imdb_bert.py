'''1. IMDB Reviews – Sentiment Classification
1.1. Overview
The IMDB dataset contains 50 000 movie reviews (≈25 000 for training, 25 000 for validation) labeled positive or negative.
Two typical pipelines:

Approach	Why?	Pros	Cons
LSTM	Classic sequence model.	Simple, fast to train, requires no external libs.	Limited context (short‑term dependencies), slower convergence on long sequences.
BERT	Pre‑trained transformer with contextual embeddings.	State‑of‑the‑art performance on many NLP tasks.	Requires GPU, larger memory, more code.
Below we provide a stand‑alone example for each approach.

1.3. BERT Pipeline (HuggingFace + PyTorch)
The BERT model is pre‑trained on Wikipedia + BookCorpus and can be fine‑tuned on the IMDB data in ~10 min on a single GPU.

# imdb_bert.py
'''

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
from datasets import load_dataset
from tqdm.auto import tqdm

# 1. Load the dataset (🤗 datasets)
raw_dataset = load_dataset("imdb")

# 2. Tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=256)

tokenized = raw_dataset.map(tokenize, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(tokenized["train"], batch_size=16, shuffle=True)
val_loader   = DataLoader(tokenized["test"], batch_size=16)

# 3. Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=2).to(device)

# 4. Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# 5. Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item():.4f})

    # Validation
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    acc = correct / total
    print(f"Val Accuracy: {acc:.4f}")

# 6. Save fine‑tuned model
model.save_pretrained("./imdb_bert_finetuned")
tokenizer.save_pretrained("./imdb_bert_finetuned")
'''
Notes

Max sequence length is 256; can increase to 512 if GPU memory allows.
Batch size is 16; 32 or 64 may be possible on a larger GPU.
Learning rate: 2e‑5 is a good default; 3e‑5 may be slightly faster.
Epochs: 3–4 usually suffices for this dataset.
Evaluation uses the test split (≈25k).
You can also use the 🤗 Trainer API for a cleaner script:
'''
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)
trainer.train()
