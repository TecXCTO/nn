'''
2. Personal Dataset – “Your Own Data”
Your personal dataset could be text, images, audio, or a mix. Below we’ll illustrate:

Text classification – generic CSV or folder structure.
Image classification – folder layout, transfer‑learning.
Feel free to swap the back‑end (TensorFlow or PyTorch) as you wish.

2.1. Text Classification – Custom CSV
Assumptions

data/train.csv & data/val.csv (or a single data.csv with a split column).
Columns: text and label (numeric or string).
Labels can be multi‑class.

2.1.1. Pre‑processing + LSTM (Keras)
# personal_text_lstm.py

# 2.1.2. Fine‑tune BERT (PyTorch)

# personal_text_bert.py
'''
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

# 1. Load data
df = pd.read_csv('data/train.csv')  # combine train+val if needed

# 2. Encode labels
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['label'])
num_classes = len(le.classes_)

# 3. Tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx],
                             truncation=True,
                             padding='max_length',
                             max_length=self.max_len,
                             return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

dataset = IMDBDataset(df['text'].tolist(),
                      df['label_id'].tolist(),
                      tokenizer,
                      max_len=256)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 4. Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=num_classes).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# 5. Train
epochs = 3
for epoch in range(epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item():.4f})

# 6. Save
model.save_pretrained('./personal_bert_finetuned')
tokenizer.save_pretrained('./personal_bert_finetuned')
'''
Tip – torch.save for the state dict and torchvision.models.efficientnet_b0() to load.
For a pure TensorFlow environment, use tf.keras.applications.EfficientNetB0.

3. General Guidelines for Your Personal Dataset
Task	Framework	Recommended Model	How to get started
Text (CSV)	TensorFlow/Keras	Embedding + LSTM, or BERT	Use pandas + Tokenizer.
Text (Folder)	PyTorch	BERT + custom classifier	Use Dataset + DataLoader.
Images	TensorFlow/Keras	ResNet / EfficientNet	Use image_dataset_from_directory.
Images	PyTorch	ResNet / EfficientNet	Use torchvision.datasets.ImageFolder.
Audio	TensorFlow	1‑D Conv + LSTM	Convert to spectrogram first.
Audio	PyTorch	torchaudio + Transformer	Pre‑compute Mel‑spects.
Common steps across all modalities

Data cleaning (remove corrupt files, standardize labels).
Data augmentation – flips, rotations, color jitter (for images); TextAttack or nlpaug for text.
Model checkpointing – ModelCheckpoint (TF) or torch.save (PyTorch).
Early stopping – monitor val loss/accuracy.
Evaluation – use an unseen test split or cross‑validation.

4. Quick Reference Cheat‑Sheet
Task	Best‑Practice	Code Snippet
IMDB classification	Use BERT for best accuracy.	Trainer + TrainingArguments (see above).
Large‑scale text	Use TextVectorization + Bidirectional LSTM.	tf.keras.layers.TextVectorization + LSTM.
Image fine‑tuning	Freeze base until after 5 epochs, then fine‑tune.	model.layers[-4].trainable = True.
Imbalanced data	Compute class weights; use WeightedRandomSampler.	torch.nn.CrossEntropyLoss(weight=class_weights).
GPU memory	Use mixed‑precision training.	tf.keras.mixed_precision.set_global_policy('mixed_float16').
Saving	Use model.save_pretrained() for hugging‑face; model.save() for Keras.	model.save_pretrained('./my_model').
Deployment	Wrap inference in a FastAPI app.	from fastapi import FastAPI + transformers.
Final Thoughts
Start small – test on a subset to debug.
Iterate – tweak learning rates, batch sizes, and augmentations.
Leverage pre‑trained models – they are the quickest path to high performance.
Version your data – store raw + processed versions.
If hitS a specific hurdle (e.g., GPU memory, data loading errors, hyper‑parameter tuning).
'''