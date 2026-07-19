import os
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "models",
    "bltsm",
    "processed_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "bert"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv(DATA_FILE)

print(df.columns)
print(df.head())

texts = df["clean_text"].astype(str).tolist()

# Numeric labels used for training
y = df["label"].astype(int).values

# ============================================================
# Create Label Encoder (for inference later)
# ============================================================

label_encoder = LabelEncoder()
label_encoder.fit(df["emotion"])

joblib.dump(
    label_encoder,
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

# ============================================================
# Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# ============================================================
# Tokenizer
# ============================================================

tokenizer = BertTokenizerFast.from_pretrained(
    "bert-base-uncased"
)

train_encodings = tokenizer(
    X_train,
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    X_test,
    truncation=True,
    padding=True,
    max_length=128
)

# ============================================================
# Dataset Class
# ============================================================

class EmotionDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmotionDataset(train_encodings, y_train)
test_dataset = EmotionDataset(test_encodings, y_test)

# ============================================================
# Load BERT
# ============================================================

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=df["label"].nunique()
)

# ============================================================
# Training Arguments
# ============================================================

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# ============================================================
# Train
# ============================================================

trainer.train()

# ============================================================
# Evaluate
# ============================================================

prediction_output = trainer.predict(test_dataset)

preds = np.argmax(
    prediction_output.predictions,
    axis=1
)

print("\nAccuracy")
print(accuracy_score(y_test, preds))

print("\nClassification Report")

used_labels = sorted(np.unique(y_test))

target_names = [
    label_encoder.classes_[i]
    for i in used_labels
]

print(
    classification_report(
        y_test,
        preds,
        labels=used_labels,
        target_names=target_names,
        zero_division=0
    )
)

# ============================================================
# Save Model
# ============================================================

trainer.save_model(MODEL_DIR)

tokenizer.save_pretrained(MODEL_DIR)

print("\nBERT model saved successfully.")
print("Label Encoder saved successfully.")