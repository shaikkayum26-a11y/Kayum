import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences

PROJECT_DIR = r"C:\Users\Lokesh S\OneDrive\Desktop\EMOTION DETECTION PROJECT"

MODEL_DIR = os.path.join(PROJECT_DIR, "models", "bltsm")

DATA_FILE = os.path.join(MODEL_DIR, "processed_dataset.csv")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.pkl")
ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")

print("Loading:", DATA_FILE)
print("Exists:", os.path.exists(DATA_FILE))

print("Loading:", TOKENIZER_FILE)
print("Exists:", os.path.exists(TOKENIZER_FILE))

print("Loading:", ENCODER_FILE)
print("Exists:", os.path.exists(ENCODER_FILE))

df = pd.read_csv(DATA_FILE)

tokenizer = joblib.load(TOKENIZER_FILE)
label_encoder = joblib.load(ENCODER_FILE)

texts = df["clean_text"].astype(str)
labels = df["label"]

MAX_LENGTH = 80

sequences = tokenizer.texts_to_sequences(texts)

X = pad_sequences(
    sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

y = labels.values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

vocab_size = len(tokenizer.word_index) + 1
num_classes = len(label_encoder.classes_)

embedding_dim = 128
lstm_units = 128

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))

print("\n==============================")
print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
print("Vocabulary Size  :", vocab_size)
print("Number of Classes:", num_classes)
print("Classes          :", label_encoder.classes_)
print("==============================")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

model = Sequential()

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=MAX_LENGTH
    )
)

model.add(
    Bidirectional(
        LSTM(
            lstm_units,
            return_sequences=False
        )
    )
)

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(
    Dense(
        num_classes,
        activation="softmax"
    )
)

optimizer = Adam(
    learning_rate=0.001
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "bilstm_emotion_model.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=16,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ],
    verbose=1
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

accuracy = accuracy_score(y_test, y_pred)

print("\n")
print("=" * 60)
print("MODEL ACCURACY :", round(accuracy * 100, 2), "%")
print("=" * 60)

print("\nClassification Report\n")

labels = np.unique(np.concatenate((y_test, y_pred)))

report = classification_report(
    y_test,
    y_pred,
    labels=labels,
    target_names=label_encoder.classes_[labels],
    zero_division=0
) 

print(report)

with open(
    os.path.join(MODEL_DIR, "classification_report.txt"),
    "w"
) as f:
    f.write(report)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix\n")
print(cm)

model.save(
    os.path.join(MODEL_DIR, "final_bilstm_model.keras")
)

joblib.dump(
    history.history,
    os.path.join(MODEL_DIR, "training_history.pkl")
)

print("\nModel Saved Successfully")
print("Classification Report Saved")
print("Training History Saved")
print("Project Completed Successfully")