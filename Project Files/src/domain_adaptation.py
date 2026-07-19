import os
import joblib
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

PROJECT_DIR = r"C:\Users\Lokesh S\OneDrive\Desktop\EMOTION DETECTION PROJECT"

MODEL_DIR = os.path.join(PROJECT_DIR, "models", "bltsm")

DATA_FILE = os.path.join(MODEL_DIR, "processed_dataset.csv")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABEL_ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "final_bilstm_model.keras")

MAX_LENGTH = 80

print("Loading resources...")

df = pd.read_csv(DATA_FILE)

tokenizer = joblib.load(TOKENIZER_FILE)

label_encoder = joblib.load(LABEL_ENCODER_FILE)

baseline_model = tf.keras.models.load_model(MODEL_FILE)

print("Resources loaded successfully.")

X = tokenizer.texts_to_sequences(df["clean_text"])

X = pad_sequences(
    X,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

y = label_encoder.transform(df["emotion"])

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

adaptive_model = tf.keras.models.clone_model(baseline_model)

adaptive_model.set_weights(baseline_model.get_weights())

adaptive_model.layers[0].trainable = False

adaptive_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

print("\nStarting Domain Adaptation...\n")

history = adaptive_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

loss, accuracy = adaptive_model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print("\nValidation Accuracy :", round(accuracy * 100, 2), "%")
print("Validation Loss :", round(loss, 4))

adaptive_model.save(
    os.path.join(
        MODEL_DIR,
        "bilstm_student_adaptive.keras"
    )
)

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Domain Adaptation Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "domain_adaptation_loss.png"
    )
)

plt.close()

with open(
    os.path.join(
        MODEL_DIR,
        "domain_adaptation_results.txt"
    ),
    "w"
) as f:

    f.write("Domain Adaptation Results\n")
    f.write("=========================\n\n")
    f.write(f"Validation Accuracy : {accuracy*100:.2f}%\n")
    f.write(f"Validation Loss : {loss:.4f}\n")

print("\nAdaptive model saved successfully.")
print("Loss graph saved.")
print("Results file saved.")
print("\nDomain Adaptation Completed Successfully.")