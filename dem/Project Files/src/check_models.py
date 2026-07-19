import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

paths = [
    "models/bltsm/bilstm_model.keras",
    "models/bltsm/tokenizer.pkl",
    "models/bltsm/label_encoder.pkl",
    "models/bltsm/processed_dataset.csv",

    "models/bert_emotion_model_final/config.json",
    "models/bert_emotion_model_final/model.safetensors",
    "models/bert_emotion_model_final/tokenizer.json",
    "models/bert_emotion_model_final/tokenizer_config.json",
    "models/bert_emotion_model_final/special_tokens_map.json",
    "models/bert_emotion_model_final/vocab.txt"
]

print("=" * 50)
print("Checking Model Files")
print("=" * 50)

for file in paths:
    full_path = os.path.join(BASE_DIR, file)

    if os.path.exists(full_path):
        print(f"✓ {file}")
    else:
        print(f"✗ {file}")

print("=" * 50)