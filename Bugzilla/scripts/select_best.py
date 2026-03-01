import json
import os
import shutil

with open("reports/metrics_lstm.json") as f:
    m1 = json.load(f)
with open("reports/metrics_gru.json") as f:
    m2 = json.load(f)

best = "lstm" if m1.get("f1_macro", 0) >= m2.get("f1_macro", 0) else "gru"
os.makedirs("models", exist_ok=True)
shutil.copy(f"models/{best}.h5", "models/best.h5")
print("Best:", best)
