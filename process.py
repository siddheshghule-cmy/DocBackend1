import os
import pandas as pd
from ocr_utils import extract_text_from_image

data = []

base_dir = "training_data"

for label in os.listdir(base_dir):
    label_dir = os.path.join(base_dir, label)
    if os.path.isdir(label_dir):
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            text = extract_text_from_image(file_bytes)
            data.append({"text": text, "label": label})

df = pd.DataFrame(data)
df.to_csv("training_data2.csv", index=False)
print("CSV saved! Total samples:", len(df))
