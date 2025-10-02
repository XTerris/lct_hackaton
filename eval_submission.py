import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm.notebook import tqdm
from typing import List, Tuple
from time import time

from ner_data_utils import get_label_map
from train_roberta import create_predictions

tqdm.pandas()


label_map = get_label_map()
id2label = {v: k for k, v in label_map.items()}

def adjust_prediction(s, pred):
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", s)]
    if not words and s.strip():
        words = [(0, len(s))]

    result = []
    for start, end in words:
        label = "O"
        for p_start, p_end, p_class in pred:
            if p_start <= start and end <= p_end:
                label = p_class
                break
        result.append((start, end, label))
    return result

def add_bio(pred):
    res = []
    seen = set()
    for i in pred:
        if i[2] == "O":
            res.append(i)
            continue
        if i[2] in seen:
            i = (i[0], i[1], "I-" + i[2])
        else:
            seen.add(i[2])
            i = (i[0], i[1], "B-" + i[2])
        res.append(i)
    return res

def predict(s, model, tokenizer, device):
    pred = create_predictions(s, model, tokenizer, device)
    res = adjust_prediction(s, pred)
    res = add_bio(res)
    return res

version = 5
MODEL_PATH = f"./fine_tuned_ruRoberta_ner_v{version}"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

df = pd.read_csv("./submission.csv", delimiter=";")

t0 = time()
with torch.no_grad():
    df["annotation"] = df["sample"].progress_apply(lambda x: predict(x, model, tokenizer, device))

df.to_csv(f"./submission_v{version}.csv", index=False, sep=";")
print(f"Time taken: {time() - t0:.4f} seconds")
print(df.head(20).to_string())
