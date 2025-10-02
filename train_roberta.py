import torch
import pandas as pd
from datasets import Dataset
from typing import List, Tuple
import os
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from ner_data_utils import (
    align_labels_with_tokens,
    get_label_map,
)

os.environ["WANDB_DISABLED"] = "true"

def prepare_dataset(texts, annotations, tokenizer):
    label_map = get_label_map()

    processed_data = {"input_ids": [], "attention_mask": [], "labels": []}

    for text, spans in zip(texts, annotations):
        tokenized = tokenizer(text, truncation=True, padding=False)

        token_labels = align_labels_with_tokens(text, spans, tokenizer)

        label_ids = [label_map[label] for label in token_labels]

        label_ids = [-100] + label_ids + [-100]  # -100 is ignored in loss calculation

        processed_data["input_ids"].append(tokenized["input_ids"])
        processed_data["attention_mask"].append(tokenized["attention_mask"])
        processed_data["labels"].append(label_ids)

    return Dataset.from_dict(processed_data)


def create_predictions(text, model, tokenizer):
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = encoded.pop("offset_mapping")[0]

    with torch.no_grad():
        outputs = model(**encoded)
        predictions = outputs.logits.argmax(-1)[0].cpu().numpy()

    label_map = get_label_map()
    id2label = {v: k for k, v in label_map.items()}

    spans = []
    current_label = None
    start_pos = None

    for pred, (start, end) in zip(predictions[1:-1], offset_mapping[1:-1]):
        label = id2label[pred]

        if label != "O":
            if current_label != label:
                if current_label is not None:
                    spans.append((start_pos, prev_end, current_label))
                start_pos = start
                current_label = label
            prev_end = end
        else:
            if current_label is not None:
                spans.append((start_pos, prev_end, current_label))
                current_label = None

    if current_label is not None:
        spans.append((start_pos, prev_end, current_label))

    return spans


MODEL_NAME = "sberbank-ai/ruRoberta-large"

def prep(x):
    return [(i[0], i[1], i[2].split("-")[-1]) for i in x]


sample_data = pd.read_csv("./train_extended.csv", delimiter=";")
sample_data["annotation"] = sample_data["annotation"].apply(eval).apply(prep)

print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
num_labels = len(get_label_map())
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels
)

# MODEL_PATH = "./fine_tuned_ruRoberta_ner_v1"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

dataset = prepare_dataset(
    sample_data["sample"].tolist(),
    sample_data["annotation"].tolist(),
    tokenizer,
)

training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=20,
        save_strategy="epoch",
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )

data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("Starting model fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

model_save_path = "./fine_tuned_ruRoberta_ner_v5"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")