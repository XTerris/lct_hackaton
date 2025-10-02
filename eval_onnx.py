import pandas as pd
import torch
import re
from transformers import AutoTokenizer
import onnxruntime
import numpy as np

from ner_data_utils import get_label_map

MODEL_PATH = "./fine_tuned_ruRoberta_ner_v2"
ONNX_MODEL_PATH = "./onnx/roberta_v2.onnx"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if torch.cuda.is_available()
    else ["CPUExecutionProvider"]
)
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
)
ort_session = onnxruntime.InferenceSession(
    ONNX_MODEL_PATH, sess_options=sess_options, providers=providers
)

device = torch.device("cuda" if "CUDAExecutionProvider" in providers else "cpu")
print(f"Using device: {device} with ONNX providers: {ort_session.get_providers()}")


def adjust_prediction(s, pred):
    words = [(match.start(), match.end()) for match in re.finditer(r"\S+", s)]
    if not words and s.strip():
        words = [(0, len(s))]

    pred_idx = 0
    result = []
    seen = set()

    for start, end in words:
        label = "O"
        while pred_idx < len(pred) and pred[pred_idx][1] < start:
            pred_idx += 1

        if (
            pred_idx < len(pred)
            and pred[pred_idx][0] <= start
            and end <= pred[pred_idx][1]
        ):
            label = pred[pred_idx][2]

        if label != "O":
            if label in seen:
                label = "I-" + label
            else:
                seen.add(label)
                label = "B-" + label
        result.append({"start_index": start, "end_index": end, "entity": label})

    return result


def create_predictions_onnx(text: str, session, tokenizer):
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = encoded.pop("offset_mapping")[0]

    ort_inputs = {
        "input_ids": encoded["input_ids"].cpu().numpy(),
        "attention_mask": encoded["attention_mask"].cpu().numpy(),
    }

    ort_outs = session.run(None, ort_inputs)
    predictions = np.argmax(ort_outs[0], axis=2)[0]

    label_map = get_label_map()
    id2label = {v: k for k, v in label_map.items()}

    spans = []
    current_label = None
    start_pos = None

    for _, (pred, (start, end)) in enumerate(
        zip(predictions[1:-1], offset_mapping[1:-1])
    ):
        if start == end:  # Skip special tokens
            continue

        label = id2label[pred]

        if label != "O":
            simple_label = label.split("-")[-1]
            if current_label != simple_label:
                if current_label is not None:
                    spans.append((start_pos, prev_end, current_label))
                start_pos = start.item()
                current_label = simple_label
            prev_end = end.item()
        else:
            if current_label is not None:
                spans.append((start_pos, prev_end, current_label))
                current_label = None

    if current_label is not None:
        spans.append((start_pos, prev_end, current_label))

    return spans


def predict(input):
    pred = create_predictions_onnx(input, ort_session, tokenizer)
    res = adjust_prediction(input, pred)
    return res
