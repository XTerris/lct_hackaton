from typing import List, Tuple
import re


def get_char_spans_and_labels(text, annotations):
    sorted_annotations = sorted(annotations, key=lambda x: x[0])

    words_with_spans = []
    for match in re.finditer(r"\S+", text):
        words_with_spans.append((match.group(), match.start(), match.end()))

    word_labels = []
    for word, start, end in words_with_spans:
        label = "O"
        for ann_start, ann_end, ann_label in sorted_annotations:
            if start >= ann_start and end <= ann_end:
                label = ann_label
                break
        word_labels.append((word, label))

    return word_labels


def create_char_spans(text, words_and_labels):
    spans = []
    pos = 0

    for word, label in words_and_labels:
        match = re.search(re.escape(word), text[pos:])
        if match:
            start = pos + match.start()
            end = pos + match.end()
            if label != "O":
                spans.append((start, end, label))
            pos = end

    return spans


def align_labels_with_tokens(text, spans, tokenizer):
    tokens = tokenizer.tokenize(text)
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoded["offset_mapping"]

    labels = ["O"] * len(tokens)

    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
        for span_start, span_end, label in spans:
            if token_start >= span_start and token_end <= span_end:
                labels[token_idx] = label
                break

    return labels


def get_label_map():
    labels = ["O", "TYPE", "BRAND", "VOLUME", "PERCENT"]
    return {label: i for i, label in enumerate(labels)}
