from pathlib import Path
from transformers.onnx.convert import export
from transformers.onnx.features import FeaturesManager
from transformers import AutoTokenizer, AutoModelForTokenClassification


model_dir = Path("./fine_tuned_ruRoberta_ner_v1")
onnx_dir = Path("./onnx")
onnx_dir.mkdir(exist_ok=True)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

feature = "token-classification"
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
    model, feature=feature
)
onnx_config = model_onnx_config(model.config)

print("Exporting model to ONNX...")
onnx_inputs, onnx_outputs = export(
    preprocessor=tokenizer,
    model=model,
    config=onnx_config,
    opset=14,
    output=onnx_dir / "roberta_v2.onnx",
    device="cpu",
)

print(f"ONNX model saved to {onnx_dir / 'roberta_v2.onnx'}")
