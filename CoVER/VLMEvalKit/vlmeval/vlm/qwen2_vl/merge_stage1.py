from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import torch
import os

base_model_path = "./output/Qwen2.5-VL-7B-sft-cover-phase1/checkpoint-245"
adapter_path = "./CoVER/VLMEvalKit/vlmeval/vlm/qwen2_vl/output_think_verify_stage1_checkpoint"
output_path = "./CoVER/VLMEvalKit/vlmeval/vlm/qwen2_vl/output_think_verify_stage1_checkpoint_merged"

print("Loading base model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {output_path}...")
os.makedirs(output_path, exist_ok=True)
model.save_pretrained(output_path, safe_serialization=True)

print("Saving tokenizer and processor...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)

try:
    processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)
    processor.save_pretrained(output_path)
except Exception:
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.save_pretrained(output_path)

print("Done! Merged model saved to:", output_path)
