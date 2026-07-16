import os
from transformers import CLIPTextModel, AutoTokenizer
import json
import pickle
import torch

ANNOTATION_PATH = "/path/to/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf/"
input_file_name = "all_words_enum.json"
output_file_name = "clip_text_features.pkl"

with open(os.path.join(ANNOTATION_PATH, input_file_name), "r") as f:
   text_items = json.load(f)
   
def get_text_features(text_encoder, tokenizer, text, device="cuda"):
   # pooling is either average or none
   text_inputs = tokenizer(text, padding=True, return_tensors="pt").to(device)

   with torch.no_grad():
      outputs = text_encoder(**text_inputs)

   hidden = outputs.last_hidden_state
   cls_embed = hidden[:, 0, :]

   return cls_embed

def get_clip_text_encoder(model_name, cache_dir=None):
   tokenizer = AutoTokenizer.from_pretrained(
      model_name, cache_dir=cache_dir, local_files_only=True if cache_dir else False
   )
   text_encoder = CLIPTextModel.from_pretrained(
      model_name, cache_dir=cache_dir, local_files_only=True if cache_dir else False
   )
   configuration = text_encoder.config
   return tokenizer, text_encoder, configuration


device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CACHE_DIR = "/path/to/models"
clip_model_path = "/path/to/models/clip-vit-base-patch32"

tokenizer, text_encoder, _ = get_clip_text_encoder(
   clip_model_path, cache_dir=MODEL_CACHE_DIR
)

for text, v in text_items.items():
   text_items[text] = get_text_features(text_encoder, tokenizer, text, device=device).squeeze(0)
   
with open(os.path.join(ANNOTATION_PATH, output_file_name), "wb") as f:
   pickle.dump(text_items, f)