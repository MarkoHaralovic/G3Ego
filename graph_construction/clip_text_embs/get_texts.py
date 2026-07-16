import os
import json
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")

def to_singular(word: str) -> str:
   if len(word.split(" ")) > 1:
      return word
   doc = nlp(word)
   return_class = []
   for tok in doc:
      if tok.pos_ in ("NOUN", "PROPN", "ADJ"):
         return_class.append(tok.lemma_.lower())
   if return_class:
      return " ".join(c for c in return_class)
   return word.lower()


ANNOTATION_PATH = "/path/to/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf/"
ann_file_name="parse_annotation.csv"
output_file_name = "all_words_enum.json"

with open(os.path.join(ANNOTATION_PATH, "verbs.json"), "r") as f:
   verbs = json.load(f)

with open(os.path.join(ANNOTATION_PATH, "relationships.json"), "r") as f:
   rels = json.load(f)

with open(os.path.join(ANNOTATION_PATH, "attributes.json"), "r") as f:
   attrs = json.load(f)
   

text_items = {}
idx = 0
for k in verbs.keys():
   text_items[idx] = k
   idx += 1
for k in rels.keys():
   text_items[idx] = k
   idx += 1
for k in attrs.keys():
   text_items[idx] = k
   idx += 1

for clip in os.listdir(ANNOTATION_PATH):
   if os.path.isdir(os.path.join(ANNOTATION_PATH, clip)):
      ann_csv = pd.read_csv(os.path.join(ANNOTATION_PATH, clip, ann_file_name))
      for frame_id in ann_csv["frame_id"].tolist():
         objects_atr_val = ann_csv[ann_csv["frame_id"] == frame_id]["all_objects"].iloc[0]
         if isinstance(objects_atr_val, str):
            obj_dict = json.loads(objects_atr_val)
            for obj in obj_dict.values():
               base  = obj.get("base_object", "")
               text = " ".join(obj.get("attributes", []) + [to_singular(base)])
               text_items[idx] = text
               idx += 1

text_items[len(text_items)] = "camera_wearer"
# soem words can be in multiple jsons, as they can be words and nouns. this ensures one is kept
seen = set()
text_items_first = {}
for k, v in text_items.items():
   if v not in seen:
      text_items_first[v] = k
      seen.add(v)
text_items = text_items_first
text_items = dict(sorted(text_items.items(), key=lambda item: item[1]))
with open(os.path.join(ANNOTATION_PATH,output_file_name), "w+") as f:
   json.dump(text_items, f, indent=5)