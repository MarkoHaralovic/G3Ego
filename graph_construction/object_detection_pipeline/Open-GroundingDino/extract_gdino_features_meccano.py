import argparse
import ast
import logging
import math
from pathlib import Path
import os
import pickle
import sys
import torch
from PIL import Image
from tqdm import tqdm
import json
import pandas as pd

ANNOTATION_BASE = "/home/s3758869/vlm_datasets/MECCANO_vlm_ann_Qwen3-VL-32B-Instruct-3fps"
IMAGES_BASE = "/deepstore/datasets/dmb/ComputerVision/information_retrieval/MECCANO/dataset/RGB_frames"
# SPLITS = ("Train", "Val", "Test")
SPLITS = ("Val")


# clone https://github.com/IDEA-Research/GroundingDINO into /tools
# move tools/GroundingDINO/groundingdino to /tools/groundingdino
tools_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools')
sys.path.insert(0, tools_path)

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from torchvision.ops import box_convert, nms


def load_model(model_config_path, model_checkpoint_path, device="cuda"):
    """Load GroundingDINO model with weights"""
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model.to(device)


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def get_object_idxs(frame_objects_dict, object_to_idx_map):
    object_indexes = []
    for base_object, base_attr_map in frame_objects_dict.items():
        object_indexes.append(object_to_idx_map[base_attr_map["base_object"]])
    return object_indexes

def prepare_caption(frame_objects_dict):
    # GroundingDINO uses '. ' as separator between object classes in the text prompt
    caption = " . ".join(frame_objects_dict.keys()) + "."
    return caption.lower()

def filter_predictions(outputs, box_threshold, text_threshold, tokenizer, caption, apply_nms=False):
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    boxes_features = outputs["bounding_boxes_features"][0]  # (nq, d)

    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    boxes_features_filt = boxes_features.cpu().clone()
    
    # filter by box threshold
    bbox_threshold_filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[bbox_threshold_filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[bbox_threshold_filt_mask]  # num_filt, 4
    boxes_features_filt = boxes_features_filt[bbox_threshold_filt_mask]  # num_filt, d

    # get phrase
    tokenized = tokenizer(caption)
    
    # build pred and filter by text threshold
    pred_phrases = []
    for logit in logits_filt:
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase.strip())

    if apply_nms and len(boxes_filt) > 0:
        scores = logits_filt.max(dim=1)[0]
        boxes_xyxy = box_convert(boxes_filt, in_fmt="cxcywh", out_fmt="xyxy")
        nms_keep = nms(boxes_xyxy, scores, iou_threshold=0.5)
        boxes_filt = boxes_filt[nms_keep]
        logits_filt = logits_filt[nms_keep]
        boxes_features_filt = boxes_features_filt[nms_keep]
        pred_phrases = [pred_phrases[i] for i in nms_keep.tolist()]

    return boxes_filt, pred_phrases, boxes_features_filt, logits_filt
        
def merge_predictions_of_same_class(boxes_filt, pred_phrases, boxes_features_filt, logits_filt):
    """Merge multiple detections of the same class into a single representation.
    Features are averaged across detections; the most confident bbox is kept.
    
    Returns:
        dict: phrase -> (best_bbox, averaged_feature, best_confidence)
    """
    phrase_to_features = {}
    unique_phrases = set(pred_phrases)
    for phrase in unique_phrases:
        phrase_indexes = [i for i, p in enumerate(pred_phrases) if p == phrase]
        phrase_indexes_t = torch.tensor(phrase_indexes)
        
        bbox_feature_per_class = boxes_features_filt[phrase_indexes_t].mean(dim=0)
        
        # keep only the most confident bbox prediction
        phrase_logits = logits_filt[phrase_indexes_t]
        best_idx = phrase_logits.max(dim=1)[0].argmax()
        phrase_bbox = boxes_filt[phrase_indexes_t][best_idx]
        best_confidence = phrase_logits[best_idx].max().item()
        
        phrase_to_features[phrase] = (phrase_bbox, bbox_feature_per_class, best_confidence)
    return phrase_to_features
        
            
def point_to_bbox_distance(x, y, x_min, y_min, x_max, y_max):
    # closest point on bbox
    closest_x = max(x_min, min(x, x_max))
    closest_y = max(y_min, min(y, y_max))

    # distance
    dx = x - closest_x
    dy = y - closest_y
    return math.sqrt(dx*dx + dy*dy)

def gazed_at_object(boxes_filt, pred_phrases, boxes_features_filt, gaze_x, gaze_y, image_size):
    """Find the object closest to the gaze point.

    Args:
        boxes_filt: filtered boxes in cx,cy,w,h normalized format from GroundingDINO
        pred_phrases: predicted phrases for each box
        boxes_features_filt: features for each box
        gaze_x: gaze x coordinate in pixels (float)
        gaze_y: gaze y coordinate in pixels (float)
        image_size: (width, height) of the original PIL image
    """
    distances_object_gaze = []
    img_w, img_h = image_size
    
    for bbox in boxes_filt:
        cx, cy, bw, bh = bbox.cpu().numpy()
        x_min = (cx - bw / 2) * img_w
        y_min = (cy - bh / 2) * img_h
        x_max = (cx + bw / 2) * img_w
        y_max = (cy + bh / 2) * img_h
        
        distance = point_to_bbox_distance(gaze_x, gaze_y, x_min, y_min, x_max, y_max)
        distances_object_gaze.append(distance)

    if len(distances_object_gaze) == 0:
        return None, None, None
    else:
        min_idx = distances_object_gaze.index(min(distances_object_gaze))
        return pred_phrases[min_idx], boxes_features_filt[min_idx], min_idx
    
def get_model_variant(config_file):
    config_name = os.path.basename(config_file).lower()
    if "swinb" in config_name:
        return "gdino_base"
    else:
        return "gdino_tiny"

def inference(args):
    logger = logging.getLogger('GroundingDINO')

    model_variant = get_model_variant(args.config_file)
    print(f"Detected model variant: {model_variant}")

    model = load_model(args.config_file, args.model_checkpoint_path, args.device)
    logger.info("Model loaded successfully")

    print("Start inference")

    for split in SPLITS:
        annotation_split_path = os.path.join(args.annotation_base, split)
        images_split_path = os.path.join(args.images_base, split)

        if not os.path.isdir(annotation_split_path):
            print(f"Split {split} not found at {annotation_split_path}, skipping")
            continue

        objects_json_path = os.path.join(annotation_split_path, "objects.json")
        if not os.path.exists(objects_json_path):
            print(f"objects.json not found at {objects_json_path}, skipping {split}")
            continue
        with open(objects_json_path, 'r') as f:
            general_object_mapping = json.load(f)  # base_object_name -> idx

        clip_names = [
            clip for clip in os.listdir(annotation_split_path)
            if os.path.isdir(os.path.join(annotation_split_path, clip))
        ]

        for clip_name in tqdm(clip_names, desc=f"{split} clips"):
            annotation_clip_path = os.path.join(annotation_split_path, clip_name)
            images_clip_path = os.path.join(images_split_path, clip_name)

            if not os.path.isdir(images_clip_path):
                print(f"Images folder not found for {clip_name}, skipping")
                continue

            parse_ann_path = os.path.join(annotation_clip_path, "parse_annotation.csv")
            if not os.path.exists(parse_ann_path):
                print(f"parse_annotation.csv not found for {clip_name}, skipping")
                continue
            parsed_annotations = pd.read_csv(parse_ann_path)

            gaze_path = os.path.join(annotation_clip_path, "annotations_qwen3vl_32b_instruct.csv")
            if not os.path.exists(gaze_path):
                print(f"gaze CSV not found for {clip_name}, skipping")
                continue
            gaze_df = pd.read_csv(gaze_path)

            groundings = {}

            for _, ann_row in tqdm(parsed_annotations.iterrows(), desc=f"{clip_name} annotations", total=len(parsed_annotations)):
                frame_id = int(ann_row["frame_id"])

                gaze_row = gaze_df[gaze_df["frame_index"] == frame_id]
                if gaze_row.empty:
                    continue

                frame_file = gaze_row["frame_file"].values[0]
                gaze_x = float(gaze_row["gaze_x"].values[0])
                gaze_y = float(gaze_row["gaze_y"].values[0])

                image_path = os.path.join(images_clip_path, frame_file)
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}, skipping")
                    continue

                image_pil, image = load_image(image_path)
                image = image.to(args.device)

                image_all_objects = ast.literal_eval(ann_row["all_objects"])

                caption = prepare_caption(frame_objects_dict=image_all_objects)

                with torch.no_grad():
                    outputs = model(image[None], captions=[caption])

                boxes_filt, pred_phrases, boxes_features_filt, logits_filt = filter_predictions(
                    outputs, args.box_threshold, args.text_threshold, model.tokenizer, caption, args.apply_nms
                )

                merged = merge_predictions_of_same_class(boxes_filt, pred_phrases, boxes_features_filt, logits_filt)

                merged_phrases = list(merged.keys())
                if len(merged_phrases) > 0:
                    merged_boxes = torch.stack([merged[p][0] for p in merged_phrases])
                    merged_features = torch.stack([merged[p][1] for p in merged_phrases])
                else:
                    merged_boxes = boxes_filt[:0]
                    merged_features = boxes_features_filt[:0]

                gazed_phrase, gazed_bbox_feature, gazed_min_idx = gazed_at_object(
                    merged_boxes, merged_phrases, merged_features, gaze_x, gaze_y, image_pil.size
                )

                gazed_idx = None
                if gazed_phrase is not None:
                    for ann_name, ann_info in image_all_objects.items():
                        if ann_name.lower() == gazed_phrase.lower():
                            gazed_idx = general_object_mapping.get(ann_info["base_object"])
                            break

                objects_dict = {}
                merged_lower = {k.lower(): v for k, v in merged.items()}
                for obj_name, obj_base_attr in image_all_objects.items():
                    obj_idx = general_object_mapping.get(obj_base_attr['base_object'])
                    if obj_idx is None:
                        continue
                    obj_name_lower = obj_name.lower()
                    if obj_name_lower in merged_lower:
                        bbox, feat, conf = merged_lower[obj_name_lower]
                        objects_dict[obj_idx] = {
                            'feats': feat,
                            'phrase': obj_name,
                            'confidence': conf
                        }
                    else:
                        objects_dict[obj_idx] = {
                            'feats': None,
                            'phrase': obj_name,
                            'confidence': 0.0
                        }

                groundings[frame_file] = {
                    "objects": objects_dict,
                    "object_gazed_at": {
                        "feats": gazed_bbox_feature,
                        "phrase": gazed_phrase,
                        "idx": gazed_idx
                    }
                }

            output_path = os.path.join(annotation_clip_path, f"grounding_results_{model_variant}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(groundings, f)
            print(f"Saved {output_path}")
        
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--model_checkpoint_path', '-p', type=str, required=True)
    parser.add_argument('--annotation_base', type=str, default=ANNOTATION_BASE,
                        help="Base directory of MECCANO VLM annotations (with Train/Val/Test subfolders)")
    parser.add_argument('--images_base', type=str, default=IMAGES_BASE,
                        help="Base directory of MECCANO RGB frames (with Train/Val/Test subfolders)")
    parser.add_argument('--box_threshold', type=float, default=0.3, help="box threshold")
    parser.add_argument('--text_threshold', type=float, default=0.15, help="text threshold")
    parser.add_argument("--apply-nms", type=bool, default=False, help="whether to use NMS or not")
    parser.add_argument("--device", type=str, default="cuda", help="device to use for inference")
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('GroundingDINO MECCANO Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    inference(args)
