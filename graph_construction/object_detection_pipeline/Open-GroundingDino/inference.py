import argparse
import logging
from pathlib import Path
import os
import pickle
import sys
import torch
from PIL import Image

# clone https://github.com/IDEA-Research/GroundingDINO into /tools
# move tools/GroundingDINO/groundingdino to /tools/groundingdino
tools_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools')
sys.path.insert(0, tools_path)

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--model_checkpoint_path', '-p', type=str, required=True)
    parser.add_argument('--image_dir', '-i', type=str, required=True)
    parser.add_argument('--text_prompt', '-t', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, default="outputs", required=True, help="output directory")
    parser.add_argument('--box_threshold', type=float, default=0.3, help="box threshold")
    parser.add_argument('--text_threshold', type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cuda", help="device to use for inference")
    
    return parser


def load_model(model_config_path, model_checkpoint_path, device="cuda"):
    """Load GroundingDINO model with weights"""
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


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



def main(args):
    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = logging.getLogger('GroundingDINO')
    
    #load the model
    model = load_model(args.config_file, args.model_checkpoint_path, args.device)
    model.to(args.device)
    logger.info("Model loaded successfully")

    caption = args.text_prompt
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
 
    results = {}
    print("Start inference")
    
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_name in image_files:
        image_path = os.path.join(args.image_dir, image_name)
        print(f"Processing {image_name}...")
        
        image_pil, image = load_image(image_path)
        image = image.to(args.device)

        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
            
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        boxes_features = outputs["bounding_boxes_features"][0]  # (nq, d)

        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        boxes_features_filt = boxes_features.cpu().clone()
        
        filt_mask = logits_filt.max(dim=1)[0] > args.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        boxes_features_filt = boxes_features_filt[filt_mask]  # num_filt, d

        # get phrase
        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)
        # build pred
        pred_phrases = []
        for logit in logits_filt:
            pred_phrase = get_phrases_from_posmap(logit > args.text_threshold, tokenized, tokenizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        
        results[image_name] = {
            "boxes": boxes_filt,
            "phrases": pred_phrases,
            "features": boxes_features_filt,
            "logits": logits_filt
        }
        
        print(f"  Found {len(pred_phrases)} objects: {pred_phrases}")
    
    output_path = os.path.join(args.output_dir, "detection_results.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GroundingDINO Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)


