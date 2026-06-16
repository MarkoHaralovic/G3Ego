import os

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image
from torchvision.transforms import v2
from transformers import AutoTokenizer, CLIPTextModel, pipeline

def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

def get_visual_features(model, images, pooling):
    # pooling is either average or concat
    pil_images = [
        Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images
    ]

    feats = []
    for img in pil_images:
        result = model(img)
        if isinstance(result, list) and len(result) > 0:
            feat = torch.tensor(result[0])
        else:
            feat = torch.tensor(result)

        if feat.dim() == 2:
            feat = feat.mean(dim=0)

        feats.append(feat)

    feats = torch.stack(feats, dim=0)

    if pooling == "average":
        feats = feats.mean(dim=0)
    elif pooling == "concat":
        feats = feats.reshape(-1)

    return feats


def get_text_features(text_encoder, tokenizer, actions, pooling, device="cuda"):
    # pooling is either average or none
    text_inputs = tokenizer(actions, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = text_encoder(**text_inputs)

    hidden = outputs.last_hidden_state
    cls_embed = hidden[:, 0, :]

    if pooling == "average":
        return cls_embed.mean(dim=0)
    else:
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

def get_dinov3_extractor(model_name_or_path, cache_dir=None):
    if os.path.exists(model_name_or_path):
        feature_extractor = pipeline(
            model=model_name_or_path, task="image-feature-extraction"
        )
    else:
        feature_extractor = pipeline(
            model=model_name_or_path,
            task="image-feature-extraction",
            model_kwargs={
                "cache_dir": cache_dir,
                "local_files_only": True if cache_dir else False,
            },
        )
    return feature_extractor

def process_folder(
    vision_backbone,
    text_backbone,
    tokenizer,
    input_folder_path,
    model_name,
    pooling="average",
    device="cuda",
):
    clips = [clip for clip in os.listdir(input_folder_path)]
    text_backbone = text_backbone.to(device)
    text_backbone.eval()

    for clip in tqdm.tqdm(clips, desc=f"Extracting features from {input_folder_path}"):
        frames_folder = os.path.join(input_folder_path, clip, "frames")
        if not os.path.exists(frames_folder):
            print(f"Skipping {clip}: frames folder not found")
            continue

        clip_output_path = os.path.join(input_folder_path, clip)
        h5_path_activity = os.path.join(
            clip_output_path,
            f"activity_features_model_{model_name}_pooling_{pooling}.h5",
        )
        h5_path_frame = os.path.join(
            clip_output_path,
            f"frame_action_features_model_{model_name}_pooling_{pooling}.h5",
        )
        h5_path_clip = os.path.join(clip_output_path, f"clip_features_{model_name}.h5")

        if (
            os.path.exists(h5_path_activity)
            and os.path.exists(h5_path_frame)
            and os.path.exists(h5_path_clip)
        ):
            continue

        image_filenames = sorted(
            [img for img in os.listdir(frames_folder) if img.endswith(".jpg")]
        )
        rgb_images = [
            cv2.cvtColor(
                cv2.imread(os.path.join(frames_folder, img)), cv2.COLOR_BGR2RGB
            )
            for img in image_filenames
        ]

        annotations = pd.read_csv(
            os.path.join(input_folder_path, clip, "annotations.csv")
        )

        activity_blocks_ids = sorted(set(annotations["activity_block_id"]))

        visual_feats = []
        text_feats = []
        activity_labels = []
        gaze_labels = []
        frame_visual_feats = []
        frame_text_feats = []
        frame_action_labels = []
        frame_gaze_labels = []

        clip_visual_feats = []

        max_len = None
        for activity_block in tqdm.tqdm(activity_blocks_ids, f"Processing {clip}"):
            data_block = annotations[annotations["activity_block_id"] == activity_block]
            frame_idxs = data_block["frame_index"].tolist()

            frames = [rgb_images[idx] for idx in frame_idxs]
            if max_len is None:
                max_len = len(frames)

            actions = data_block["action"].tolist()
            activity = data_block["activity"].iloc[0]
            gaze_x = data_block["gaze_x"].to_numpy()
            gaze_y = data_block["gaze_y"].to_numpy()
            gazes = np.stack([gaze_x, gaze_y], axis=1)

            for idx, (frame, action, gaze) in enumerate(zip(frames, actions, gazes)):
                frame_visual_feat = get_visual_features(
                    vision_backbone, [frame], "average"
                )
                frame_text_feat = get_text_features(
                    text_backbone, tokenizer, [action], "average", device
                )

                if isinstance(frame_visual_feat, torch.Tensor):
                    frame_visual_feat = frame_visual_feat.detach().cpu().numpy()
                if isinstance(frame_text_feat, torch.Tensor):
                    frame_text_feat = frame_text_feat.detach().cpu().numpy()

                frame_visual_feats.append(frame_visual_feat)
                frame_text_feats.append(frame_text_feat)
                frame_action_labels.append(action)
                frame_gaze_labels.append(gaze)
                clip_visual_feats.append(frame_visual_feat)

            if pooling == "concat" and len(frames) < max_len:
                pad_count = max_len - len(frames)
                last_frame = frames[-1]
                last_action = actions[-1]
                last_gaze = gazes[-1:]

                frames = frames + [last_frame] * pad_count
                actions = actions + [last_action] * pad_count
                gazes = np.concatenate([gazes] + [last_gaze] * pad_count, axis=0)

            image_features = get_visual_features(vision_backbone, frames, pooling)
            text_features = get_text_features(
                text_backbone, tokenizer, actions, pooling, device
            )

            if isinstance(image_features, torch.Tensor):
                image_features = image_features.detach().cpu().numpy()
            if isinstance(text_features, torch.Tensor):
                text_features = text_features.detach().cpu().numpy()

            visual_feats.append(image_features)
            text_feats.append(text_features)
            activity_labels.append(activity)
            gaze_labels.append(gazes)

        visual_feats = np.stack(visual_feats, axis=0)
        text_feats = np.stack(text_feats, axis=0)
        activity_labels = np.array(
            [str(a) for a in activity_labels], dtype=h5py.string_dtype(encoding="utf-8")
        )
        gaze_feats = np.stack(gaze_labels, axis=0)

        with h5py.File(h5_path_activity, "w") as f:
            f.create_dataset("visual_features", data=visual_feats, dtype="float32")
            f.create_dataset("text_features", data=text_feats, dtype="float32")
            f.create_dataset("activity_labels", data=activity_labels)
            f.create_dataset("gaze_labels", data=gaze_feats)

        frame_visual_feats = np.stack(frame_visual_feats, axis=0)
        frame_text_feats = np.stack(frame_text_feats, axis=0)
        frame_action_labels = np.array(
            [str(a) for a in frame_action_labels],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        frame_gaze_feats = np.stack(frame_gaze_labels, axis=0)

        with h5py.File(h5_path_frame, "w") as f:
            f.create_dataset(
                "visual_features", data=frame_visual_feats, dtype="float32"
            )
            f.create_dataset("text_features", data=frame_text_feats, dtype="float32")
            f.create_dataset("action_labels", data=frame_action_labels)
            f.create_dataset("gaze_labels", data=frame_gaze_feats)

        clip_visual_feats = np.stack(clip_visual_feats, axis=0)
        with h5py.File(h5_path_clip, "w") as f:
            f.create_dataset("visual_features", data=clip_visual_feats, dtype="float32")


if __name__ == "__main__":
    POOLING = "concat"
    MODEL_CACHE_DIR = None 
    clip_model_path = ...
    dinov3_model_path = ...
    dinov3_model_name_ab = "dinov3s16+"
    INPUT_DATA_FOLDER = (
        ...
    )

    vision_backbone = get_dinov3_extractor(dinov3_model_path, cache_dir=MODEL_CACHE_DIR)
    tokenizer, text_backbone, _ = get_clip_text_encoder(
        clip_model_path, cache_dir=MODEL_CACHE_DIR
    )

    process_folder(
        vision_backbone,
        text_backbone,
        tokenizer,
        INPUT_DATA_FOLDER,
        dinov3_model_name_ab,
        POOLING,
        device = "cuda",
    )
