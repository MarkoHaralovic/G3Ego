import os
import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image
from torchvision.transforms import v2
from transformers import pipeline

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
    results = model(pil_images, batch_size=len(pil_images))
    feats = []
    for result in results:
        if isinstance(result, list):
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


def get_dinov3_extractor(model_name_or_path, cache_dir=None, device="cpu"):
    if os.path.exists(model_name_or_path):
        feature_extractor = pipeline(
            model=model_name_or_path,
            task="image-feature-extraction",
            device=device,
        )
    else:
        feature_extractor = pipeline(
            model=model_name_or_path,
            task="image-feature-extraction",
            device=device,
            model_kwargs={
                "cache_dir": cache_dir,
                "local_files_only": True if cache_dir else False,
            },
        )
    return feature_extractor

def process_folder(
    vision_backbone,
    images_folder_path,
    annotation_folder_path,
    model_name,
    batch_size=128
):
    """Extract DINOv2 features for each clip and save h5 files into the
    annotation clip folder alongside the existing CSV annotations.
    """
    clips = [
        c for c in os.listdir(annotation_folder_path)
        if os.path.isdir(os.path.join(annotation_folder_path, c))
    ]
    
    for clip in tqdm.tqdm(clips, desc=f"Extracting features from {annotation_folder_path}"):
        frames_folder = os.path.join(images_folder_path, clip)
        annotation_csv = os.path.join(
            annotation_folder_path, clip, "annotations_qwen3vl_32b_instruct.csv"
        )
        clip_output_path = os.path.join(annotation_folder_path, clip)

        if not os.path.exists(frames_folder):
            print(f"Skipping {clip}: frames folder not found at {frames_folder}")
            continue
        if not os.path.exists(annotation_csv):
            print(f"Skipping {clip}: annotation CSV not found at {annotation_csv}")
            continue

        h5_path_frame = os.path.join(
            clip_output_path,
            f"frame_features_model_{model_name}.h5",
        )
        h5_path_clip = os.path.join(clip_output_path, f"clip_features_{model_name}.h5")

        if os.path.exists(h5_path_frame) and os.path.exists(h5_path_clip):
            continue

        image_filenames = sorted(
            [img for img in os.listdir(frames_folder) if img.endswith(".jpg")]
        )

        # Build gaze lookup: frame_file -> (gaze_x, gaze_y) from annotation CSV
        annotations = pd.read_csv(annotation_csv)
        gaze_lookup = {
            row["frame_file"]: (row["gaze_x"], row["gaze_y"])
            for _, row in annotations.iterrows()
        }

        frame_ids = []
        frame_names = []
        frame_visual_feats = []
        frame_gaze_labels = []

        for start in tqdm.tqdm(
            range(0, len(image_filenames), batch_size),
            desc=clip,
            leave=False,
        ):
            batch_fnames = image_filenames[start:start + batch_size]
            batch_images = []

            for fname in batch_fnames:
                img_path = os.path.join(frames_folder, fname)
                img_bgr = cv2.imread(img_path)
                image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                batch_images.append(image)

            if len(batch_images) == 0:
                continue

            batch_feats = get_visual_features(
                vision_backbone,
                batch_images,
                pooling="none",
            )

            if isinstance(batch_feats, torch.Tensor):
                batch_feats = batch_feats.detach().cpu().numpy()

            for i, fname in enumerate(batch_fnames):
                frame_id = start + i
                gaze = gaze_lookup.get(fname, (float("nan"), float("nan")))

                frame_ids.append(frame_id)
                frame_names.append(fname)
                frame_visual_feats.append(batch_feats[i])
                frame_gaze_labels.append(gaze)

        frame_visual_feats_arr = np.stack(frame_visual_feats, axis=0)  # (N, D)
        frame_ids_arr = np.array(frame_ids, dtype="int32")
        frame_names_arr = np.array(
            frame_names, dtype=h5py.string_dtype(encoding="utf-8")
        )
        frame_gaze_arr = np.array(frame_gaze_labels, dtype="float32")  # (N, 2)

        with h5py.File(h5_path_frame, "w") as f:
            f.create_dataset("frame_ids", data=frame_ids_arr)
            f.create_dataset("frame_names", data=frame_names_arr)
            f.create_dataset("visual_features", data=frame_visual_feats_arr, dtype="float32")
            f.create_dataset("gaze_labels", data=frame_gaze_arr)

if __name__ == "__main__":
    MODEL_CACHE_DIR = None
    dinov3_model_path = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    dinov3_model_name_ab = "dinov3_vits16"

    SPLITS = ("Train", "Val", "Test")
    ANNOTATION_BASE = "/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/MECCANO_vlm_ann_Qwen3-VL-32B-Instruct-3fps/"
    IMAGES_BASE = "/deepstore/datasets/dmb/ComputerVision/information_retrieval/MECCANO/dataset/RGB_frames"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    vision_backbone = get_dinov3_extractor(
        dinov3_model_path, cache_dir=MODEL_CACHE_DIR, device=device
    )

    for split in SPLITS:
        process_folder(
            vision_backbone,
            images_folder_path=os.path.join(IMAGES_BASE, split),
            annotation_folder_path=os.path.join(ANNOTATION_BASE, split),
            model_name=dinov3_model_name_ab,
        )
