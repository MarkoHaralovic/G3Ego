import os
import json
import argparse
import importlib

import cv2
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import tqdm
from pathlib import Path

def get_gazes_for_frames(gaze_input_folder : Path, mode : str, clip_names : str):
    gaze_data = {}
    for clip_name in clip_names:
        gaze_file = os.path.join(gaze_input_folder, mode, f"{clip_name}_gaze-data.csv")
        if not os.path.exists(gaze_file):
            print(f"Gaze file not found for {clip_name}, lookign elsewhere.")
            other_modes = [m for m in ["Train", "Val", "Test"] if m != mode]
            for m in other_modes:
                gaze_file_alt = os.path.join(gaze_input_folder, m, f"{clip_name}_gaze-data.csv")
                if os.path.exists(gaze_file_alt):
                    print(f"Found gaze file for {clip_name} in {m} folder, using it.")
                    gaze_file = gaze_file_alt
                    break
        if not os.path.exists(gaze_file):
            gaze_data[clip_name] = {}
            continue

        # frame,confidence,x_pixel_coord,y_pixel_coord
        gaze_rows = pd.read_csv(gaze_file)
        gazes = {}
        for _, row in gaze_rows.iterrows(): 
            gazes[Path(str(row["frame"])).stem] = {
                "confidence": row["confidence"],
                "gaze_x": row["x_pixel_coord"],
                "gaze_y": row["y_pixel_coord"],
            }
        gaze_data[clip_name] = gazes
    return gaze_data

def get_frames(frames_input_folder : Path, mode : str):
    frames = {}
    for clip_name in os.listdir(os.path.join(frames_input_folder, mode)):
        frames[clip_name] = {}
        clip_folder = os.path.join(frames_input_folder, mode, clip_name)
        if os.path.exists(clip_folder):
            for frame_file in sorted(os.listdir(clip_folder)):
                if frame_file.endswith(".jpg") or frame_file.endswith(".png"):
                    frame_index = Path(str(frame_file)).stem 
                    frames[clip_name][frame_index] = os.path.join(clip_folder, frame_file)
    return frames

def action_annotate(cfg, vlm, model, processor, image_paths):
    local_window = cfg["local_window_for_action_size"]
    image_size = cfg["image_size"]
    actions = []
    prev, after = int(local_window // 2), int(local_window // 2)
    for i in tqdm.tqdm(range(len(image_paths)), desc="Action annotating clip", total=len(image_paths)):
        images_for_act = [cv2.imread(image_paths[j]) for j in range(max(0, i - prev), min(len(image_paths), i + after + 1))]
        result = vlm.recognize_action_single_frame(
            model, processor, images_for_act[0], cfg["action_recognition_prompt"], image_size
        )
        action = vlm.parse_output(result)
        actions.append(action)
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
    return actions

def save_images_annotations(
    image_paths, output_path, actions, gazes, model_id,
):
    os.makedirs(output_path, exist_ok=True)
    img_dir = os.path.join(output_path, "RGB_frames")
    os.makedirs(img_dir, exist_ok=True)

    csv_records = []

    for i, image_path in enumerate(image_paths):
        frame_name = os.path.basename(image_path)
        frame_path = os.path.join(img_dir, frame_name)
        image = cv2.imread(image_path)
        if image is not None:
            cv2.imwrite(frame_path, image)

        gaze = gazes[i] if gazes and i < len(gazes) else None

        csv_records.append(
            {
                "frame_index": i,
                "frame_file": frame_name,
                "action": actions[i] if i < len(actions) else "",
                "gaze_x": gaze["gaze_x"] if gaze else None,
                "gaze_y": gaze["gaze_y"] if gaze else None,
                "confidence": gaze["confidence"] if gaze else None,
            }
        )

    with open(os.path.join(output_path, "actions.txt"), "w") as f:
        for a in actions:
            f.write(a + "\n")

    df = pd.DataFrame(csv_records)
    df.to_csv(os.path.join(output_path, f"annotations_{model_id}.csv"), index=False)

    print(
        f"Saved:\n  {len(image_paths)} frames\n  {len(actions)} actions"
    )


def annotate_clip(
    cfg, vlm, model, processor, frames, gazes, output_clip_path
):  
    ordered_frames = sorted(frames.items(), key=lambda item: int(item[0]))
    image_paths = [path for _, path in ordered_frames]
    ordered_gazes = [gazes.get(frame_key) for frame_key, _ in ordered_frames]

    actions_in_a_clip = action_annotate(
        cfg, vlm, model, processor, image_paths,
    )
    
    save_images_annotations(
        image_paths,
        output_clip_path,
        actions_in_a_clip,
        ordered_gazes,
        model_id=cfg["model_id"]
    )


def annotate_dataset(cfg, vlm, model, processor):
    output_path = cfg["output_data_folder"]
    
    modes = ["Train", "Val", "Test"]

    for mode in modes:
        frames = get_frames(cfg["frames_input_folder"], mode)
        gazes = get_gazes_for_frames(cfg["gaze_input_folder"], mode, list(frames.keys()))
        
        for clip_name in frames.keys():
            annotate_clip(
                cfg, vlm, model, processor, frames[clip_name], gazes[clip_name],
                os.path.join(output_path, mode, clip_name)
            )

def gpu_worker(gpu_id, task_queue, result_queue, cfg, vlm):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    print(f"[GPU {gpu_id}] Loading model...")
    model, processor = vlm.load_model(
        cfg["model_path"], device,
        four_bit=cfg["four_bit_quantization"],
        eight_bit=cfg["eight_bit_quantization"],
        model_cache_dir=cfg["model_cache_dir"],
    )
    print(f"[GPU {gpu_id}] Model loaded on {model.device}")

    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:
                break

            input_path, clip_name, output_path = task
            output_clip_path = os.path.join(output_path, clip_name)

            print(f"[GPU {gpu_id}] Processing {clip_name}")

            try:
                annotate_clip(
                    cfg, vlm, model, processor,
                    input_path, clip_name, output_clip_path,
                )
                result_queue.put((clip_name, "success", None))
            except Exception as e:
                result_queue.put((clip_name, "error", str(e)))

        except Exception:
            continue
        
def annotate_dataset_mp(cfg, vlm, num_gpus):
    input_path = cfg["input_data_folder"]
    output_path = cfg["output_data_folder"]
    clip_names = [
        name
        for name in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, name))
    ]
    clip_names = [
        name
        for name in clip_names
        if not os.path.exists(os.path.join(output_path, name))
    ]
    total_clips = len(clip_names)

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for clip_name in clip_names:
        task_queue.put((input_path, clip_name, output_path))

    for _ in range(num_gpus):
        task_queue.put(None)

    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, cfg, vlm),
        )
        p.start()
        workers.append(p)

    completed = 0
    errors = []
    with tqdm.tqdm(total=total_clips, desc="Total progress") as pbar:
        while completed < total_clips:
            clip_name, status, error = result_queue.get()
            completed += 1
            pbar.update(1)
            if status == "error":
                errors.append((clip_name, error))
                pbar.set_postfix({"errors": len(errors)})

    for p in workers:
        p.join()

    if errors:
        for clip_name, error in errors:
            print(f"  - {clip_name}: {error}")
            
def main():
    parser = argparse.ArgumentParser("VLM Annotation Pipeline")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    # ---- load config ----
    with open(args.config, "r") as f:
        cfg = json.load(f)

    # ---- load prompts from the JSON file specified in config ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_path = os.path.join(script_dir, cfg["prompts_file"])
    with open(prompts_path, "r") as f:
        prompts = json.load(f)

    cfg["action_recognition_prompt"] = prompts[cfg["action_recognition_prompt"]]

    # ---- import vlm module from the path specified in config ----
    vlm = importlib.import_module(cfg["vlm_module"])

    os.makedirs(cfg["output_data_folder"], exist_ok=True)

    num_gpus = torch.cuda.device_count()
    quant = cfg["four_bit_quantization"] or cfg["eight_bit_quantization"]

    if num_gpus == 1 or quant:
        device = "cuda:0" if num_gpus == 1 else "cuda"
        model, processor = vlm.load_model(
            cfg["model_path"], device,
            four_bit=cfg["four_bit_quantization"],
            eight_bit=cfg["eight_bit_quantization"],
            model_cache_dir=cfg["model_cache_dir"],
        )
        print(model.device)
        annotate_dataset(cfg, vlm, model, processor)
    elif num_gpus > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        annotate_dataset_mp(cfg, vlm, num_gpus)
    else:
        raise Exception("No GPU detected.")


if __name__ == "__main__":
    main()
