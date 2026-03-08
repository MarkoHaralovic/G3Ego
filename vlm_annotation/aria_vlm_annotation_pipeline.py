import os
import json
import random
import argparse
import importlib

import cv2
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import tqdm

from projectaria_tools.core.mps.utils import get_eyegaze_point_at_depth
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.aea import AriaEverydayActivitiesDataProvider
from aux import stream_rgb_vrs_recording, get_every_nth_second_frame

RGB_STREAM_ID = StreamId("214-1")

def action_annotate(cfg, vlm, model, processor, images):
    local_window = cfg["local_window_for_action_size"]
    image_size = cfg["image_size"]
    actions = []
    prev, after = int(local_window // 2), int(local_window // 2)
    for i in tqdm.tqdm(range(len(images)), desc="Action annotating clip"):
        images_for_act = images[max(0, i - prev) : min(len(images), i + after + 1)]
        images_for_act = [np.rot90(img, k=-1) for img in images_for_act]
        if local_window == 1:
            result = vlm.recognize_action_single_frame(
                model, processor, images_for_act[0], cfg["_action_single_frame_prompt"], image_size
            )
        else:
            result = vlm.recognize_action_local_window(
                model, processor, images_for_act, cfg["_action_window_prompt"], image_size
            )

        action = vlm.parse_output(result)
        actions.append(action)

        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    return actions


def activity_annotate(cfg, vlm, model, processor, images, actions=None):
    N_frames_for_activity = cfg["n_frames_for_activity"]
    image_size = cfg["image_size"]
    activities = []
    T = len(images)

    max_activities = (T + N_frames_for_activity - 1) // N_frames_for_activity

    for i in tqdm.tqdm(range(max_activities), desc="Activity annotating clip"):
        torch.cuda.empty_cache()

        start = i * N_frames_for_activity
        end = min(start + N_frames_for_activity, len(images) - 1)

        images_for_act = images[start:end]

        if len(images_for_act) == 0:
            print(f"Warning: No images in block {i}, skipping")
            activities.append("not_annotated")
            continue

        images_for_act = [np.rot90(img, k=-1) for img in images_for_act]

        redo = True
        act_image_size = image_size
        act_images = images_for_act[:]

        while redo:
            if len(act_images) == 0:
                print(f"Error: Image list became empty in block {i}, skipping")
                result = None
                redo = False
                break

            try:
                activity_prompt = cfg["_activity_prompt"].format(actions=actions)
                result = vlm.recognize_activity(
                    model, processor, act_images, activity_prompt, act_image_size
                )
                redo = False
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(act_images) > 5 and act_image_size == image_size:
                    current_num = len(act_images)
                    new_num = max(5, current_num - 1)
                    act_images = random.sample(act_images, new_num)
                    print(f"OOM: Reducing image count to {new_num}")
                elif act_image_size == 336 and len(act_images) >= 5:
                    act_image_size = 118
                    print(f"OOM: Reducing image size to {act_image_size}")
                elif len(act_images) > 2 and act_image_size == 118:
                    current_num = len(act_images)
                    new_num = max(2, current_num - 1)
                    act_images = random.sample(act_images, new_num)
                    print(f"OOM: Reducing image count to {new_num}")
                elif len(act_images) == 2 and act_image_size == 118:
                    # Try with just 1 image
                    act_images = random.sample(act_images, 1)
                    print(f"OOM: Reducing to 1 image")
                else:
                    print("OOM: Cannot reduce further, skipping")
                    result = None
                    redo = False

        activity = vlm.parse_output(result) if result is not None else "not_annotated"
        activities.append(activity)

        del act_images
        torch.cuda.empty_cache()

    return activities


def save_images_annotations(
    images_np, output_path, actions, activities, N_frames_for_activity, gazes=None, model_id=None
):
    os.makedirs(output_path, exist_ok=True)
    img_dir = os.path.join(output_path, "frames")
    os.makedirs(img_dir, exist_ok=True)

    csv_records = []

    for i, img in enumerate(images_np):
        img = np.rot90(img, k=-1)
        frame_name = f"frame_{i:04d}.jpg"
        frame_path = os.path.join(img_dir, frame_name)

        cv2.imwrite(frame_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        block_idx = i // N_frames_for_activity

        h, w = img.shape[:2]
        gaze_x = gazes[i]["gaze_x"] if gazes and i < len(gazes) else None
        gaze_y = gazes[i]["gaze_y"] if gazes and i < len(gazes) else None

        if gaze_x is not None and gaze_y is not None:
            gaze_x_rot = h - gaze_y
            gaze_y_rot = gaze_x
        else:
            gaze_x_rot = None
            gaze_y_rot = None

        csv_records.append(
            {
                "frame_index": i,
                "frame_file": frame_name,
                "action": actions[i] if i < len(actions) else "",
                "activity_block_id": block_idx,
                "activity": (
                    activities[block_idx] if block_idx < len(activities) else ""
                ),
                "gaze_x": gaze_x_rot,
                "gaze_y": gaze_y_rot,
            }
        )

    with open(os.path.join(output_path, "actions.txt"), "w") as f:
        for a in actions:
            f.write(a + "\n")

    with open(os.path.join(output_path, "activities.txt"), "w") as f:
        for act in activities:
            f.write(act + "\n")

    df = pd.DataFrame(csv_records)
    df.to_csv(os.path.join(output_path, f"annotations_{model_id}.csv"), index=False)

    print(
        f"Saved:\n  {len(images_np)} frames\n  {len(actions)} actions\n  {len(activities)} activities"
    )


def get_gaze_projection(aea_data_provider, device_time_ns, depth_m=1.0):
    rgb_stream_label = aea_data_provider.vrs.get_label_from_stream_id(RGB_STREAM_ID)
    device_calibration = aea_data_provider.vrs.get_device_calibration()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

    eye_gaze = aea_data_provider.mps.get_general_eyegaze(device_time_ns)

    if eye_gaze is None:
        return None, None

    gaze_vector_in_cpf = get_eyegaze_point_at_depth(
        eye_gaze.yaw, eye_gaze.pitch, depth_m
    )
    T_device_CPF = device_calibration.get_transform_device_cpf()
    gaze_center_in_camera = (
        rgb_camera_calibration.get_transform_device_camera().inverse()
        @ T_device_CPF
        @ gaze_vector_in_cpf
    )
    gaze_projection = rgb_camera_calibration.project(gaze_center_in_camera)

    if gaze_projection is None:
        return None, None

    return gaze_projection[0], gaze_projection[1]


def get_saliency_map_gaze(gazes_x, gazes_y, H, W, sigma=15):
    saliency = np.zeros((H, W), dtype=np.float32)

    for x, y in zip(gazes_x, gazes_y):
        if 0 <= x < W and 0 <= y < H:
            saliency[int(y), int(x)] += 1.0

    saliency = cv2.GaussianBlur(saliency, (0, 0), sigma)
    saliency = saliency / saliency.max()
    saliency = (saliency * 255).astype(np.uint8)
    return saliency


def get_gazes_for_frames(aea_data_provider, timestamps_ns):
    gazes = []
    for ts in timestamps_ns:
        x, y = get_gaze_projection(aea_data_provider, ts)
        gazes.append({"gaze_x": x, "gaze_y": y})
    return gazes


def annotate_clip(
    cfg, vlm, model, processor,
    input_clip_path, input_clip_name, output_clip_path,
):
    actions_file = os.path.join(output_clip_path, "actions.txt")
    activities_file = os.path.join(output_clip_path, "activities.txt")

    actions_exist = os.path.exists(actions_file)
    activities_exist = os.path.exists(activities_file)

    aea_data_provider = AriaEverydayActivitiesDataProvider(
        os.path.join(input_clip_path, input_clip_name)
    )

    rgb_images, rgb_timestamps = stream_rgb_vrs_recording(
        aea_data_provider, RGB_STREAM_ID
    )

    timestamps_ns = np.array(rgb_timestamps)
    samples = get_every_nth_second_frame(timestamps_ns, n_seconds=cfg["each_nth_frame_sec"])

    rgb_images_to_select = [rgb_images[k] for k in samples]
    timestamps_to_select = [rgb_timestamps[k] for k in samples]

    gazes = get_gazes_for_frames(aea_data_provider, timestamps_to_select)

    if actions_exist:
        print(f"Loading existing actions for {input_clip_name}")
        with open(actions_file, "r") as f:
            actions_in_a_clip = [line.strip() for line in f.readlines()]
    else:
        actions_in_a_clip = action_annotate(
            cfg, vlm, model, processor, rgb_images_to_select,
        )

    if activities_exist:
        print(f"Loading existing activities for {input_clip_name}")
        with open(activities_file, "r") as f:
            activity_in_a_clip = [line.strip() for line in f.readlines()]
    else:
        activity_in_a_clip = activity_annotate(
            cfg, vlm, model, processor, rgb_images_to_select,
        )

    save_images_annotations(
        rgb_images_to_select,
        output_clip_path,
        actions_in_a_clip,
        activity_in_a_clip,
        cfg["n_frames_for_activity"],
        gazes,
        model_id=cfg["model_id"]
    )


def annotate_dataset(cfg, vlm, model, processor):
    input_path = cfg["input_data_folder"]
    output_path = cfg["output_data_folder"]
    clip_names = [
        clip
        for clip in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, clip))
    ]
    for clip_name in tqdm.tqdm(clip_names, desc="Processing clips"):
        output_clip_path = os.path.join(output_path, clip_name)

        # Check if both annotations exist
        actions_file = os.path.join(output_clip_path, "actions.txt")
        activities_file = os.path.join(output_clip_path, "activities.txt")

        if os.path.exists(actions_file) and os.path.exists(activities_file):
            print(f"Skipping {clip_name}: already fully annotated")
            continue
        elif os.path.exists(actions_file) or os.path.exists(activities_file):
            print(f"Processing {clip_name}: partial annotations found, completing...")
        else:
            print(f"Processing clip: {clip_name}")

        try:
            annotate_clip(
                cfg, vlm, model, processor,
                input_path, clip_name, output_clip_path,
            )
        except RuntimeError as e:
            print(e)
            continue


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

    cfg["_action_single_frame_prompt"] = prompts[cfg["action_single_frame_prompt"]]
    cfg["_action_window_prompt"] = prompts[cfg["action_window_prompt"]]
    cfg["_activity_prompt"] = prompts[cfg["activity_prompt"]]

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
