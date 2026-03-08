import os
import random
import re
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava16.llava import load_model as load_llava_model, parse_output as parse_llava_output

testing_folder = "/home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf/loc1_script1_seq1_rec1"
frames_path = os.path.join(testing_folder, "frames")
actions_path = os.path.join(testing_folder, "actions.txt")
activities_path = os.path.join(testing_folder, "activities.txt")
annotations_path = os.path.join(testing_folder, "annotations.csv")

VLM_ANNOTATOR = "/home/s3758869/models/llava-v1.6-34b-hf"
IMAGE_SIZE_VLM_INPUT = 336
N_FRAMES_FOR_ACTIVITY = 10


def shorten_action(action_sentence):
    s = action_sentence.strip()
    return re.sub("The camera wearer is", "", s)


def prompt_1(actions, start, end):
    return f"""
         You are given {end - start} egocentric images that belong to one short time span. 
         They are ordered in time from earliest to latest.

         You also receive a list of short action descriptions predicted for each individual frame:
         {actions}
      
         Your task:
         1. Look at ALL the images together.
         2. Decide the SINGLE main activity the human (camera wearer) is performing across the whole sequence.
         3. Express this activity as one compact label in the format:

         <verb>_<object>

         Formatting rules:
         - Use exactly one verb and one object.
         - Verb: base form, lowercase, no tense (e.g., open, close, take, put, use, wash, cut, pour, eat, drink, look, walk, call).
         - Object: a single simple noun, lowercase, no adjectives (e.g., door, phone, cup, bed, plate, laptop, sink, window).
         - Use only lowercase letters and the underscore `_`. No spaces, commas, or extra words.

         Examples:
         - someone streching -> exercise_stretch
         - someone opening a door to go outside → open_door
         - someone scrolling on a smartphone → use_phone
         - someone eating the last bites of lunch → finish_lunch
         - someone talking on a phone → talk_phone
         IMPORTANT:
         - Return ONLY the activity label token. 
         - Do NOT add explanations, sentences, or quotes.
         """


def prompt_2(actions, start, end):
    return f"""
      You are given {end - start} egocentric images that belong to one short time span.
      They are ordered in time from earliest to latest.

      You also receive a list of short action descriptions predicted for each individual frame:
      {actions}

      Your task:
      1. Look at ALL the images together.
      2. Read the frame-level actions.
      3. Decide the SINGLE main activity the human (camera wearer) is performing across MOST of the frames.
         - The main activity should be the one that is most frequent or lasts the longest.
         - Ignore very short or transitional actions that only happen in 1–2 frames (for example, briefly opening a door between rooms).
         - Base annotation on visual information of frames, but also on per-frame action labels. Choose major activity performed the longest across the frames.
         
      Express this main activity as one compact label in the format:

         <verb>_<object>

      Formatting rules:
      - Use exactly one verb and one object.
      - Verb: base form, lowercase, no tense (e.g., open, close, take, put, use, wash, cut, pour, eat, drink, look, walk, call, sit, stand).
      - Object: a single simple noun, lowercase, no adjectives (e.g., door, phone, cup, bed, plate, laptop, sink, window, couch, table).
      - Use only lowercase letters and the underscore `_`. No spaces, commas, or extra words.
      - If you truly cannot tell, output the word: unknown

      Decision rules:
      - Prefer the activity that is supported by the largest number of frames.
      - If the person spends most frames sitting or standing in the same place, choose that activity rather than a brief door-opening transition.
      - If multiple activities have similar support, choose the most visually central interaction (e.g., with phone, computer, cup).

      Examples:
      - Frames mostly show someone sitting on a couch, with one frame opening a door → sit_couch
      - Frames mostly show someone scrolling on a smartphone → use_phone
      - Frames mostly show someone running on a treadmill → exercise_treadmill
      IMPORTANT:
         - Return ONLY the activity label token. 
         - Do NOT add explanations, sentences, or quotes.
      """


def prompt_3(actions, gazes_textual, start, end):
    return f"""
   "text": (
         "You are an expert in gaze-based event causal understanding within ego-centric video environments. "
         "You are given {end - start} egocentric images that belong to one short time span.They are ordered in time from earliest to latest."
         "Your task is to express this main activity across frames. Desribe which action is being perfoermed across input frames. Read the frame-level actions.
         Express this main activity as one compact label in the format: <verb>_<object>"
         "The benchmarks should focus on complex, high-level reasoning that integrates gaze dynamics with event interactions.\n\n"
         "**Input data**:\n"
         "1. Visual: first-person RGB frames.\n"
         "2. Textual: Ego-centric gaze information: Includes gaze fixation points.\n\n"
         "3. Textual: action annotation per frame by a VLM.\n\n"
         "**Event Parsing**:\n"
         "- Identify action chains (e.g., pick glass → drink → glance at bottle → pour water).\n"
         "- Link actions to gaze patterns, focusing on implicit cause-and-effect relationships.\n\n"
         "**Gaze Dynamics**:\n"
         "- Cluster gaze points as natural scene regions (e.g., 'cup rim', 'bottle cap area').\n"
         "- Track gaze trajectory shifts (e.g., 'suddenly locked onto...').\n"
         "- Focus on: Gaze as a predictive signal for upcoming actions; Ambiguous gaze paths that may point to multiple plausible behaviors.\n\n"
         "Return short and desriptive action annotation, focusing on main activity.Examples:
      - Frames mostly show someone sitting on a couch, with one frame opening a door → sit_couch
      - Frames mostly show someone scrolling on a smartphone → use_phone
      - Frames mostly show someone running on a treadmill → exercise_treadmill
      IMPORTANT:
         - Return ONLY the activity label token. 
         - Do NOT add explanations, sentences, or quotes."
         "Gazes are: {gazes_textual}"
      )
   """


def prompt_4(actions, start, end):
    return f"""
   "text": (
         "You are an expert in gaze-based event causal understanding within ego-centric video environments. "
         "You are given {end - start} egocentric images that belong to one short time span.They are ordered in time from earliest to latest."
         "Your task is to express this main activity across frames. Desribe which action is being perfoermed across input frames. Read the frame-level actions.
         Express this main activity as one compact label in the format:<verb>_<object>"
         "The benchmarks should focus on complex, high-level reasoning that integrates gaze dynamics with event interactions.\n\n"
         "**Input data**:\n"
         "1. Visual: first-person RGB frames.\n"
         "2. Visual: Ego-centric gaze information: Includes gaze fixation points.\n\n"
         "3. Textual: action annotation per frame by a VLM.\n\n"
         "**Event Parsing**:\n"
         "- Identify action chains (e.g., pick glass → drink → glance at bottle → pour water).\n"
         "- Link actions to gaze patterns, focusing on implicit cause-and-effect relationships.\n\n"
         "**Gaze Dynamics**:\n"
         "- Cluster gaze points as natural scene regions (e.g., 'cup rim', 'bottle cap area').\n"
         "- Track gaze trajectory shifts (e.g., 'suddenly locked onto...').\n"
         "- Focus on: Gaze as a predictive signal for upcoming actions; Ambiguous gaze paths that may point to multiple plausible behaviors.\n\n"
         "Examples:
      - Frames mostly show someone sitting on a couch, with one frame opening a door → sit_couch
      - Frames mostly show someone scrolling on a smartphone → use_phone
      - Frames mostly show someone running on a treadmill → exercise_treadmill
      IMPORTANT:
         - Return ONLY the activity label token. 
         - Do NOT add explanations, sentences, or quotes."
      )
   """


def prompt_5(actions, start, end):
    return f"""
   "text": "
      You are given {end - start} egocentric images that belong to one short time span. 
      They are ordered in time from earliest to latest.

      Your task:
      1. Look at ALL the images together.
      2. Decide the SINGLE main activity the human (camera wearer) is performing across the whole sequence.
      3. Express this activity as one compact label in the format:

      <verb>_<object>

      Formatting rules:
      - Use exactly one verb and one object.
      - Verb: base form, lowercase, no tense (e.g., open, close, take, put, use, wash, cut, pour, eat, drink, look, walk, call).
      - Object: a single simple noun, lowercase, no adjectives (e.g., door, phone, cup, bed, plate, laptop, sink, window).
      - Use only lowercase letters and the underscore `_`. No spaces, commas, or extra words.
      - If you truly cannot tell, output the word: unknown

      Examples:
      - someone opening a door to go outside → open_door
      - someone scrolling on a smartphone → use_phone
      - someone eating the last bites of lunch → finish_lunch
      - someone running on a treadmill → exercise_treadmill
      - someone talking on a phone → talk_phone
      - someone streching -> exercise_stretch
      IMPORTANT:
      - Return ONLY the activity label token (like open_door). 
      - Do NOT add explanations, sentences, or quotes.
      
      "
   """


def prompt_6(actions, start, end):
    return f"""
    You are given {end - start} egocentric images from a short continuous time span.
    They are ordered from earliest to latest.

    Input:
    - A sequence of first-person RGB frames.
    - A list of per-frame action predictions: {actions}

    Task:
    1. Infer the SINGLE main activity performed across the sequence.
    2. Use both:
       - Visual continuity across the frames
       - Temporal patterns in the per-frame actions (majority, duration, consistency)
    3. Ignore brief, transitional, or ambiguous actions lasting only 1–2 frames.
    4. Resolve conflicts between competing activities by:
       - Preferring the longest continuous activity
       - Preferring visually central objects (e.g., phone, cup, coffee maker)
       - Rejecting actions inconsistent with most frames

    Output Format:
    - A single label in the form <verb>_<object>
    - Verb: base form, lowercase (open, close, take, put, use, wash, cut, pour, eat, drink, walk, call, sit, stand)
    - Object: simple noun, lowercase (door, phone, cup, coffee, cabinet, sink, window)
    - Only lowercase letters and an underscore.
    - If no dominant activity exists, output: unknown

    Examples:
    - Mostly using a smartphone while not performing any other action → use_phone
    - Mostly operating a coffee machine → use_coffee_maker
    - Briefly opening a cabinet but mostly drinking coffee → drink_coffee
    - Performing strecth in a room -> exercise_stretch
    - Cooking some food and looking at the food -> cook_food
    - Watching TV -> watch_tv

    Important:
    - Return ONLY the label, nothing else.
    """


def prompt_7(actions, start, end):
    return f"""
      You are given several egocentric images that belong to one short time span. 
      They are ordered in time from earliest to latest.

      Your task:
      1. Look at ALL the images together.
      2. Decide the SINGLE main activity the human (camera wearer) is performing across the whole sequence.
      3. Express this activity as one compact label in the format:

      <verb>_<object>

      Formatting rules:
      - Use exactly one verb and one object.
      - Verb: base form, lowercase, no tense (e.g., open, close, take, put, use, wash, cut, pour, eat, drink, look, walk, call).
      - Object: a single simple noun, lowercase, no adjectives (e.g., door, phone, cup, bed, plate, laptop, sink, window).
      - Use only lowercase letters and the underscore `_`. No spaces, commas, or extra words.

      Examples:
      - someone opening a door to go outside → open_door
      - someone scrolling on a smartphone → use_phone
      - someone eating the last bites of lunch → finish_lunch
      - someone running on a treadmill → exercise_treadmill
      - someone talking on a phone → talk_phone
      - someone streching -> exercise_stretch
      IMPORTANT:
      - Return ONLY the activity label token. (like open_door)
      - Do NOT add explanations, sentences, or quotes.
      
      Previous actions based on single frame were : {actions}
      """


def prompt_8(actions, start, end):
    return f"""
      You are analyzing egocentric video frames - images captured from a first-person perspective using META smart glasses 
      mounted on a person's head. These images show exactly what the camera wearer sees as they perform daily activities.
      
      Key characteristics of egocentric images:
      - The viewpoint is from the person's head/eye level, looking outward at the world
      - You will often see the person's hands, arms, or objects they are manipulating in the foreground
      - The perspective is naturally shaky and dynamic as the person moves
      - Objects of interest are typically in the center of view where the person is looking
      - Background context shows the environment (kitchen, living room, office, etc.)
      - The camera wearer themselves is NOT visible (except possibly their hands/arms)
      
      The images provided are sequential frames ordered chronologically from earliest to latest, capturing a continuous activity.

      Your task:
      1. Analyze ALL the images together as a temporal sequence.
      2. Identify the SINGLE main activity the camera wearer is performing across the entire sequence.
      3. Focus on what the person's hands are doing and what objects they are interacting with.
      4. Express this activity as one compact label in the format:

      <verb>_<object>

      Formatting rules:
      - Use exactly one verb and one object.
      - Verb: base form, lowercase, no tense 
      - Object: a single simple noun, lowercase, no adjectives 
      - Use only lowercase letters and the underscore `_`. No spaces, commas, or extra words.

      Examples:
      - person eating the last bites of lunch from a plate → finish_lunch
      - person running on a treadmill (legs/floor visible) → exercise_treadmill
      - person holding a phone to their ear → talk_phone
      - person stretching their arms or legs (legs visible) → exercise_stretch
      - person's hands scrolling on a smartphone → use_phone
      
      IMPORTANT:
      - Return ONLY the activity label token. 
      - Do NOT add explanations, sentences, or quotes, only <verb>_<object>      
      Previous actions based on single frame were : {actions}
      """


def prompt_9(actions, start, end):
    return f"""
      You are analyzing egocentric video frames - images captured from a first-person perspective using META smart glasses 
      mounted on a person's head. These images show exactly what the camera wearer sees as they perform daily activities.
      
      Key characteristics of egocentric images:
      - The viewpoint is from the person's head/eye level, looking outward at the world
      - You will often see the person's hands, arms, or objects they are manipulating in the foreground
      - The perspective is naturally shaky and dynamic as the person moves
      - Objects of interest are typically in the center of view where the person is looking
      - Background context shows the environment (kitchen, living room, office, etc.)
      - The camera wearer themselves is NOT visible (except possibly their hands/arms)
      
      The images provided are sequential frames ordered chronologically from earliest to latest, capturing a continuous activity.

      Your task:
      1. Analyze ALL the images together as a temporal sequence.
      2. Identify the SINGLE main activity the camera wearer is performing across the entire sequence.
      3. Focus on what the person's hands are doing and what objects they are interacting with.
      4. Express this activity as one compact label in the format:

      <verb>_<object>

      Formatting rules:
      - Use exactly one verb and one object.
      - Verb: base form, lowercase, no tense 
      - Object: a single simple noun, lowercase, no adjectives 
      - Use only lowercase letters and the underscore `_`. No spaces, commas, or extra words.

      Examples:
      - exercise_stretch,eating_lunch,exercise_treadmill,use_phone,exercise_stretch
      
      IMPORTANT:
      - Return ONLY the activity label token. 
      - Do NOT add explanations, sentences, or quotes, only <verb>_<object>      
      Previous actions based on single frame were : {actions}
      """


def generate_gaze_description(gazes):
    valid_gazes = [
        (g["gaze_x"], g["gaze_y"])
        for g in gazes
        if g["gaze_x"] is not None and g["gaze_y"] is not None
    ]

    gazes_x = [g[0] for g in valid_gazes]
    gazes_y = [g[1] for g in valid_gazes]

    prompt_text = ""
    for i in range(len(gazes_x)):
        prompt_text += f"Frame {i+1}: Gaze({gazes_x[i]}, {gazes_y[i]})"
    return prompt_text


def generate_saliency_map(gazes, h, w, sigma=15):
    saliency = np.zeros((h, w), dtype=np.float32)

    for g in gazes:
        saliency[int(g["gaze_y"]), int(g["gaze_x"])] += 1.0

    saliency = cv2.GaussianBlur(saliency, (0, 0), sigma)
    saliency = saliency / saliency.max()
    saliency = (saliency * 255).astype(np.uint8)

    saliency_colored = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    return saliency_colored


def run_activity_recognition(
    model, processor, images, prompt_text, image_size=IMAGE_SIZE_VLM_INPUT
):
    torch.cuda.empty_cache()

    pil_images = [Image.fromarray(img) for img in images]
    for img in pil_images:
        img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)

    content = []
    for _ in pil_images:
        content.append({"type": "image"})

    content.append({"type": "text", "text": prompt_text})

    conversation = [
        {"role": "user", "content": content},
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=pil_images, text=prompt, return_tensors="pt").to(
        model.device
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=50, do_sample=False, use_cache=False
        )
        result = processor.decode(output[0], skip_special_tokens=True).strip()

    del inputs, output, pil_images, content, conversation
    torch.cuda.empty_cache()

    return result


def test_prompts(
    model,
    processor,
    input_folder,
    output_folder,
    N_frames_for_activity=N_FRAMES_FOR_ACTIVITY,
    image_size=IMAGE_SIZE_VLM_INPUT,
):
    frames_folder = os.path.join(input_folder, "frames")
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg")])

    with open(os.path.join(input_folder, "actions.txt"), "r") as f:
        actions_list = [line.strip() for line in f.readlines()]

    with open(os.path.join(input_folder, "activities.txt"), "r") as f:
        activities_list = [line.strip() for line in f.readlines()]

    annotations = pd.read_csv(os.path.join(input_folder, "annotations.csv"))

    os.makedirs(output_folder, exist_ok=True)

    T = len(frame_files)
    max_activities = (T + N_frames_for_activity - 1) // N_frames_for_activity

    # results = {
    #    "prompt_1": [],
    #    "prompt_2": [],
    #    "prompt_3": [],
    #    "prompt_4": [],
    #    "prompt_5": [],
    #    "prompt_6": []
    # }

    results = {"prompt_6": [], "prompt_7": [], "prompt_8": [], "prompt_9": []}
    for i in tqdm.tqdm(
        range(max_activities), desc="Testing prompts on activity blocks"
    ):
        start = i * N_frames_for_activity
        end = min(start + N_frames_for_activity, T)

        images = []
        for idx in range(start, end):
            img_path = os.path.join(frames_folder, frame_files[idx])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        images = random.sample(images, min(3, len(images)))

        actions_block = actions_list[start:end]
        actions_str = "\n".join(
            [f"{shorten_action(act)}" for j, act in enumerate(actions_block)]
        )

        gazes_block = []
        for idx in range(start, end):
            row = annotations.iloc[idx]
            gazes_block.append({"gaze_x": row["gaze_x"], "gaze_y": row["gaze_y"]})

        gaze_text = generate_gaze_description(gazes_block)

        # try:
        #    prompt_text = prompt_1(actions_str, start, end-1)
        #    result = run_activity_recognition(model, processor, images, prompt_text, image_size)
        #    parsed = parse_llava_output(result)
        #    results["prompt_1"].append(parsed)
        #    print(f"Prompt 1 result: {parsed}")
        # except Exception as e:
        #    results["prompt_1"].append(f"ERROR: {str(e)}")
        #    print(f"Prompt 1 ERROR: {e}")

        # try:
        #    prompt_text = prompt_2(actions_str, start, end-1)
        #    result = run_activity_recognition(model, processor, images, prompt_text, image_size)
        #    parsed = parse_llava_output(result)
        #    results["prompt_2"].append(parsed)
        #    print(f"Prompt 2 result: {parsed}")
        # except Exception as e:
        #    results["prompt_2"].append(f"ERROR: {str(e)}")
        #    print(f"Prompt 2 ERROR: {e}")

        # try:
        #    prompt_text = prompt_3(actions_str, gaze_text, start, end-1)
        #    result = run_activity_recognition(model, processor, images, prompt_text, image_size)
        #    parsed = parse_llava_output(result)
        #    results["prompt_3"].append(parsed)
        #    print(f"Prompt 3 result: {parsed}")
        # except Exception as e:
        #    results["prompt_3"].append(f"ERROR: {str(e)}")
        #    print(f"Prompt 3 ERROR: {e}")

        # try:
        #    h, w = images[0].shape[:2]
        #    saliency_map = generate_saliency_map(gazes_block, h, w)
        #    images_with_saliency = images + [saliency_map]

        #    prompt_text = prompt_4(actions_str, start, end-1)
        #    result = run_activity_recognition(model, processor, images_with_saliency, prompt_text, image_size)
        #    parsed = parse_llava_output(result)
        #    results["prompt_4"].append(parsed)
        #    print(f"Prompt 4 result: {parsed}")
        # except Exception as e:
        #    results["prompt_4"].append(f"ERROR: {str(e)}")
        #    print(f"Prompt 4 ERROR: {e}")

        # try:
        #    prompt_text = prompt_5(actions_str, start, end-1)
        #    result = run_activity_recognition(model, processor, images, prompt_text, image_size)
        #    parsed = parse_llava_output(result)
        #    results["prompt_5"].append(parsed)
        #    print(f"Prompt 5 result: {parsed}")
        # except Exception as e:
        #    results["prompt_5"].append(f"ERROR: {str(e)}")
        #    print(f"Prompt 5 ERROR: {e}")

        try:
            prompt_text = prompt_6(actions_str, start, end - 1)
            result = run_activity_recognition(
                model, processor, images, prompt_text, image_size
            )
            parsed = parse_llava_output(result)
            results["prompt_6"].append(parsed)
            print(f"Prompt 6 result: {parsed}")
        except Exception as e:
            results["prompt_6"].append(f"ERROR: {str(e)}")
            print(f"Prompt 6 ERROR: {e}")

        try:
            prompt_text = prompt_7(actions_str, start, end - 1)
            result = run_activity_recognition(
                model, processor, images, prompt_text, image_size
            )
            parsed = parse_llava_output(result)
            results["prompt_7"].append(parsed)
            print(f"Prompt 7 result: {parsed}")
        except Exception as e:
            results["prompt_7"].append(f"ERROR: {str(e)}")
            print(f"Prompt 7 ERROR: {e}")

        try:
            prompt_text = prompt_8(actions_str, start, end - 1)
            result = run_activity_recognition(
                model, processor, images, prompt_text, image_size
            )
            parsed = parse_llava_output(result)
            results["prompt_8"].append(parsed)
            print(f"Prompt 8 result: {parsed}")
        except Exception as e:
            results["prompt_8"].append(f"ERROR: {str(e)}")
            print(f"Prompt 8 ERROR: {e}")

        try:
            prompt_text = prompt_9(actions_str, start, end - 1)
            result = run_activity_recognition(
                model, processor, images, prompt_text, image_size
            )
            parsed = parse_llava_output(result)
            results["prompt_9"].append(parsed)
            print(f"Prompt 9 result: {parsed}")
        except Exception as e:
            results["prompt_9"].append(f"ERROR: {str(e)}")
            print(f"Prompt 9 ERROR: {e}")

    results_df = pd.DataFrame(
        {
            "block_id": list(range(max_activities)),
            "ground_truth": activities_list[:max_activities],
            # "prompt_1": results["prompt_1"],
            # "prompt_2": results["prompt_2"],
            # "prompt_3": results["prompt_3"],
            # "prompt_4": results["prompt_4"],
            # "prompt_5": results["prompt_5"],
            "prompt_6": results["prompt_6"],
            "prompt_7": results["prompt_7"],
            "prompt_8": results["prompt_8"],
            "prompt_9": results["prompt_9"],
        }
    )

    results_df.to_csv(os.path.join(output_folder, "prompt_comparison.csv"), index=False)

    return results_df


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, processor = load_llava_model(VLM_ANNOTATOR, device)
    print(f"Model loaded on {model.device}")

    output_folder = os.path.join(testing_folder, "testing_prompts")
    results = test_prompts(model, processor, testing_folder, output_folder)

    print(results.to_string())


if __name__ == "__main__":
    main()
