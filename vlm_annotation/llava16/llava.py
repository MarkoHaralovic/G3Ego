from transformers import (
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)
import torch
from PIL import Image


def load_model(
    model_path,
    device="cuda",
    four_bit=False,
    eight_bit=False,
    model_cache_dir=None,
):
    if four_bit or eight_bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=four_bit,
            load_in_8bit=eight_bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        device_map = (
            {"": device} if isinstance(device, str) and "cuda" in device else "auto"
        )

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            low_cpu_mem_usage=True,
            cache_dir=model_cache_dir,
            local_files_only=True,
        )

        model.gradient_checkpointing_enable()

    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=True,
            cache_dir=model_cache_dir,
            local_files_only=True,
        ).to(device)

    processor = LlavaNextProcessor.from_pretrained(
        model_path, cache_dir=model_cache_dir, local_files_only=True, use_fast=True
    )
    model.eval()
    return model, processor


def parse_output(vlm_output):
    parsed = vlm_output

    if "[/INST]" in vlm_output:
        parsed = vlm_output.split("[/INST]")[-1].strip()
    elif "ASSISTANT: " in vlm_output:
        parsed = vlm_output.split("ASSISTANT: ")[-1].strip()
    elif "assistant/n" in vlm_output:
        parsed = vlm_output.split("assistant/n")[-1].strip()
    elif "assistant" in vlm_output:
        parsed = vlm_output.split("assistant")[-1].strip()
    return parsed


def recognize_action_single_frame(model, processor, image_np, prompt_text, image_size=336):
    torch.cuda.empty_cache()

    img = Image.fromarray(image_np)
    img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        result = processor.decode(output[0], skip_special_tokens=True)

    del inputs, output
    torch.cuda.empty_cache()
    return result


def recognize_action_local_window(model, processor, images_np, prompt_text, image_size=336):
    torch.cuda.empty_cache()

    images = [Image.fromarray(img_np) for img_np in images_np]
    for img in images:
        img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)

    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt_text})

    conversation = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        result = processor.decode(output[0], skip_special_tokens=True)

    del inputs, output
    torch.cuda.empty_cache()
    return result


def recognize_activity(model, processor, images_np, prompt_text, image_size=336):
    torch.cuda.empty_cache()

    images = [Image.fromarray(img_np) for img_np in images_np]
    for img in images:
        img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)

    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt_text})

    conversation = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(model.device)

    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=20, do_sample=False, use_cache=False
            )
            result = processor.decode(output[0], skip_special_tokens=True).strip()

        del inputs, output
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        del inputs
        torch.cuda.empty_cache()
        return None

    del images, content, conversation
    torch.cuda.empty_cache()
    return result