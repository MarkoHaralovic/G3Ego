from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
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
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            low_cpu_mem_usage=True,
            cache_dir=model_cache_dir,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=model_cache_dir,
        ).to(device)

    processor = AutoProcessor.from_pretrained(model_path, cache_dir=model_cache_dir)
    model.eval()
    return model, processor


def parse_output(vlm_output):
    if vlm_output is None:
        return "not_annotated"
    return vlm_output.strip()


def _build_messages(images, prompt_text):
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def _generate(model, processor, pil_images, prompt_text, max_new_tokens):
    torch.cuda.empty_cache()
    messages = _build_messages(pil_images, prompt_text)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        result = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    del inputs, output
    torch.cuda.empty_cache()
    return result


def recognize_action_single_frame(model, processor, image_np, prompt_text, image_size=336):
    img = Image.fromarray(image_np)
    img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    return _generate(model, processor, [img], prompt_text, max_new_tokens=100)