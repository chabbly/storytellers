from diffusers import AutoPipelineForImage2Image
import torch
from PIL import Image
import time
import math

# code adapted from https://huggingface.co/spaces/diffusers/unofficial-SDXL-Turbo-i2i-t2i

i2i_pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
)
# t2i_pipe = AutoPipelineForText2Image.from_pretrained(
#     "stabilityai/sdxl-turbo",
#     safety_checker=None,
#     torch_dtype=torch_dtype,
#     variant="fp16" if torch_dtype == torch.float16 else "fp32",
# )

i2i_pipe.to("mps")
i2i_pipe.set_progress_bar_config(disable=True)


def resize_crop(image, size=512):
    image = image.convert("RGB")
    w, h = image.size
    image = image.resize((size, int(size * (h / w))), Image.BICUBIC)
    return image


def predict(init_image, prompt, strength, steps, seed=1231231):
    init_image = resize_crop(init_image)
    generator = torch.manual_seed(seed)
    last_time = time.time()

    if int(steps * strength) < 1:
        steps = math.ceil(1 / max(0.10, strength))

    results = i2i_pipe(
        prompt=prompt,
        image=init_image,
        generator=generator,
        num_inference_steps=steps,
        guidance_scale=0.0,
        strength=strength,
        width=512,
        height=512,
        output_type="pil",
    )
    print(f"Pipe took {time.time() - last_time} seconds")
    nsfw_content_detected = (
        results.nsfw_content_detected[0]
        if "nsfw_content_detected" in results
        else False
    )
    if nsfw_content_detected:
        return Image.new("RGB", (512, 512))
    return results.images[0]
